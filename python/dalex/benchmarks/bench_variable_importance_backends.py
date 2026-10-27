"""Reproducible end-to-end and dataframe-only trials; ratios may favor either backend.

Each trial runs in a fresh spawned process. Timed regions include constructing the
native frame from identical NumPy values; model training and process startup are
excluded. RSS is sampled every 2ms (an observational lower bound on true peak).
"""
import argparse
import gc
import importlib.metadata
import json
import multiprocessing as mp
import platform
import sys
import threading
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import polars as pl
import psutil
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

import dalex as dx
from dalex._dataframe import shuffle_columns, take_rows


class RSS:
    def __enter__(self):
        self.process = psutil.Process()
        self.baseline = self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.stop.wait(.002):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


class TimedPredict:
    def __init__(self):
        self.seconds = 0.

    def __call__(self, model, frame):
        start = perf_counter()
        # The identical fitted estimator consumes NumPy on both backends.
        # This explicit model boundary conversion is included in measured time.
        values = frame if isinstance(frame, np.ndarray) else frame.to_numpy()
        result = model.predict(values)
        self.seconds += perf_counter() - start
        return result


def trial(queue, backend, values, y, weights, model, rounds):
    try:
        with threadpool_limits(limits=1):
            names = [f'x{i}' for i in range(values.shape[1])]
            def make_frame():
                return (pd.DataFrame(values.copy(), columns=names) if backend == 'pandas'
                        else pl.DataFrame(values.copy(), schema=names))
            gc.collect()
            predict = TimedPredict()
            with RSS() as rss:
                start = perf_counter()
                frame = make_frame()
                explainer = dx.Explainer(model, frame, y, weights=weights, predict_function=predict,
                                         model_type='regression', verbose=False, precalculate=False)
                result = explainer.model_parts(N=min(len(y), 10_000), B=rounds,
                                               random_state=42, keep_raw_permutations=True,
                                               loss_function=weighted_rmse)
                elapsed = perf_counter() - start
            end_to_end = {'seconds': elapsed, 'inference_seconds': predict.seconds,
                          'peak_rss_bytes': rss.peak, 'rss_increase_bytes': max(0, rss.peak-rss.baseline)}
            # Independently measure adapter work including native construction.
            del frame, explainer
            gc.collect()
            rng = np.random.default_rng(42)
            positions = rng.choice(len(y), min(len(y), 10_000), replace=False)
            permutations = [rng.permutation(len(positions)) for _ in range(rounds)]
            with RSS() as rss:
                start = perf_counter()
                frame = take_rows(make_frame(), positions)
                for order in permutations:
                    for name in names:
                        shuffled = shuffle_columns(frame, name, order)
                dataframe_seconds = perf_counter()-start
            queue.put({'backend': backend, 'end_to_end': end_to_end,
                       'dataframe': {'seconds': dataframe_seconds, 'peak_rss_bytes': rss.peak,
                                     'rss_increase_bytes': max(0, rss.peak-rss.baseline)},
                       'outcome': result.result.sort_values('variable').dropout_loss.tolist(),
                       'raw': result.permutation.sort_index(axis=1).to_numpy().tolist()})
    except BaseException:
        queue.put({'error': traceback.format_exc()})


def weighted_rmse(y, prediction, sample_weight=None):
    return np.sqrt(np.average((y-prediction)**2, weights=sample_weight))


def summary(values):
    return {'median': float(np.median(values)), 'q25': float(np.quantile(values, .25)),
            'q75': float(np.quantile(values, .75)), 'min': min(values), 'max': max(values)}


def run(sizes, n_features, repetitions, rounds):
    report = {'environment': {'python': sys.version, 'platform': platform.platform(),
              'versions': {name: importlib.metadata.version(name) for name in
                           ('dalex','numpy','pandas','polars','narwhals','scipy','scikit-learn','psutil')}},
              'method': {'repetitions': repetitions, 'rounds': rounds, 'features': n_features,
                         'seed_data': 123, 'seed_explanation': 42, 'sample_limit': 10_000,
                         'threads_estimator': 1, 'rss_poll_seconds': .002,
                         'conversion_included': True, 'warmup_trials_per_backend': 1,
                         'memory_note': 'Sampled process RSS, including interpreter/dependencies; peak may be underestimated.'},
              'datasets': []}
    context = mp.get_context('spawn')
    for n in sizes:
        rng = np.random.default_rng(123)
        values = rng.normal(size=(n, n_features))
        y = values[:, :min(5,n_features)].sum(axis=1) + rng.normal(size=n)*.1
        weights = np.linspace(1, 2, n)
        with threadpool_limits(limits=1):
            model = HistGradientBoostingRegressor(max_iter=20, random_state=42).fit(values, y)
        trials = []
        expected = None
        max_error = 0.
        for repeat in range(-1, repetitions):
            for backend in (('pandas','polars') if repeat % 2 == 0 else ('polars','pandas')):
                queue = context.Queue()
                worker = context.Process(target=trial, args=(queue,backend,values,y,weights,model,rounds))
                worker.start()
                item = queue.get(timeout=600)
                worker.join(timeout=30)
                if worker.exitcode != 0 or 'error' in item:
                    raise RuntimeError(item)
                raw = np.array(item.pop('raw'))
                outcome = np.array(item.pop('outcome'))
                if expected is None:
                    expected = (raw, outcome)
                for actual, reference in zip((raw,outcome),expected):
                    np.testing.assert_allclose(actual, reference, atol=1e-10, rtol=1e-10)
                    max_error = max(max_error, float(np.max(np.abs(actual-reference))))
                if repeat >= 0:
                    item['repetition'] = repeat
                    trials.append(item)
        stats = {}
        for backend in ('pandas','polars'):
            items = [t for t in trials if t['backend']==backend]
            stats[backend] = {phase: {key: summary([t[phase][key] for t in items])
                              for key in items[0][phase]} for phase in ('end_to_end','dataframe')}
        ratios = {phase: stats['pandas'][phase]['seconds']['median']/stats['polars'][phase]['seconds']['median']
                  for phase in ('end_to_end','dataframe')}
        report['datasets'].append({'rows':n,'max_absolute_parity_error':max_error,
                                   'passed_numerical_parity':True,'trials':trials,'summary':stats,
                                   'pandas_time_over_polars_time':ratios})
        print(f'{n} rows: pandas/Polars time ratios {ratios}; max parity error {max_error}', flush=True)
    Path('dalex-narwhals-benchmark.json').write_text(json.dumps(report,indent=2)+'\n')
    lines = ['# Measured pandas versus Polars benchmark', '',
             'Python '+platform.python_version()+'. Ratios above 1 favor Polars; below 1 favor pandas.', '',
             '| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for item in report['datasets']:
        a,b = (item['summary'][k]['end_to_end'] for k in ('pandas','polars'))
        def timing(x):
            t=x['seconds']; return f"{t['median']:.4f} ({t['q25']:.4f}-{t['q75']:.4f})"
        ratios=item['pandas_time_over_polars_time']
        lines.append(f"| {item['rows']} | {timing(a)} | {timing(b)} | {ratios['end_to_end']:.3f} | {ratios['dataframe']:.3f} | {a['peak_rss_bytes']['median']/2**20:.1f}/{b['peak_rss_bytes']['median']/2**20:.1f} | {item['max_absolute_parity_error']:.3g} |")
    lines += ['', 'Native frame construction and model conversion/inference are included. Training and process startup are excluded.',
              'One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.',
              'RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.',
              'JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.']
    Path('dalex-narwhals-benchmark.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sizes', nargs='+', type=int, default=[1000,10000,100000])
    parser.add_argument('--features', type=int, default=10)
    parser.add_argument('--repetitions', type=int, default=5)
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    run(args.sizes,args.features,args.repetitions,args.rounds)
