"""Compare private pandas helper contracts and index-sensitive public callbacks.

The reference package is unmodified and imported in a separate interpreter.
Pickles retain indexes, extension dtypes and heterogeneous metadata cells.
"""
import argparse
import os
from pathlib import Path
import pickle
import subprocess
import sys
import json
from contextlib import redirect_stdout
from io import StringIO


def indexed_predict(model, frame):
    return frame.to_numpy().sum(axis=1) + frame.index.to_numpy() / 10


def residual(model, frame, target):
    return target - model.predict(frame)


def capture(root):
    sys.path.insert(0, str(Path(root).resolve()))
    import numpy as np
    import pandas as pd
    import dalex as dx
    from sklearn.linear_model import LinearRegression
    from dalex.predict_explanations._break_down import utils as bd
    from dalex.predict_explanations._shap import utils as shap
    from dalex.predict_explanations._ceteris_paribus import utils as cp
    from dalex.model_explanations._aggregated_profiles import utils as ap
    from dalex.aspect._predict_aspect_importance import utils as pai
    from dalex.aspect._predict_triplot import utils as pt
    from dalex.aspect._model_triplot import utils as mt
    from dalex.aspect import utils as dependence
    if not hasattr(np, 'in1d'):
        np.in1d = np.isin
    rng = np.random.default_rng(682)
    data = pd.DataFrame(rng.normal(size=(32, 3)), columns=['z', 'a', 'b'], index=np.arange(32) * 3 + 7)
    y = data.z.to_numpy() * 2 + data.a.to_numpy() / 3 + np.sin(data.b.to_numpy())
    model = LinearRegression().fit(data, y)
    exp = dx.Explainer(model, data, y, residual_function=residual, label='model', verbose=False)
    observation = data.iloc[[2]]
    result = {}

    def save(name, value):
        if isinstance(value, tuple):
            for index, item in enumerate(value):
                save(f'{name}/{index}', item)
        else:
            result[name] = value

    mixed = data.assign(cat1=['low', 'high'] * 16, cat2=['yes', 'no', 'no', 'yes'] * 8)
    save('categorical-association', dependence.calculate_cat_assoc_matrix(mixed, ['cat2', 'cat1'], len(mixed)))
    save('categorical-numeric-association', dependence.calculate_cat_num_assoc_matrix(
        mixed, ['cat2', 'cat1'], ['a', 'z'], len(mixed)))

    splits = {'z': np.array([-1., 0., 1.]), 'a': np.array([-.5, .5])}
    save('cp-single', cp.single_variable_profile(exp.predict_function, exp.model, data.iloc[:3], 'z', splits['z']))
    save('cp-many', cp.calculate_variable_profile(exp.predict_function, exp.model, data.iloc[:3], splits, 1, False))
    save('cp-both', cp.calculate_ceteris_paribus(exp, data.iloc[:3], splits, y[:3], 1, False))
    for kind in ['1d', '2d']:
        for order in [None, np.array([2, 0, 1]), np.array(['b', 'z', 'a'])]:
            for distributions in [False, True]:
                save(f'bd-{kind}-{order}-{distributions}', bd.local_interactions(
                    exp, observation, 1., kind, order, distributions))
    one = bd.calculate_1d_changes(exp, observation)
    save('bd-one', one)
    indices = pd.DataFrame({'ind1': [1, 2, 2], 'ind2': [0, 0, 1]})
    save('bd-two', bd.calculate_2d_changes(exp, observation, indices, one - exp.y_hat.mean()))
    feature_path = pd.DataFrame({'diff': one, 'ind1': np.arange(3)})
    for order in [None, np.array([2, 0, 1]), np.array(['b', 'z', 'a'])]:
        ordered = bd.create_ordered_path(feature_path, one.index, '1d', order)
        save(f'bd-order-{order}', ordered)
        save(f'bd-contributions-{order}', bd.calculate_contributions_along_path(
            exp, observation, ordered, exp.y_hat.mean(), exp.predict(observation), '1d', True))
    tied = pd.DataFrame({'adiff_norm': [1., 2., 1., 2., 1., 2.] * 5}, index=[f'x{i}' for i in range(30)])
    save('bd-floating-tie-order', bd.create_ordered_path(tied, tied.index, '2d', None))
    for path in [None, 'average', np.array([2, 0, 1])]:
        for distributions in [False, True]:
            np.random.seed(42)
            save(f'shap-{path}-{distributions}', shap.shap(exp, observation, path, distributions, 3, 1, 42))
    save('shap-processes', shap.shap(exp, observation, 'average', False, 3, 2, 42))
    save('shap-single', shap.get_single_random_path(exp.predict_function, exp.model, data, exp.label,
                                                  observation, np.array([2, 0, 1]), 7))
    save('shap-iterate', shap.iterate_paths(exp.predict_function, exp.model, data, exp.label,
                                         observation, 3, 7, np.random.RandomState(42)))
    save('shap-distributions', shap.calculate_yhats_distributions(exp))
    profiles = cp.calculate_variable_profile(exp.predict_function, exp.model, data.iloc[:5], splits, 1, False)
    profiles['_label_'] = exp.label
    profiles, variables = ap.prepare_numerical_categorical(profiles, None, 'numerical')
    save('ap-variables', variables)
    profiles = ap.prepare_x(profiles, 'numerical')
    save('ap-prepared', profiles)
    profiles['group'] = np.where(profiles['_ids_'].to_numpy() % 2 == 0, 'even', 'odd')
    for kind in ['partial', 'conditional', 'accumulated']:
        for groups in [[], ['group']]:
            save(f'ap-{kind}-{groups}', ap.aggregate_profiles(profiles.copy(), exp.y_hat.mean(), kind,
                                                             groups, True, .25, False))
            save(f'ap-split-{kind}-{groups}', ap.split_over_variables_and_labels(
                profiles.loc[profiles['_vname_'] == 'z'].copy(), kind, groups, .25))
    groups = {'za': np.array(['z', 'a']), 'b': np.array(['b'])}
    for count in [None, 1]:
        for sampling in ['default', 'binom']:
            save(f'aspect-default-{count}-{sampling}', pai.calculate_predict_aspect_importance(
                exp, observation, groups, 20, count, sampling, 2, 42))
    save('aspect-one-group', pai.calculate_predict_aspect_importance(
        exp, observation, {'all': np.array(data.columns)}, 20, None, 'default', 2, 42))
    for processes in [1, 2]:
        np.random.seed(42)
        save(f'aspect-shap-{processes}', pai.calculate_shap_predict_aspect_importance(
            exp, observation, groups, 20, 3, processes, 42))
    save('aspect-path', pai.get_single_random_path(exp.predict, data.iloc[:20], observation,
                                                groups, np.array([1, 0]), 7))
    aspect = dx.Aspect(exp)
    save('model-hierarchy', mt.calculate_model_hierarchical_importance(
        aspect, 'rmse', 'variable_importance', 20, 3, 1, 42))
    save('model-single', mt.calculate_single_variable_importance(
        aspect, 'rmse', 'variable_importance', 20, 3, 1, 42))
    for kind in ['default', 'shap']:
        np.random.seed(42)
        save(f'predict-hierarchy-{kind}', pt.calculate_predict_hierarchical_importance(
            aspect, observation, kind, 20, 3, 'default', 2, 1, 42))
        np.random.seed(42)
        save(f'predict-single-{kind}', pt.calculate_single_variable_importance(
            aspect, observation, kind, 20, 3, 'default', 2, 1, 42))
        indexed = dx.Explainer(model, data, y, predict_function=indexed_predict,
                                model_type='regression', verbose=False, label='indexed')
        save(f'indexed-public-aspect-{kind}', dx.Aspect(indexed).predict_parts(
            observation, variable_groups=groups, type=kind, N=20, B=3, random_state=42).result)
    from dalex.fairness import GroupFairnessClassification, GroupFairnessRegression
    protected = np.tile(['c', 'a', 'b', 'a'], 8)
    for regression in [False, True]:
        for edit in ['none', 'nan', 'nullable', 'duplicate']:
            value = (GroupFairnessRegression(y, exp.y_hat, protected, 'a', 'model', verbose=False) if regression else
                     GroupFairnessClassification((y > np.median(y)).astype(int),
                         1 / (1 + np.exp(-exp.y_hat)), protected, 'a', 'model', verbose=False))
            metric = 'independence' if regression else 'TPR'
            if edit != 'none':
                value.result.index.name = 'edited-group'
                if edit == 'nan':
                    value.result.loc['c', metric] = np.nan
                elif edit == 'nullable':
                    value.result[metric] = value.result[metric].astype('Float64')
                    value.result.loc['c', metric] = pd.NA
                else:
                    value.result = pd.concat([value.result.iloc[::-1], value.result.loc[['c']]])
            buffer = StringIO()
            with redirect_stdout(buffer):
                value.fairness_check(epsilon=.85, verbose=True)
            save(f'fairness-console-{regression}-{edit}', buffer.getvalue())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture-root')
    parser.add_argument('--reference-root')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.capture_root:
        output.write_bytes(pickle.dumps(capture(args.capture_root)))
        return
    roots = [('reference', Path(args.reference_root).resolve()), ('current', Path(__file__).resolve().parents[1])]
    captures = []
    for name, root in roots:
        path = output.with_suffix(f'.{name}.pkl')
        subprocess.run([sys.executable, __file__, '--capture-root', str(root), '--output', str(path)], check=True,
                       env=dict(os.environ, PYTHONHASHSEED='0'))
        captures.append(pickle.loads(path.read_bytes()))
    import pandas as pd
    import numpy as np
    import dalex as dx
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'benchmarks'))
    from _benchmark_utils import source_identity

    def compare(a, b):
        if isinstance(a, pd.DataFrame):
            pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)
        elif isinstance(a, pd.Series):
            pd.testing.assert_series_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-12)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for key in a:
                compare(a[key], b[key])
        elif a is None or isinstance(a, str):
            assert a == b
        elif np.asarray(a).dtype.kind in 'biufc':
            np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
        else:
            np.testing.assert_array_equal(a, b)

    reference, current = captures
    assert reference.keys() == current.keys()
    for name in reference:
        try:
            compare(reference[name], current[name])
        except (AssertionError, TypeError) as error:
            raise AssertionError(f'Legacy adapter mismatch: {name}') from error
    report = {'passed': True, 'outputs': list(reference), 'output_count': len(reference),
              'current_source': source_identity(dx), 'tolerance': {'rtol': 1e-12, 'atol': 1e-12},
              'scope': 'private helper values, indexes, dtypes and metadata; index-sensitive public Aspect callbacks',
              'reference_runtime_shim': 'removed numpy.in1d spelling uses numpy.isin on 1-D feature lists'}
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
