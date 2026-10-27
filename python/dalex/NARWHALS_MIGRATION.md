# Native dataframe migration (incremental)

The Python package now accepts eager pandas and Polars frames. `Explainer.data`
and feature batches passed to the predictor retain their native backend. NumPy
inputs still become pandas, and the existing H2O conversion remains in place.
This is an incremental contribution for ModelOriented/DALEX#575, not a claim that
all DALEX integrations support Polars.

## Supported and pending paths

| API | Native feature batches | Remaining pandas boundary |
|---|---|---|
| prediction / residuals | yes | target/prediction vectors use NumPy |
| permutation importance, including grouped features | yes | result and raw permutation tables |
| model performance | yes | metric and residual result tables |
| residual diagnostics | yes | feature/result table for plotting |
| breakdown / breakdown interactions | yes | observation metadata and contributions |
| DALEX Shapley values (`type='shap'`) | yes | observation metadata and contributions |
| ceteris paribus | yes | profile table after prediction |
| partial, accumulated, conditional profiles | yes | aggregation of profile result tables |
| external SHAP wrapper, LIME, model surrogates | pending: explicit `NotImplementedError` for non-pandas | legacy third-party model adapters |
| Aspect, Arena | pending: explicit `NotImplementedError` for non-pandas | dependence calculations and Arena metadata |

Classification performance, permutation importance and group fairness have
cross-backend integration coverage; regression fairness is not yet certified. Direct use of specialized Aspect internals
is outside the tested native API surface. Other Narwhals eager backends may validate
but have not been certified. Lazy inputs are rejected without collecting them.

Native observation frames must use the same backend and ordered column names as
the explainer. Lists and NumPy observations are constructed using that backend.
Pandas observations retain the existing positional column-renaming behavior.
Non-default pandas indexes, repeated row positions, grouped shuffles and input
immutability are regression tested. Native predictors are not probed with NumPy;
the historical pandas array-acceptance probe remains unchanged.

Polars prediction Series are normalized to NumPy vectors. Invalid multi-dimensional
NumPy prediction outputs retain the existing warning behavior. Unsupported target
objects fail validation rather than triggering dataframe conversion.

## Compatibility change for maintainers

This contribution proposes **Python >=3.10**, dropping 3.9, with `narwhals>=2,<3`.
Python 3.14 is the primary development and benchmark version. CI tests 3.10–3.14
on Linux and macOS, plus experimental Linux 3.14t with `PYTHON_GIL=0`.
Only that experimental entry permits job failure; ordinary builds remain required.
Scientific extensions require wheels in CI so unavailable free-threaded wheels
produce a clear dependency-install failure instead of an unbounded source build. Maintainers should explicitly
approve the minimum-version change before release. Pandas remains required at
`>=1.5.3,<3` for the legacy API and output tables; pandas 3 is not supported here.

No new mandatory Polars dependency is imposed on pandas-only users. Install `dalex[polars]`
(Polars and PyArrow for pandas result materialization) for native Polars inputs.
The test dependencies always install both, so backend tests cannot silently skip.

The default permutation RMSE currently rejects sample weights in existing DALEX.
Use a loss callable accepting `sample_weight` for weighted explanations; this
migration tests that route without changing existing loss semantics.

## Validation and runtime evidence

```sh
python -m pip install -e ./python/dalex -r python/dalex/ci/requirements.txt
python python/dalex/ci/verify_runtime.py
python -m pytest -q -ra python/dalex/test
python python/dalex/benchmarks/bench_variable_importance_backends.py
```

`verify_runtime.py` checks the actual GIL state after importing the scientific
and optional test libraries, and checks `Py_GIL_DISABLED` on Python 3.14t. CI
archives JUnit results and dependency versions for each matrix entry. Upstream
dependency installation/import failures must be reported separately from DALEX
assertion failures; free-threading performance is not inferred from ordinary tests.

The inherited suite skips Aspect tests for ppscore dependency problems and four
Arena server tests behind a `sys.platform.startswith('ubuntu')` guard (standard
Linux reports `linux`, so that guard also skips on Linux). Those skips are not
new backend test skips and are reported rather than counted as passing coverage.

Local Windows 3.14t dependency setup selected pandas 2.3.3 source and failed because
MSVC's `vswhere.exe` was missing. This occurred before importing or testing DALEX.
The Linux CI experiment provides separate evidence about wheel availability there.

## Benchmark protocol

The benchmark uses one fitted HistGradientBoostingRegressor per dataset size,
identical NumPy values, targets, weights, seeds, sampled rows, permutations and
settings for each backend. Its explicit model callback converts either native
frame to NumPy; that conversion is included in inference and end-to-end timings.
Native frame construction is also timed. Training and process startup are excluded.

Trials run in fresh spawned processes, alternate backend order, exclude one warm-up
per backend, and report all trials, medians and interquartile ranges. Both result
and raw permutation losses are checked before reporting ratios. A second region
measures frame construction, sampling and column permutations without inference.
Process RSS is sampled every 2ms and reports absolute peak and increase; sampling
can miss short-lived allocations. These are observational results, not guarantees.

Outputs: `dalex-narwhals-benchmark.json` and `dalex-narwhals-benchmark.md`.
Excluded warm-up trials use fresh processes and warm OS caches rather than persistent worker state.
Defaults: 1k, 10k, 100k rows, ten float64 features, five measured trials, three
permutation rounds, at most 10k sampled rows. CI uses two sizes and three trials.
Ratios above one favor Polars; regressions below one must also be reported.

Measured results and validation details: [benchmarks/results/VALIDATION.md](benchmarks/results/VALIDATION.md).
