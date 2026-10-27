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
| SHAP kernel/tree/linear wrappers, LIME, model surrogates | yes | explicit NumPy input/output boundaries of the third-party libraries |
| Aspect importance/triplots | yes | correlation/contingency statistics and result tables |
| Arena | yes | labels, attributes, EDA and serialized plot tables |

Classification and regression fairness have cross-backend integration coverage. The optional PPS dependence method still depends on ppscore and its pandas compatibility.
Deep/Gradient SHAP matrix boundaries are implemented, but TensorFlow/PyTorch integration
has not been certified by this test matrix. Other Narwhals eager backends may validate
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

## Remaining-family migration and performance attribution

LIME and kernel SHAP require NumPy matrices. Their callbacks now explicitly restore
that matrix to the explainer's native backend before invoking the original model.
Model-specific SHAP explainers consume their estimator directly. Surrogate models
train on the selected native feature frame. No lazy frame is collected.

Aspect keeps feature sampling and masked replacement native; its statistical
correlation/contingency matrices remain pandas-backed to preserve their numerical
semantics. Custom dependence callables receive the native input. Arena retains the
native Explainer, with a pandas copy for observation labels/EDA metadata only;
observation and background SHAP callbacks restore or retain native feature batches.
The HTTP custom-observation route and all 13 plot container families have integration
coverage, including numeric/categorical fairness and native SHAP resources.

The first implementation repeatedly unwrapped and revalidated selected columns and
converted row indices into Python lists for each permutation. The next revision
keeps a single Narwhals wrapper and gathers using the NumPy index vector directly.
On the 10k-row cProfile workload, wrapper calls fell from 570 to 210 across six
explanations and cumulative shuffle time fell from 0.130s to 0.066s. Those are
profiled attribution measurements, not isolated end-to-end speedup claims.

Most wall time in that workload is the same sklearn histogram-tree inference
(about 0.27s per explanation on either backend). Matrix materialization/model
validation adds backend-dependent overhead; replacing pandas cannot accelerate
the estimator itself. Small-column permutation work also differs from large
relational queries where Polars often helps. Use `benchmarks/profile_backend_overhead.py`
for attribution and the isolated-process benchmark for performance claims.

The cold benchmark deliberately starts each measured explanation in a fresh process.
The first Narwhals/Polars initialization cost about 13ms in the profile, versus
about 0.4ms on subsequent explanations in that process. The original excluded
warm-up workers did not warm the measured worker's Python/runtime caches. Use
`--warm-runtime` to perform an excluded explanation inside each measured worker;
its actual timed region still includes constructing a fresh native input frame.
Cold-start and warm-runtime results are reported separately, including memory:
warm allocator caches may change both absolute and incremental RSS.
