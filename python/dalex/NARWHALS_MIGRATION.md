# Native dataframe migration (incremental)

The Python package now accepts eager pandas, Polars and Arrow frames. `Explainer.data`
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
| SHAP kernel/tree/linear/deep/gradient wrappers, LIME, model surrogates | yes | explicit NumPy/tensor boundaries of third-party libraries |
| Aspect importance/triplots | yes | correlation/contingency statistics and result tables |
| Arena | yes | labels, attributes, EDA and serialized plot tables |

Classification and regression fairness have cross-backend integration coverage. The optional PPS dependence method is tested separately with Python 3.10/pandas 1.5/ppscore 1.3.
Its upstream pandas<2 bound and pkg_resources requirement prevent support in the primary 3.14 environment.
Deep/Gradient SHAP has required CPU integration jobs for PyTorch on Python 3.14
and TensorFlow on 3.13, with custom native-frame prediction callbacks. The tested
models are single-output dense regressors; complex/multi-input/GPU networks remain
uncertified. Other Narwhals eager backends may validate but are not certified.
Lazy inputs are rejected without collecting them.

Scikit-learn and LightGBM regression/binary-classification tests fit once on
Polars and compare predictions/explanations across pandas, Polars and Arrow.
Native LightGBM prediction uses its 4.7+ Arrow-stream support. Its stream rejects
dictionary categorical columns, so use consistent explicit training encoding or
preprocessing. Older sklearn with Polars 2 also needs positional pipeline selectors
and an explicit numeric transformer; newer named-selector support should not be
assumed on older dependency versions. No training/prediction conversion to pandas
is inserted by DALEX to conceal unsupported model inputs.

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
on Linux and macOS, plus required Linux 3.14t with `PYTHON_GIL=0`.
The regular matrix retains pandas 2, with an additional Python 3.14/pandas 3 job.
Scientific extensions require wheels in CI so unavailable free-threaded wheels
produce a clear dependency-install failure instead of an unbounded source build. Maintainers should explicitly
approve the minimum-version change before release. Pandas remains required at
`>=1.5.3,<4` for result/statistics tables. Pandas 3 compatibility fixes include
copy-on-write array mutation, fairness table reshaping/string group detection and
breakdown plot string labels. Minimum-pandas CI retains 1.5.3 coverage.
The `<4` cap is a precautionary major-version policy, not evidence that pandas 4
breaks DALEX. No pandas 4 release/prerelease/nightly was available when checked
on 2026-10-27. A separate forward-compatibility run tests the available pandas
3.2 development wheel; that does not certify pandas 4 support.

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

### Reproducing Python 3.14t

Polars' PyPI runtime does not supply cp314t wheels. Its conda-forge builds do;
`ci/free-threaded/pixi.toml` and `pixi.lock` pin that interpreter/Polars route on
Linux and Windows. The remaining scientific packages use PyPI wheels, including
pandas 3 (the pandas<3 PyPI releases have no matching cp314t wheel).

```sh
pixi install --locked --manifest-path python/dalex/ci/free-threaded/pixi.toml
pixi run --manifest-path python/dalex/ci/free-threaded/pixi.toml python -m pip install -e ./python/dalex -r python/dalex/ci/requirements.txt
# Set PYTHON_GIL=0 and EXPECT_GIL=0 in the shell before these commands.
pixi run --manifest-path python/dalex/ci/free-threaded/pixi.toml python python/dalex/ci/verify_runtime.py
pixi run --manifest-path python/dalex/ci/free-threaded/pixi.toml python -m pytest -q -ra python/dalex/test
```

The CI job checks the interpreter build and actual GIL state after each scientific
dependency import, then runs the entire suite with `PYTHON_GIL=0`. A separate
`ci/probe_gil_imports.py` starts isolated interpreters with that override removed.
SHAP 0.54.1's `shap._cext` automatically enables the GIL in that probe because it
does not declare free-threading safety. Passing tests with the explicit override
does not establish SHAP C-extension safety under concurrent calls. No third-party
extension safety flags are patched. No free-threading speed claim is made.

Explicitly seeded core explanation streams are local MT19937 generators, retaining
legacy serial values without resetting another thread's NumPy state. Kernel and
Gradient SHAP's global NumPy APIs use a cooperating lock/state scope; sampled
Kernel SHAP resumes the legacy state after background sampling. Threaded tests
compare serial/concurrent values and verify global RNG isolation. Unseeded calls,
arbitrary external RNG consumers and arbitrary model-library concurrency remain
outside that guarantee. See [current review evidence](benchmarks/results/REFINEMENT_VALIDATION.md).

## Benchmark protocol

The benchmark uses one fitted estimator per dataset size,
identical NumPy values, targets, weights, seeds, sampled rows, permutations and
settings for each backend. In the default sklearn-numpy mode, the callback converts
either native frame to NumPy; that conversion is included in the timings.
Native frame construction is also timed. Training and process startup are excluded.
The default `--model sklearn-numpy` preserves the original explicit-NumPy protocol.
`--model sklearn-native` and `--model lightgbm-native` fit on Polars and pass native
frames directly to the model; conversion within the model is still timed. Use
`--sample-limit 100000` to sample all rows through the 100k case rather than the
historical 10k cap. `bench_existing_polars.py` separately measures six explanation
families starting with an existing Polars frame and reports both whole-workflow
and explanation-only time. [Native estimator tables](benchmarks/results/REFINEMENT_BENCHMARKS.md)
retain performance/memory improvements and regressions.

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

Current results: [benchmarks/results/REFINEMENT_VALIDATION.md](benchmarks/results/REFINEMENT_VALIDATION.md).
Earlier snapshots: [benchmarks/results/PHASE2_VALIDATION.md](benchmarks/results/PHASE2_VALIDATION.md).

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
# Refinement notes

The adapter uses `narwhals.stable.v2`, including at the Narwhals 2.0 minimum.
Native gathers and shuffles validate integer positions before narrowing, so an
oversized uint64 cannot wrap into a valid negative index. Observation broadcasts
also cover nested cells, timezone-aware nanosecond timestamps and durations.
Arrow field/table metadata and Decimal precision survive reconstruction.
Arrow feature permutations also reuse the complete reference schema over the
permuted buffers: Narwhals' name-based Arrow column replacement otherwise drops
metadata on the replaced fields. A metadata-sensitive prediction callback covers
this boundary in the regression suite.

Arrow dictionary values are decoded explicitly at NumPy boundaries: the tested
PyArrow conversion can expose an unused dictionary entry underneath a null index.
This is required for missing categorical profile values and Kernel SHAP parity.

Numeric profile grids ignore missing values consistently across backends. This
also fixes legacy pandas quantile grids that previously became all-NaN. An
all-missing numeric column has one NaN split; explicit observation values are
still added when requested. Boolean variables use their observed levels.

Categorical estimator semantics belong to the model: sklearn's OrdinalEncoder
can learn Polars `None` as a level while treating pandas categorical NaN as
missing. Tests include explicit native preprocessing to give these values a
consistent model contract, followed by an explicit NumPy estimator boundary
for compatibility with older sklearn ColumnTransformer versions.

Conditional categorical profiles previously compared numeric/boolean originals
with string intervention labels, which could produce zero predictions. Labels
now match values before formatting, normalize missing sentinels to the grid's
representation and append ` (missing)` if that label collides with a real level.
These are intentional corrections to pre-existing pandas behavior, covered by
mathematical expected-value tests, rather than a claim of unchanged broken output.
Nullable profile columns are assigned positionally to support strict pandas 3.

Older Polars export methods can hold mutable borrows on a frame. Buffer-sharing
clones at prediction and result/NumPy export boundaries isolate simultaneous
calls without converting the feature backend. A numeric target-leakage check
does not export unrelated temporal/nested features just to compare them with y.

[Native backend recipes](NATIVE_BACKEND_RECIPES.md) show direct Polars fitting,
explicit categorical codes/grids and consistent missing-category preprocessing.
The Torch model bridge supports buffer-only inference modules as well as
parameter-based modules; CPU trainable/frozen checks cover both SHAP APIs.

The clone workaround is restricted to Polars 1.x (and older). Polars 2.0/2.1
protect their dataframe storage with internal read/write locks; concurrent export
and default-model prediction probes on ordinary/free-threaded Python did not
reproduce the old borrow error. This avoids a measured wide-frame overhead from
cloning column references for every prediction. It does not certify arbitrary
model internals or prediction options as thread-safe.

Numeric nullable pandas/Arrow storage is normalized only at the external SHAP
matrix boundary. Missing values use NumPy NaN; integer matrices without missing
values retain exact integer precision. LightGBM 4.7 itself rejects pandas
ArrowDtype feature columns in the tested environment. Use native Polars/Arrow or
an explicit model-side conversion rather than an implicit DALEX backend change.

Fairness accepts native Series, Arrow arrays (including chunked dictionaries)
and one-column frames for protected memberships. Reweighting/resampling also
normalize native target vectors at the statistical NumPy boundary. Regression
fairness uses those normalized memberships consistently. Membership strings,
including any missing-group policy, remain the caller's responsibility.

Partial numeric profiles retain an explicitly missing intervention, including
the automatic grid for an entirely missing numeric feature. Conditional profiles
treat that intervention as a separate missing-value bin: only observations with
a missing original value receive weight in that bin. A numeric accumulated-effect
grid containing a missing value raises a clear error because the value has no
numeric ordering; use finite splits or a partial/conditional profile instead.
