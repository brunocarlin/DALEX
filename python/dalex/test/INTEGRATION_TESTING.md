# Backend integration testing

The backend migration includes analytical unit tests, real-model integration
tests, the existing DALEX suite, upstream numerical comparisons and isolated
benchmarks. These checks answer different questions; backend parity alone would
not catch a common error shared by both implementations.

## Run the complete suite

Install the package and CI requirements from the repository root:

```sh
python -m pip install -e ./python/dalex -r python/dalex/ci/requirements.txt
python python/dalex/ci/verify_runtime.py
python -m pytest -q -ra --timeout=120 --timeout-method=thread python/dalex/test
```

Use `MPLBACKEND=Agg` for headless execution. CI fixes model/BLAS/OpenMP threads
to one. Backend libraries and the real estimators are required dependencies of
the test environment; their integration cases do not silently skip when absent.
PPS runs separately with its upstream-compatible pandas<2 dependencies.

## Workflow coverage

| Test module | Integration boundary |
|---|---|
| `test_native_profile_workflows.py` | Fit sklearn/LightGBM regression and binary pipelines once on Polars; compare pandas/Polars/Arrow importance, contributions, CP-to-aggregate, diagnostics, performance, Aspect/triplots and fairness; exercise exports, plots, pickle and legacy mutations |
| `test_polars_output_pipelines.py` | Model-owned imputation/encoding and Polars-output preprocessing with sklearn/LightGBM prediction |
| `test_native_arena_workflows.py` | Model registration, all plot families, background threads/resources, incremental Shapley reductions, serialization and writable public resource dictionaries |
| `test_arena_missing_json.py` | HTTP custom observations and strict JSON with missing/nonfinite display values |
| `test_native_surrogate_workflows.py` | Real models through LIME and sklearn tree/linear surrogates, prediction adapters, exports, pickle and matplotlib plots |
| `test_native_label_workflows.py` | Custom labels across nine explanation methods, lazy native storage, original pandas index/column layouts and pickle |
| `test_native_estimators.py`, `test_native_tree_wrappers.py` | Polars-trained real estimators and external SHAP tree boundaries |
| `test_native_categorical_estimators.py`, `test_missing_native_estimators.py` | Explicit model-owned categorical/missing preprocessing and unsupported-input policies |
| `test_nullable_shap_boundaries.py`, `test_external_shap_seed.py` | Nullable numeric matrix adapters and sampled Kernel SHAP random-state behavior |
| `test_seeded_concurrency.py` | Serial/concurrent seeded explanation parity and caller RNG isolation |
| `test_native_datasets.py` | Every bundled CSV loader, parser parity, native backends and explicit row identifiers |

`native_computation_only` guards reject pandas DataFrame/Series construction and
Narwhals-to-pandas conversion inside the guarded native computation. Public
legacy views and plotting are exercised after that scope. Thus the tests check
both backend preservation and the compatible pandas publication boundary.

Independent expected-value tests cover typed UInt64 grids above 2**63, exact
nanosecond values, finite Decimal interventions, native null versus NaN, missing/unused groups, mutation and
metadata. `test_native_profile_grid_scaling.py` also guards against exporting
every reference cell to Python merely to construct an automatic numeric grid.
Aspect/dendrogram tests reject browser-renderer calls and preserve the original
hover vertices and endpoints, including constant panels.
`ci/compare_legacy.py` compares numerical outputs with an unmodified
upstream checkout, using a documented NumPy spelling shim for the old reference.
`ci/check_legacy_pickles.py` loads real upstream explanation pickles in the new
package, checks their pandas tables, restores the documented Explainer dump
protocol and reuses old CP/Aspect objects. A required Python 3.14 CI job checks
both scripts against a pinned upstream revision.

## Runtime matrix and external libraries

`Python-check.yaml` runs the full suite on Python 3.10–3.14 on Linux/macOS,
Windows 3.14, pandas 3, minimum pandas, and Polars 1/Narwhals 2. The required
3.14t job checks the actual disabled GIL state after each dependency import.
Separate required CPU jobs exercise PyTorch and TensorFlow Deep/Gradient SHAP;
those tests live under `integration_tests/`, outside the ordinary test directory.

Forced-disabled-GIL passing tests do not certify arbitrary third-party compiled
extension concurrency. The isolated automatic-GIL probe records dependencies
that re-enable it without the override. Tested neural models are single-output
dense regressors; arbitrary architectures, GPUs, multiclass models, other
dataframe engines and lazy/distributed execution are not certified.

Benchmarks run separately from tests, using identical saved models, values,
seeds and explanation settings. Source/model fingerprints, every trial,
numerical errors, elapsed time and RSS belong in the report. A speed ratio is
not a substitute for integration correctness, and regressions remain visible.

## Native plotting follow-up

`test_native_plot_preparation.py` exercises native plot preparation for importance,
diagnostics, ECDF/ROC/lift, Break Down/interactions and DALEX Shapley, together with
real Tree/Kernel SHAP metadata fitting, exports, plotting inputs and serialization.
It guards legacy table access and conversions, compares all figure fields across
pandas/Polars/Arrow at 1e-12, and checks edited views, mixed feature metadata, large
integers, positional sampling and unequal/small lift datasets. KernelExplainer's
upstream local closure is not claimed to be pickleable.

`ci/compare_plots.py` independently compares 53 complete figure payloads with the
pinned, unmodified upstream pandas implementation. This runs in the required
upstream-compatibility job; test equality covers layout and hover text as well as
numerical traces. It complements the existing 27 explanation-output comparisons.

## Aggregated-profile plotting follow-up

`test_native_aggregate_plotting.py` adds 99 cases for partial, conditional and
accumulated line/bar figures, multiple models, categorical profiles, grouping,
mutable pandas views, exact large integer axes, missing/nullable numeric groups, nanosecond/timezone captions and native
fit-to-plot construction guards. The pandas-prepared raw CP overlay is exercised
for compatibility but excluded from the native construction guarantee.
`ci/compare_plots.py` now also compares 30 profile figures with unchanged upstream
(83 complete figure comparisons overall).
