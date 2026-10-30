# Full Narwhals migration: implementation contract and progress

This is the expanded work following the native-feature migration. It is **in
progress**, and the earlier source-stamped validation reports do not certify new
implementation commits automatically. PR #587 remains draft for manual review.

## Compatibility contract

- Existing pandas inputs, default return types, column labels, indexes, plotting
  methods and writable `.result` tables continue to work. Native feature batches
  must not become pandas on their way to a predictor.
- Dataframe computation uses `narwhals.stable.v2`; numerical vector algorithms
  may use NumPy/SciPy. Explicit estimator/tensor and serialization boundaries
  remain separate from dataframe manipulation.
- Public pandas result tables can be materialized lazily. Once read, their
  mutations are authoritative for subsequent exports and serialization.
- `explanation.result_as()` exports an independent table in its computation
  backend. `result_as('pandas'|'polars'|'pyarrow')` requests an explicit backend.
  `table='permutation'` exports raw importance draws. SHAP's non-tabular arrays
  retain their existing representation and support only the native export.
- CP's native `result_as(partition=variable)` retains each original feature's
  scalar type. Whole-table native exports reject partitions with incompatible
  scalar types or unsafe integer promotion instead of silently stringifying or
  rounding. The combined, writable legacy pandas table remains available.
- Native grouped-profile tables retain typed grouping columns; the legacy
  pandas projection encodes those memberships into its existing string labels.
  Once a pandas projection is materialized it is authoritative, including its
  column layout, for subsequent exports.
- Arbitrary Python/non-string labels that cannot be physical native column names
  remain at an explicit pandas compatibility boundary, rather than being changed
  to strings silently. Default installation behavior remains compatible; moving
  pandas out of required dependencies is a separate packaging decision until
  no-pandas runtime coverage and default-output behavior have been demonstrated.

## Evidence and completion gates

The following are gates, not claims that all have been completed:

1. Native storage and reductions for importance, contributions, profiles,
   performance/diagnostics, Aspect, fairness and Arena.
2. Plotting/data preparation and dataset loaders use explicit backend boundaries.
3. Compatible pandas results and native exports are tested for missing values,
   integer precision, mixed/temporal data, labels, indexes and mutation.
4. Real fitted sklearn/LightGBM preprocessing pipelines run end-to-end across
   pandas, Polars and Arrow, with conversion guards on native computation.
5. CP-to-aggregate, explanation-to-plot, serialization and Arena/fairness/Aspect
   integration tests complement analytical unit tests and upstream comparisons.
6. The complete suite, minimum dependencies, current/nightly pandas, installed
   wheel, neural jobs and actual-disabled-GIL 3.14t CI are revalidated.
7. Benchmarks identify exact source/model fingerprints and retain performance,
   memory and numerical regressions as well as improvements.
8. A final inventory classifies remaining pandas operations as public
   compatibility, external dependency boundaries or unfinished migration work.

## Current implementation

- Shared result-table storage supports lazy legacy pandas views, explicit native
  exports, in-place pandas edits, independent exports and old pickle state.
- Permutation repetitions and their means run through Narwhals, returning native
  result/raw tables until their legacy public attributes are requested.
- CP prediction/profile-grid construction and partial, conditional and
  accumulated-profile computation run through typed native partitions. Group
  display labels and legacy row indexes are formatted at the pandas boundary.
- Analytical kernel tests and real sklearn/LightGBM pipeline integrations cover
  native model-to-CP-to-aggregate flows, missing groups, native exports, legacy
  result mutation, pickle, plots and nanosecond categorical levels. Guards reject
  pandas DataFrame/Series construction during native computation.
- Other remaining areas above are still being implemented and audited.

## External migration precedents

[Plotly 6](https://plotly.com/python/v6-migration/) adopted Narwhals for native
pandas/Polars/Arrow dataframe processing. Its change illustrates using a shared
internal dataframe layer while retaining existing pandas usage.

[Pandera's Narwhals backend](https://pandera.readthedocs.io/en/latest/narwhals_backend.html)
keeps its existing public imports and offers opt-in backend activation. This
supports separating internal migration from changes to a library's public API;
DALEX's specific API and numerical semantics still require its own tests.

[Narwhals' stable API policy](https://narwhals-dev.github.io/narwhals/backcompat/)
provides a versioned namespace for compatibility. The declared 2.0 minimum is
tested explicitly; newer convenience methods cannot be assumed available there.
