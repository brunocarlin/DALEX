# Migration evidence

Start with [second refinement review](ROUND2_REVIEW.md) and
[latest benchmark tables](ROUND2_BENCHMARKS.md). They identify the tested
implementation commits and distinguish source validation, installed-package
validation, dependency limitations and performance measurements.

Raw JSON is generated measurement output. GitHub collapses it by default in the
PR diff; it remains available to expand or download. Every trial and its summary
statistics are retained. The `round2/` directory contains the latest experiments;
`round2-validation.json` records source/package/CI validation. The `refinement/`
directory retains the preceding experiments and standalone dependency probes.

Earlier reports in this directory are historical, with their own source commits
and dependency versions. In particular, `NATIVE_ESTIMATOR_*`, `PHASE2_VALIDATION`,
`REVIEW_READINESS` and `VALIDATION` describe earlier implementations and test
counts. Their measurements are not silently reassigned to the latest source.

The benchmark entry points are [bench_existing_polars.py](../bench_existing_polars.py)
(data already held in Polars), [bench_variable_importance_backends.py](../bench_variable_importance_backends.py)
(importance and independent dataframe/inference attribution), and
[profile_backend_overhead.py](../profile_backend_overhead.py). Run them
from the package directory, or use repository-relative paths from its root.
Fit models and construct original inputs outside timing only where the stated
protocol explicitly excludes them. Keep benchmark workers isolated from test
jobs, fix threads/seeds/settings, and report regressions alongside gains.
