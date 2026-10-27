# Remaining-family migration and performance attribution

Native model feature batches now cover LIME, SHAP kernel/tree/linear wrappers,
model surrogates, Aspect importance/triplots, Arena (all 13 plot containers,
background SHAP resources and HTTP custom observations), and regression fairness.
Pandas statistical/output tables and NumPy third-party model boundaries remain
explicit. Native-only predictors assert that callbacks keep their backend.

## Validation

- Local Python 3.14 full suite: 128 passed / 22 inherited skips before adding two
  additional categorical replacement tests. Both added tests pass separately.
- Seventeen remaining-family integration tests pass on Python 3.14 and Python
  3.12 with Narwhals 2.0.0.
- Nineteen adapter tests pass with Python 3.10, pandas 1.5.3, NumPy 1.24.4 and
  Narwhals 2.0.0, including categorical extension-array replacement.
- Seventeen upstream numerical comparisons still have maximum absolute error 0.
- Optional PPS matrices match exactly for pandas and Polars in the compatible
  Python 3.10/pandas 1.5.3/ppscore 1.3.0 environment. Its legacy pkg_resources
  build/runtime needs setuptools<81. PPS is tested separately in minimum-pandas
  CI, not added to the primary scientific dependency stack.
- The inherited 22 skips remain: 18 Aspect/ppscore tests and four Arena tests with
  the historical Ubuntu platform guard. The new tests exercise Aspect and Arena
  independently instead of counting those skips as coverage.

[Implementation CI](https://github.com/brunocarlin/DALEX/actions/runs/40583947744)
and subsequent branch runs archive JUnit results, dependency lists and cold/warm
benchmark artifacts. Consult the run for each supported job's status/counts.
The CI matrix has 130 test passes plus 22 inherited skips when all collected tests
pass; the two additional helper cases account for the difference from the earlier
local full-suite count above.

## Why the original benchmark was slower

1. Fresh-process trials included lazy Polars/Narwhals initialization each time:
   about 13ms initially versus 0.4ms after initialization in the same process.
   Excluded warm-up *workers* warmed OS caches but not the measured worker's runtime.
2. Every permutation unnecessarily unwrapped/revalidated the frame multiple times
   and converted NumPy row indices to Python lists. The new adapter holds one
   Narwhals frame and uses the NumPy index directly. Across six profiled explanations,
   wrapper calls fell from 570 to 210 and cumulative shuffle time from 0.130s to
   0.066s (about half). Profiled elapsed times are attribution, not speed guarantees.
3. Most work is the same sklearn estimator inference (~0.27s at 10k rows).
   Native Polars must still materialize the estimator's NumPy input; a faster
   dataframe operation cannot remove that estimator cost.

`backend-profile-before.json` and `backend-profile-after.json` preserve the raw
source-stamped profiles. `profile_backend_overhead.py` reproduces the attribution.

## Updated measured results

The reported implementation uses identical fitted estimator/data/weights/seeds and
settings within each comparison. Windows 3.14.4 results use ten float64 features,
three rounds, five measured trials per backend, and a 10k importance sample cap.
Ratios are pandas time divided by Polars time; above 1 favors Polars.

| Rows | Cold E2E ratio | Warm E2E ratio | Warm dataframe-only ratio |
|---:|---:|---:|---:|
| 1k | 0.767 | 1.047 | 1.400 |
| 10k | 0.943 | 0.986 | 2.351 |
| 100k | 0.928 | 0.971 | 1.524 |

Cold results still regress. In initialized workers Polars is modestly faster at
1k, approximately tied/slower at 10k, and slightly slower at 100k. These small
end-to-end differences must be interpreted with the IQRs in the raw reports.
Polars process RSS remains higher in these experiments. The JSON/Markdown files
include individual trials, medians/IQR, inference time and sampled absolute/
incremental RSS; sampling every 2ms can miss peaks.

`--warm-runtime` performs an excluded full explanation *inside each measured worker*.
The timed region still constructs a new native dataframe and includes model-boundary
conversion. Warm allocator state changes the RSS baseline; cold and warm memory
numbers should not be interchanged. Training and process startup are excluded.

## Remaining boundaries

Lazy and other eager backends remain uncertified. SHAP Deep/Gradient matrix
boundaries are implemented but TensorFlow/PyTorch integration has not been tested.
PPS's upstream pandas<2 dependency prevents its use in the primary Python 3.14
stack. Free-threaded Python 3.14t is still blocked by unavailable Polars runtime
wheels; its post-dependency GIL check/tests cannot run. The Python >=3.10 proposal
and pandas result/statistics dependencies remain documented for maintainer review.
