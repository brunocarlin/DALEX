# Measured pandas versus Polars benchmark

Python 3.14.4. Ratios above 1 favor Polars; below 1 favor pandas.

| Rows | pandas E2E s (IQR) | Polars E2E s (IQR) | E2E ratio | Adapter ratio | pandas/Polars peak RSS MiB | Max parity error |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.0590 (0.0582-0.0604) | 0.0767 (0.0765-0.0771) | 0.769 | 1.561 | 210.7/214.6 | 0 |
| 10000 | 0.3109 (0.3106-0.3111) | 0.3391 (0.3390-0.3396) | 0.917 | 1.107 | 215.1/221.5 | 0 |
| 100000 | 0.3156 (0.3148-0.3157) | 0.3409 (0.3402-0.3413) | 0.926 | 1.029 | 230.3/240.7 | 0 |

Native frame construction and model conversion/inference are included. Training and process startup are excluded.
One fitted estimator, identical seeds, data, weights and settings per size; alternating backend order; one excluded warm-up per backend.
RSS is sampled every 2 ms in isolated processes and is a lower bound on true peak. No free-threading speed claim is made.
JSON includes every trial, inference time, adapter time, absolute and incremental RSS, dispersion, and package versions.

Measured source revision: `b5e1e1753c3bbfc829d5aaacab13b9bc9487e237`.
