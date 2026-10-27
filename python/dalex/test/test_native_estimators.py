"""Models trained on Polars must retain native explanation prediction batches."""
import numpy as np
import pandas as pd
import polars as pl
import pytest

import dalex as dx


@pytest.mark.parametrize("estimator", ["sklearn_regression", "sklearn_classification", "lightgbm_regression", "lightgbm_classification"])
@pytest.mark.parametrize("family", ["importance", "breakdown", "shap", "profile", "performance"])
def test_polars_trained_estimator_backend_parity(estimator, family):
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier

    rng = np.random.default_rng(123)
    data = pl.DataFrame(rng.normal(size=(80, 3)), schema=["x", "z", "u"])
    continuous = 2 * data["x"].to_numpy() - data["z"].to_numpy() + rng.normal(size=80) * .1
    classifier = estimator.endswith("classification")
    y = (continuous > 0).astype(int) if classifier else continuous
    if estimator.startswith("lightgbm"):
        import lightgbm as lgb
        cls = lgb.LGBMClassifier if classifier else lgb.LGBMRegressor
        model = cls(n_estimators=12, n_jobs=1, random_state=42, verbosity=-1)
    else:
        cls = HistGradientBoostingClassifier if classifier else HistGradientBoostingRegressor
        model = cls(max_iter=12, random_state=42)
    model.fit(data, y)
    original = model.predict_proba if classifier else model.predict
    results = []
    for backend in ("pandas", "polars", "arrow"):
        frame = {"pandas": data.to_pandas, "polars": lambda: data, "arrow": data.to_arrow}[backend]()
        seen = []

        def guarded(batch):
            assert isinstance(batch, type(frame)), type(batch)
            seen.append(batch)
            return original(batch)

        method = "predict_proba" if classifier else "predict"
        setattr(model, method, guarded)
        exp = dx.Explainer(model, frame, y, label="model", verbose=False)
        obs = frame.iloc[[0]] if backend == "pandas" else (frame.head(1) if backend == "polars" else frame.slice(0, 1))
        if family == "importance":
            explanation = exp.model_parts(N=40, B=3, random_state=42, keep_raw_permutations=True)
        elif family in ("breakdown", "shap"):
            explanation = exp.predict_parts(obs, type="break_down" if family == "breakdown" else "shap", N=40, B=3, random_state=42)
        elif family == "profile":
            explanation = exp.model_profile(N=20, variables=["x"], grid_points=4, random_state=42, verbose=False)
        else:
            explanation = exp.model_performance()
        assert exp.data is frame and seen
        results.append(explanation.result.select_dtypes(include=np.number).reset_index(drop=True))
        setattr(model, method, original)
    for result in results[1:]:
        pd.testing.assert_frame_equal(results[0], result, check_exact=False, rtol=1e-10, atol=1e-10)
