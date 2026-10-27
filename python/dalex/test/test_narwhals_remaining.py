"""Remaining explanation families must retain the original model's native features."""

import numpy as np
import pandas as pd
import polars as pl
import pytest
import dalex as dx


class NativeModel:
    def __init__(self, backend):
        self.backend = backend
        self.seen = []

    def predict(self, data):
        assert isinstance(
            data, pd.DataFrame if self.backend == "pandas" else pl.DataFrame
        ), type(data)
        self.seen.append(data)
        return (2 * data["x"] - 0.5 * data["z"] + 0.1 * data["u"]).to_numpy()


def fixture(backend):
    rng = np.random.default_rng(123)
    frame = pd.DataFrame(rng.normal(size=(48, 3)), columns=["x", "z", "u"])
    data = frame if backend == "pandas" else pl.from_pandas(frame)
    model = NativeModel(backend)
    y = (2 * frame.x - 0.5 * frame.z + 0.1 * frame.u).to_numpy() + rng.normal(
        size=48
    ) * 0.1
    exp = dx.Explainer(
        model, data, y, model_type="regression", label="model", verbose=False
    )
    obs = data.iloc[[0]] if backend == "pandas" else data.head(1)
    return exp, obs


@pytest.mark.parametrize(
    "family",
    [
        "lime",
        "kernel_shap",
        "tree_surrogate",
        "linear_surrogate",
        "aspect_default",
        "aspect_shap",
        "model_aspect",
        "predict_triplot",
        "model_triplot",
    ],
)
def test_remaining_families_parity(family):
    results = []
    for backend in ("pandas", "polars"):
        exp, obs = fixture(backend)
        if family == "lime":
            explanation = exp.predict_surrogate(
                obs, num_samples=80, num_features=3, random_state=42
            )
            result = explanation.result
        elif family == "kernel_shap":
            # pandas callback-only models are also handled through the matrix bridge.
            explanation = exp.predict_parts(
                obs,
                type="shap_wrapper",
                shap_explainer_type="KernelExplainer",
                nsamples=20,
                silent=True,
                l1_reg=0,
            )
            results.append(explanation.result)
            continue
        elif family in ("tree_surrogate", "linear_surrogate"):
            model = (
                exp.model_surrogate(
                    type=family.split("_")[0], max_vars=2, random_state=42
                )
                if family.startswith("tree")
                else exp.model_surrogate(type="linear", max_vars=2)
            )
            assert len(model.feature_names) == 2
            from dalex._dataframe import select_columns

            results.append(model.predict(select_columns(exp.data, model.feature_names)))
            continue
        else:
            aspect = dx.Aspect(exp)
            if family.startswith("aspect_"):
                explanation = aspect.predict_parts(
                    obs,
                    type=family.split("_")[1],
                    variable_groups={"pair": ["x", "z"], "single": ["u"]},
                    N=32,
                    B=3,
                    random_state=42,
                )
            elif family == "model_aspect":
                explanation = aspect.model_parts(N=32, B=3, random_state=42)
            elif family == "predict_triplot":
                explanation = aspect.predict_triplot(obs, N=32, B=3, random_state=42)
            else:
                explanation = aspect.model_triplot(N=32, B=3, random_state=42)
            result = explanation.result
            explanation.plot(show=False)
        assert exp.data is not None
        results.append(result.select_dtypes(include=np.number).to_numpy(dtype=float))
    np.testing.assert_allclose(
        results[0], results[1], rtol=1e-10, atol=1e-10, equal_nan=True
    )


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_arena_native_observations_and_plots(backend):
    exp, obs = fixture(backend)
    arena = dx.Arena(verbose=False)
    arena.push_model(exp)
    arena.push_observations(obs)
    data = (
        exp.data.assign(target=exp.y)
        if backend == "pandas"
        else exp.data.with_columns(pl.Series("target", exp.y))
    )
    arena.push_dataset(data, target="target", label="data")
    model = arena.get_params("model")[0]
    observation = arena.get_params("observation")[0]
    native = observation.get_row_for_model(model)
    assert type(native) is type(exp.data)
    assert isinstance(arena.get_param_attributes("variable", "x")["min"], float)
    arena.set_option("DatasetShapleyValues", "N", 3)
    arena.set_option("DatasetShapleyValues", "B", 2)
    arena.set_option("DatasetShapleyValues", "cpus", 1)
    arena.set_option("VariableImportance", "B", 2)
    arena.set_option("ShapleyValues", "B", 2)
    arena.set_option("ShapleyValues", "cpus", 1)
    for plot in [
        "Breakdown",
        "SHAPValues",
        "ShapleyValuesDependence",
        "ShapleyValuesVariableImportance",
        "CeterisParibus",
        "Metrics",
        "PartialDependence",
        "AccumulatedDependence",
        "FeatureImportance",
        "VariableDistribution",
        "VariableAgainstAnother",
    ]:
        params = {
            "model": model,
            "observation": observation,
            "variable": arena.find_param_value("variable", "x"),
            "dataset": arena.get_params("dataset")[0],
        }
        result = arena.plots_manager.get_plot(plot, params, wait=True)
        assert result.plot_component != "Message", (plot, result.data)
    assert type(exp.data) is type(native)


def test_regression_fairness_backend_parity():
    results = []
    for backend in ("pandas", "polars"):
        exp, _ = fixture(backend)
        result = exp.model_fairness(protected=np.tile(["a", "b"], 24), privileged="a")
        results.append(result.result)
    pd.testing.assert_frame_equal(*results)


@pytest.mark.parametrize("kind", ["TreeExplainer", "LinearExplainer"])
def test_model_specific_shap_backend_parity(kind):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    exp, _ = fixture("pandas")
    model = (
        RandomForestRegressor(n_estimators=5, random_state=42)
        if kind == "TreeExplainer"
        else LinearRegression()
    ).fit(exp.data, exp.y)
    results = []
    for backend in ("pandas", "polars"):
        data = exp.data if backend == "pandas" else pl.from_pandas(exp.data)

        def predict(m, frame):
            assert isinstance(
                frame, pd.DataFrame if backend == "pandas" else pl.DataFrame
            )
            return m.predict(frame)

        wrapper = dx.Explainer(
            model, data, exp.y, predict_function=predict, verbose=False
        )
        obs = data.iloc[[0]] if backend == "pandas" else data.head(1)
        result = wrapper.predict_parts(
            obs, type="shap_wrapper", shap_explainer_type=kind
        )
        results.append(result.result)
    np.testing.assert_allclose(*results, atol=1e-12, rtol=1e-12)


class NativeClassifier:
    def __init__(self, backend):
        self.backend = backend

    def predict_proba(self, data):
        assert isinstance(
            data, pd.DataFrame if self.backend == "pandas" else pl.DataFrame
        )
        p = 1 / (1 + np.exp(-data["x"].to_numpy()))
        return np.column_stack([1 - p, p])


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_arena_classification_and_fairness(backend):
    rng = np.random.default_rng(123)
    frame = pd.DataFrame({"x": rng.normal(size=80), "group": np.tile(["a", "b"], 40)})
    y = (frame.x.to_numpy() + rng.normal(size=80) * 0.5 > 0).astype(int)
    data = frame if backend == "pandas" else pl.from_pandas(frame)
    exp = dx.Explainer(
        NativeClassifier(backend),
        data,
        y,
        model_type="classification",
        verbose=False,
        label="model",
    )
    arena = dx.Arena(verbose=False)
    arena.push_model(exp)
    params = {
        "model": arena.get_params("model")[0],
        "variable": arena.find_param_value("variable", "group"),
    }
    for plot in ["ROC", "Fairness"]:
        result = arena.plots_manager.get_plot(plot, params, wait=True)
        assert result.plot_component != "Message", result.data
        assert result.data


def test_arena_http_custom_polars_observation():
    import json
    import requests
    from dalex.arena.static import get_free_port

    exp, obs = fixture("polars")
    arena = dx.Arena(verbose=False)
    arena.push_model(exp)
    port = get_free_port()
    arena.run_server(host="127.0.0.1", port=port)
    try:
        response = requests.get(
            f"http://127.0.0.1:{port}/Breakdown",
            params={"model": "model", "observation": json.dumps(obs.to_dicts()[0])},
            timeout=15,
        )
        response.raise_for_status()
        assert response.json()["plotComponent"] != "Message", response.text
    finally:
        arena.stop_server()


def test_arena_resource_notification_survives_until_consumer():
    from dalex.arena._resource import Resource

    class Notification(Resource):
        resource_type = "notification-test"
        required_params = []
        options_category = "notification-test"
        options = {}

    resource = Notification(dx.Arena(verbose=False))
    resource._emit_update()
    assert resource.update_event.wait(timeout=0.1)
    resource.wait_for_update()
    assert not resource.update_event.is_set()
    resource.cancel()
    assert resource.update_event.wait(timeout=0.1)


def test_arena_resource_background_error_is_reported():
    from dalex.arena._resource import Resource

    class Failure(Resource):
        resource_type = "failure-test"
        required_params = []
        options_category = "failure-test"
        options = {}

        def _fit(self):
            raise RuntimeError("predictor failed")

    resource = Failure(dx.Arena(verbose=False)).fit({})
    assert resource.update_event.wait(timeout=2)
    resource.thread.join(timeout=2)
    assert not resource.thread.is_alive()
    with pytest.raises(RuntimeError, match="predictor failed"):
        resource.get_result()


def test_arena_immediate_stop_and_restart():
    from dalex.arena.static import get_free_port

    arena = dx.Arena(verbose=False)
    port = get_free_port()
    for _ in range(2):
        arena.run_server(host="127.0.0.1", port=port)
        arena.stop_server()
        assert arena.server_thread is None


def test_arena_startup_failure_reaches_caller(monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("cannot bind")

    monkeypatch.setattr("dalex.arena.object.start_server", fail)
    arena = dx.Arena(verbose=False)
    with pytest.raises(RuntimeError, match="failed to start") as error:
        arena.run_server()
    assert isinstance(error.value.__cause__, OSError)
    assert arena.server_thread is None
