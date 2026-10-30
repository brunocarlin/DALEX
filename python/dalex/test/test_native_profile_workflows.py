"""Real fitted preprocessing/model -> CP -> exports/plots/pickle integrations."""
import pickle
from contextlib import contextmanager
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import narwhals.stable.v2 as nw
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
import dalex as dx


def polars_features(frame):
    if isinstance(frame, pl.DataFrame):
        return frame
    if isinstance(frame, pa.Table):
        return pl.from_arrow(frame)
    if isinstance(frame, pd.DataFrame):
        return pl.from_pandas(frame)
    raise TypeError('This fitted pipeline accepts dataframe inputs')


def years(frame):
    return nw.from_native(frame).select(nw.col('when').dt.year()).to_numpy()


def transformed_polars(values):
    return pl.DataFrame(values, schema=['x', 'z', 'kind_a', 'kind_b', 'when'], orient='row')


@pytest.fixture(scope='module', params=[('sklearn', False), ('sklearn', True), ('lightgbm', False), ('lightgbm', True)])
def fitted_pipeline(request):
    library, classification = request.param
    rng = np.random.default_rng(615)
    data = pl.DataFrame({'x': rng.normal(size=64), 'z': rng.normal(size=64),
                         'kind': ['a', 'b'] * 32,
                         'when': pd.date_range('2020-01-01', periods=64, freq='ns', tz='UTC'),
                         'group': pl.Series(['u', None, 'u', None] * 16).cast(pl.Enum(['u', 'unused']))})
    response = 2 * data['x'].to_numpy() - data['z'].to_numpy() + np.tile([0., 2.], 32)
    y = (response > np.median(response)).astype(int) if classification else response
    if library == 'lightgbm':
        from lightgbm import LGBMClassifier, LGBMRegressor
        estimator = (LGBMClassifier if classification else LGBMRegressor)(n_estimators=8, n_jobs=1, random_state=42, verbosity=-1)
    else:
        estimator = (HistGradientBoostingClassifier if classification else HistGradientBoostingRegressor)(max_iter=8, random_state=42)
    preparation = ColumnTransformer([
        ('numeric', make_pipeline(SimpleImputer(), StandardScaler()), [0, 1]),
        ('category', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [2]),
        ('time', FunctionTransformer(years, validate=False), [3]),
    ])
    # The fitted model owns these explicit adapter boundaries. DALEX must pass
    # through each original backend unchanged. Positional selectors also cover
    # older sklearn versions that cannot directly index Arrow/modern Polars.
    with threadpool_limits(limits=1):
        model = make_pipeline(FunctionTransformer(polars_features, validate=False), preparation,
                              FunctionTransformer(transformed_polars, validate=False), estimator).fit(data, y)
    return model, data, y, classification


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
@pytest.mark.parametrize('mixed', [False, True])
def test_native_profile_complete_model_workflow(fitted_pipeline, backend, mixed, monkeypatch):
    model, original, y, classification = fitted_pipeline
    data = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    observation = data.head(2) if backend != 'arrow' else data.slice(0, 2)
    expected_type = type(data)
    seen = []
    def predict(model, batch):
        assert isinstance(batch, expected_type)
        seen.append(len(batch))
        return model.predict_proba(batch)[:, 1] if classification else model.predict(batch)
    with threadpool_limits(limits=1):
        exp = dx.Explainer(model, data, y, predict_function=predict,
                           model_type='classification' if classification else 'regression', verbose=False)
        variables = ['x', 'kind', 'when'] if mixed else ['x', 'z']
        if backend != 'pandas':
            frame_type, series_type = pd.DataFrame, pd.Series
            class ForbiddenFrame(frame_type):
                def __init__(self, *args, **kwargs):
                    raise AssertionError('Native model workflow constructed pandas DataFrame')
            class ForbiddenSeries(series_type):
                def __init__(self, *args, **kwargs):
                    raise AssertionError('Native model workflow constructed pandas Series')
            def forbidden(*args, **kwargs):
                raise AssertionError('Native model workflow converted to pandas')
            with monkeypatch.context() as patch:
                patch.setattr(pd, 'DataFrame', ForbiddenFrame)
                patch.setattr(pd, 'Series', ForbiddenSeries)
                patch.setattr(nw.DataFrame, 'to_pandas', forbidden)
                profile = exp.predict_profile(observation, variables=variables, grid_points=4, verbose=False)
                native = profile.result_as(partition='x')
                assert isinstance(native, expected_type)
                if mixed:
                    temporal = nw.from_native(profile.result_as(partition='when'))
                    assert temporal.schema['when'] == nw.Datetime('ns', 'UTC')
        else:
            profile = exp.predict_profile(observation, variables=variables, grid_points=4, verbose=False)
            native = profile.result_as(partition='x')
        assert seen
        assert isinstance(profile.result, pd.DataFrame)
        assert isinstance(profile.new_observation, pd.DataFrame)
        np.testing.assert_allclose(profile.new_observation['_yhat_'], exp.predict(observation), rtol=1e-12, atol=1e-12)
        selected = profile.result.loc[profile.result['_vname_'] == 'x', '_yhat_'].to_numpy()
        np.testing.assert_allclose(nw.from_native(native)['_yhat_'].to_numpy(), selected, rtol=1e-12, atol=1e-12)
        first = profile.result.columns.get_loc('_yhat_')
        profile.result.iloc[0, first] += .25
        changed = profile.result_as(partition='x')
        assert nw.from_native(changed)['_yhat_'].to_numpy()[0] == profile.result.iloc[0, first]
        if mixed:
            temporal = nw.from_native(profile.result_as(partition='when'))
            assert temporal.schema['when'] == nw.Datetime('ns', 'UTC')
        restored = pickle.loads(pickle.dumps(profile))
        pd.testing.assert_frame_equal(restored.result, profile.result)
        assert restored.plot(variables=['x'], show=False) is not None


@contextmanager
def native_computation_only(monkeypatch, backend):
    if backend == 'pandas':
        yield
        return
    frame_type, series_type = pd.DataFrame, pd.Series
    class ForbiddenFrame(frame_type):
        def __init__(self, *args, **kwargs):
            raise AssertionError('Native aggregate workflow constructed pandas DataFrame')
    class ForbiddenSeries(series_type):
        def __init__(self, *args, **kwargs):
            raise AssertionError('Native aggregate workflow constructed pandas Series')
    def forbidden(*args, **kwargs):
        raise AssertionError('Native aggregate workflow converted to pandas')
    with monkeypatch.context() as patch:
        patch.setattr(pd, 'DataFrame', ForbiddenFrame)
        patch.setattr(pd, 'Series', ForbiddenSeries)
        patch.setattr(nw.DataFrame, 'to_pandas', forbidden)
        yield


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
@pytest.mark.parametrize('kind', ['partial', 'conditional', 'accumulated'])
@pytest.mark.parametrize('categorical', [False, True])
def test_complete_cp_to_aggregate_real_model_workflow(fitted_pipeline, backend, kind, categorical, monkeypatch):
    model, original, y, classification = fitted_pipeline
    data = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    expected_type = type(data)
    def predict(model, batch):
        assert isinstance(batch, expected_type)
        return model.predict_proba(batch)[:, 1] if classification else model.predict(batch)
    variable = 'kind' if categorical else 'x'
    settings = dict(type=kind, variables=[variable], N=12, grid_points=4, random_state=42,
                    variable_type='categorical' if categorical else 'numerical',
                    groups=None if categorical else ['group'], verbose=False)
    with threadpool_limits(limits=1):
        exp = dx.Explainer(model, data, y, predict_function=predict,
                           model_type='classification' if classification else 'regression', label='pipeline', verbose=False)
        with native_computation_only(monkeypatch, backend):
            profile = exp.model_profile(**settings)
            native = profile.result_as()
            raw = profile.raw_profiles.result_as(partition=variable)
            assert isinstance(native, expected_type) and isinstance(raw, expected_type)
            assert '_compatibility_result' not in profile.raw_profiles.__dict__
            assert '_compatibility_new_observation' not in profile.raw_profiles.__dict__
        reference = dx.Explainer(model, original.to_pandas(), y,
                                 model_type='classification' if classification else 'regression', label='pipeline', verbose=False)
        expected = reference.model_profile(**settings)
        pd.testing.assert_frame_equal(profile.result, expected.result, check_exact=False, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(nw.from_native(native)['_yhat_'].to_numpy(), profile.result['_yhat_'],
                                   rtol=1e-10, atol=1e-10)
        restored = pickle.loads(pickle.dumps(profile))
        pd.testing.assert_frame_equal(restored.result, profile.result)
        assert restored.plot(show=False) is not None


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
def test_categorical_temporal_aggregation_does_not_merge_nanosecond_levels(fitted_pipeline, backend, monkeypatch):
    model, original, y, classification = fitted_pipeline
    data = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    with threadpool_limits(limits=1):
        exp = dx.Explainer(model, data, y, model_type='classification' if classification else 'regression', verbose=False)
        with native_computation_only(monkeypatch, backend):
            profile = exp.model_profile(variables=['when'], variable_type='categorical',
                                        N=6, random_state=42, verbose=False)
            result = nw.from_native(profile.result_as())
            assert len(result) == 64
            assert result['_x_'].n_unique() == 64
        assert profile.plot(show=False) is not None


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
@pytest.mark.parametrize('method', ['model_performance', 'model_diagnostics'])
def test_native_performance_and_diagnostics_real_models(fitted_pipeline, backend, method, monkeypatch):
    model, original, y, classification = fitted_pipeline
    data = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    with threadpool_limits(limits=1):
        exp = dx.Explainer(model, data, y.copy(), model_type='classification' if classification else 'regression',
                           label='pipeline', verbose=False, precalculate=False)
        with native_computation_only(monkeypatch, backend):
            result = getattr(exp, method)()
            native = result.result_as()
            assert isinstance(native, type(data))
            if method == 'model_performance':
                assert isinstance(result.result_as(table='residuals'), type(data))
        reference = dx.Explainer(model, original.to_pandas(), y.copy(),
                                 model_type='classification' if classification else 'regression',
                                 label='pipeline', verbose=False, precalculate=False)
        expected = getattr(reference, method)()
        pd.testing.assert_frame_equal(result.result, expected.result, check_exact=False, rtol=1e-10, atol=1e-10)
        if method == 'model_performance':
            assert list(result.result.index) == ['pipeline']
            pd.testing.assert_frame_equal(result.residuals, expected.residuals)
            result.result.index = pd.Index(['renamed'])
            assert nw.from_native(result.result_as())['label'].to_list() == ['renamed']
        restored = pickle.loads(pickle.dumps(result))
        pd.testing.assert_frame_equal(restored.result, result.result)
        assert restored.plot(show=False) is not None


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'arrow'])
@pytest.mark.parametrize('kind', ['break_down', 'break_down_interactions', 'shap'])
def test_native_contributions_real_models_and_legacy_results(fitted_pipeline, backend, kind, monkeypatch):
    model, original, y, classification = fitted_pipeline
    data = original.to_pandas() if backend == 'pandas' else original if backend == 'polars' else original.to_arrow()
    observation = data.head(1) if backend != 'arrow' else data.slice(0, 1)
    with threadpool_limits(limits=1):
        exp = dx.Explainer(model, data, y, model_type='classification' if classification else 'regression',
                           label='pipeline', verbose=False, precalculate=False)
        settings = dict(type=kind, N=20, B=3, random_state=42, keep_distributions=True)
        with native_computation_only(monkeypatch, backend):
            result = exp.predict_parts(observation, **settings)
            native = result.result_as(partition='x') if kind == 'shap' else result.result_as()
            assert isinstance(native, type(data))
            assert isinstance(result.result_as(table='yhats_distributions'), type(data))
            assert '_compatibility_result' not in result.__dict__ if backend != 'pandas' else True
        reference = dx.Explainer(model, original.to_pandas(), y,
                                 model_type='classification' if classification else 'regression',
                                 label='pipeline', verbose=False, precalculate=False)
        expected = reference.predict_parts(original.to_pandas().head(1), **settings)
        pd.testing.assert_frame_equal(result.result, expected.result, check_exact=False, rtol=1e-10, atol=1e-10)
        pd.testing.assert_frame_equal(result.yhats_distributions, expected.yhats_distributions,
                                      check_exact=False, rtol=1e-10, atol=1e-10)
        restored = pickle.loads(pickle.dumps(result))
        pd.testing.assert_frame_equal(restored.result, result.result)
        assert restored.plot(show=False) is not None
