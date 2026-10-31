"""Independent Gaussian references for optimized numeric profile boundaries."""
import narwhals.stable.v2 as nw
import numpy as np
import pandas as pd
import polars as pl
import pytest
from scipy.stats import norm

from dalex.model_explanations._aggregated_profiles._native import reduce_profiles


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('span', [1e-6, .1, 1., 1e6])
@pytest.mark.parametrize('dtype', [pl.Float32, pl.Float64])
def test_conditional_gaussian_vectors_match_scipy(backend, span, dtype):
    grid = np.repeat([-1., 0., 1.], 2)
    original = np.tile([-1., 1.], 3)
    predictions = 5 + original * 3 + grid**2
    data = pl.DataFrame({'_x_': grid, '_original_': original, '_yhat_': predictions}).cast(dtype)
    frame = data.to_pandas() if backend == 'pandas' else data if backend == 'polars' else data.to_arrow()
    actual = nw.from_native(reduce_profiles(frame, 'conditional', span=span))
    expected = []
    for point in [-1., 0., 1.]:
        mask = grid == point
        weights = norm.pdf((original[mask] - point) / 2, scale=span)
        expected.append(np.average(predictions[mask], weights=weights) if weights.sum() else 0.)
    assert actual['_x_'].to_list() == [-1., 0., 1.]
    np.testing.assert_allclose(actual['_yhat_'].to_numpy(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('backend', ['pandas', 'polars', 'pyarrow'])
@pytest.mark.parametrize('missing', [False, True])
@pytest.mark.parametrize('span', [.1, 1.])
def test_gaussian_vectors_do_not_narrow_large_integer_keys(backend, missing, span):
    base = 2**63 + 129
    points = [base, base + 1, base + 2]
    grid = [point for point in points for _ in range(2)]
    original = [None if missing else base, base + 2] * 3
    predictions = [2., 8., 3., 9., 4., 10.]
    data = pl.DataFrame({'_x_': pl.Series(grid, dtype=pl.UInt64),
                         '_original_': pl.Series(original, dtype=pl.UInt64), '_yhat_': predictions})
    frame = (pd.DataFrame({'_x_': pd.array(grid, dtype='UInt64'),
                           '_original_': pd.array(original, dtype='UInt64'), '_yhat_': predictions})
             if backend == 'pandas' else data if backend == 'polars' else data.to_arrow())
    before = nw.from_native(frame).to_arrow()
    actual = nw.from_native(reduce_profiles(frame, 'conditional', span=span))
    expected = []
    for index, point in enumerate(points):
        # Subtract Python integers before entering the external floating statistic.
        distances = [np.nan if value is None else (value - point) / 2 for value in original[index*2:index*2+2]]
        weights = norm.pdf(distances, scale=span)
        weights[np.isnan(weights)] = 0.
        expected.append(np.average(predictions[index*2:index*2+2], weights=weights) if weights.sum() else 0.)
    assert actual['_x_'].to_list() == points
    assert actual.schema['_x_'] == nw.UInt64
    np.testing.assert_allclose(actual['_yhat_'].to_numpy(), expected, rtol=1e-12, atol=1e-12)
    assert nw.from_native(frame).to_arrow().equals(before)
