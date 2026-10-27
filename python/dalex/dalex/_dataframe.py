"""Narwhals-based dataframe interoperability for DALEX.

Operate on eager native dataframes and return them to the model in their original
backend.  The pandas-specific row-taking fast path intentionally preserves the
existing DALEX behavior for non-default pandas indexes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import narwhals as nw


def as_eager_dataframe(data):
    """Validate a Narwhals-supported *eager* dataframe without converting it."""
    try:
        frame = nw.from_native(data, eager_only=True)
    except TypeError as exc:
        raise TypeError(
            "'data' must be a pandas.DataFrame, numpy.ndarray, "
            "or Narwhals-supported eager dataframe"
        ) from exc
    if not isinstance(frame, nw.DataFrame):
        raise TypeError("'data' must be a two-dimensional eager dataframe")
    return frame


def as_numpy_vector(value, name):
    """Normalize Series or one-column eager DataFrame targets and weights."""
    try:
        series = nw.from_native(value, series_only=True)
    except TypeError:
        frame = as_eager_dataframe(value)
        if len(frame.columns) != 1:
            raise ValueError(f"'{name}' must have one column") from None
        return frame.to_numpy().reshape(-1)
    return series.to_numpy()


def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    return as_eager_dataframe(data).to_numpy()


def take_rows(data, positions):
    """Take positional rows (including repeats) while preserving native backend.

    Pandas .iloc is kept for its existing index-retention semantics. On other
    backends Narwhals gathers the requested positional order.
    """
    index = np.asarray(positions)
    if index.ndim != 1 or (index.size and index.dtype.kind not in "iu"):
        raise TypeError("row positions must be a one-dimensional integer sequence")
    index = index.astype(np.int64)
    size = data.shape[0]
    if np.any(index >= size) or np.any(index < -size):
        raise IndexError("row position out of bounds")
    if isinstance(data, pd.DataFrame):
        return data.iloc[index, :]
    frame = as_eager_dataframe(data)
    if not index.size:
        return frame.head(0).to_native()
    index = np.where(index < 0, index + size, index)
    return frame[index.tolist(), :].to_native()


def shuffle_columns(data, columns, positions):
    """Permute one or more columns with a shared row permutation.

    Narwhals preserves native backends; pandas assignment retains legacy index semantics.
    """
    names = [columns] if isinstance(columns, str) else list(columns)
    frame = as_eager_dataframe(data)
    if len(positions) != data.shape[0]:
        raise ValueError("permutation must have the same length as data")
    selected = frame.select(names).to_native()
    other = take_rows(selected, positions)
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        # Positional assignment is essential with non-default or repeated indexes.
        result.loc[:, names] = other.to_numpy()
        return result
    other = as_eager_dataframe(other)
    return frame.with_columns(other.get_column(col) for col in names).to_native()


def to_pandas(data):
    """Materialize an explanation result/plot boundary; never use for model input."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return as_eager_dataframe(data).to_pandas()


def column_values(data, name):
    return as_eager_dataframe(data).get_column(name).to_numpy()


def set_columns(data, values):
    """Return a frame with positional column replacements, without mutating data."""
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        for name, value in values.items():
            result.loc[:, name] = value
        return result
    frame = as_eager_dataframe(data)
    backend = nw.get_native_namespace(frame)
    expressions = []
    for name, value in values.items():
        if np.ndim(value) == 0:
            expression = nw.lit(value)
            dtype = frame.schema[name]
            if isinstance(dtype, (nw.Categorical, nw.Enum)):
                expression = expression.cast(dtype)
            expressions.append(expression.alias(name))
        else:
            series = nw.from_dict({name: value}, backend=backend).get_column(name)
            if isinstance(frame.schema[name], (nw.Categorical, nw.Enum)):
                series = series.cast(frame.schema[name])
            expressions.append(series)
    return frame.with_columns(expressions).to_native()


def native_observation(observation, reference, single=False):
    """Validate observations for a non-pandas explainer without collecting lazy data."""
    frame = as_eager_dataframe(reference)
    if isinstance(observation, (list, np.ndarray, pd.Series)):
        values = np.asarray(observation)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(frame.columns):
            raise ValueError("Wrong new_observation dimension")
        observation = nw.from_dict(
            {name: values[:, i] for i, name in enumerate(frame.columns)},
            backend=nw.get_native_namespace(frame),
        ).to_native()
    obs = as_eager_dataframe(observation)
    if nw.get_native_namespace(obs) is not nw.get_native_namespace(frame):
        raise TypeError("new_observation must use the same dataframe backend as explainer.data")
    if obs.columns != frame.columns:
        raise ValueError("new_observation columns must match explainer.data in order")
    if single and len(obs) != 1:
        raise ValueError("Wrong new_observation dimension")
    return observation


class NativePredictFunction:
    """Picklable prediction adapter: normalize series outputs, preserve input backend."""

    def __init__(self, function):
        self.function = function

    def __call__(self, model, data):
        prediction = self.function(model, data)
        if isinstance(prediction, np.ndarray):
            return prediction
        return as_numpy_vector(prediction, "prediction")


def require_pandas(data, method):
    """Fail explicitly at an explanation family whose native migration is pending."""
    if data is not None and not isinstance(data, pd.DataFrame):
        raise NotImplementedError(
            f"{method} currently requires pandas features; native backend support is pending. "
            "Use predict_parts(type='shap') for native DALEX Shapley values."
        )
