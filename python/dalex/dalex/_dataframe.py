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
    if isinstance(data, pd.DataFrame):
        return data.to_numpy()
    return as_eager_dataframe(data).to_numpy()


def column_names(data):
    """Feature labels, including backends whose .columns contains column arrays."""
    return data.columns if isinstance(data, pd.DataFrame) else as_eager_dataframe(data).columns


def take_rows(data, positions):
    """Take positional rows (including repeats) while preserving native backend.

    Pandas .iloc is kept for its existing index-retention semantics. On other
    backends Narwhals gathers the requested positional order.
    """
    index = np.asarray(positions)
    if index.ndim != 1 or (index.size and index.dtype.kind not in "iu"):
        raise TypeError("row positions must be a one-dimensional integer sequence")
    index = index.astype(np.int64, copy=False)
    size = data.shape[0]
    if np.any(index >= size) or np.any(index < -size):
        raise IndexError("row position out of bounds")
    if isinstance(data, pd.DataFrame):
        return data.iloc[index, :]
    frame = as_eager_dataframe(data)
    if not index.size:
        return frame.head(0).to_native()
    index = np.where(index < 0, index + size, index)
    return frame[index, :].to_native()


def shuffle_columns(data, columns, positions):
    """Permute one or more columns with a shared row permutation.

    Narwhals preserves native backends; pandas assignment retains legacy index semantics.
    """
    names = [columns] if np.isscalar(columns) else list(columns)
    if len(positions) != data.shape[0]:
        raise ValueError("permutation must have the same length as data")
    if isinstance(data, pd.DataFrame):
        other = take_rows(data.loc[:, names], positions)
        result = data.copy()
        # Extension arrays assign positionally without losing categorical/nullable
        # dtypes on older pandas, or aligning shuffled row labels.
        for name in names:
            result[name] = other[name].array
        return result
    frame = as_eager_dataframe(data)
    # Keep the Narwhals wrapper throughout the operation. Round-tripping the
    # selected native frame through take_rows used to validate/wrap it twice more.
    other = frame[np.asarray(positions, dtype=np.int64), names]
    return frame.with_columns(other.get_column(col) for col in names).to_native()


def to_pandas(data):
    """Materialize an explanation result/plot boundary; never use for model input."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return as_eager_dataframe(data).to_pandas()


def column_values(data, name):
    if isinstance(data, pd.DataFrame):
        return data[name].to_numpy()
    return as_eager_dataframe(data).get_column(name).to_numpy()


def set_columns(data, values):
    """Return a frame with positional column replacements, without mutating data."""
    if isinstance(data, pd.DataFrame):
        result = data.copy()
        for name, value in values.items():
            if np.ndim(value) and isinstance(result[name].dtype, pd.CategoricalDtype):
                column = result[name].copy()
                column.iloc[:] = value
                result[name] = column.array
            else:
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
        values = np.asarray(observation, dtype=object) if isinstance(observation, (list, pd.Series)) else np.asarray(observation)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[1] != len(frame.columns):
            raise ValueError("Wrong new_observation dimension")
        observation = frame_from_values(values, reference)
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



class NativeResidualFunction:
    """Preserve the three-argument callback contract and normalize native vectors."""

    def __init__(self, function):
        self.function = function

    def __call__(self, model, data, y):
        result = self.function(model, data, y)
        return result if isinstance(result, np.ndarray) else as_numpy_vector(result, "residuals")


def select_columns(data, columns):
    names = list(columns)
    if isinstance(data, pd.DataFrame):
        return data.loc[:, names]
    return as_eager_dataframe(data).select(names).to_native()


def numeric_columns(data):
    if isinstance(data, pd.DataFrame):
        return list(data.select_dtypes(include=np.number).columns)
    return [name for name, dtype in as_eager_dataframe(data).schema.items() if dtype.is_numeric()]


def frame_from_values(values, reference, columns=None):
    """Explicit NumPy-library boundary: restore native feature batches for callbacks."""
    names = list(column_names(reference)) if columns is None else list(columns)
    array = np.asarray(values)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] != len(names):
        raise ValueError("Feature matrix must have the same ordered columns as the reference")
    if isinstance(reference, pd.DataFrame):
        # A mixed NumPy matrix has object dtype for every column. Reconstruct
        # each column separately so numeric features remain numeric, and restore
        # categorical metadata expected by dataframe-aware model pipelines.
        result = pd.DataFrame({name: array[:, i].tolist() for i, name in enumerate(names)})
        for name in names:
            if isinstance(reference[name].dtype, pd.CategoricalDtype):
                result[name] = pd.Categorical(result[name], dtype=reference[name].dtype)
        return result
    frame = as_eager_dataframe(reference)
    native = nw.from_dict(
        {name: array[:, i].tolist() for i, name in enumerate(names)},
        backend=nw.get_native_namespace(frame),
    )
    casts = [nw.col(name).cast(frame.schema[name]) for name in names
             if isinstance(frame.schema[name], (nw.Categorical, nw.Enum))]
    return native.with_columns(casts).to_native() if casts else native.to_native()
