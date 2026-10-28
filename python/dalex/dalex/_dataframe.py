"""Narwhals-based dataframe interoperability for DALEX.

Operate on eager native dataframes and return them to the model in their original
backend.  The pandas-specific row-taking fast path intentionally preserves the
existing DALEX behavior for non-default pandas indexes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import narwhals.stable.v2 as nw
from narwhals.dependencies import is_polars_dataframe, is_pyarrow_table


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
        return to_numpy(frame.to_native()).reshape(-1)
    native = series.to_native()
    if hasattr(getattr(native, "type", None), "index_type"):
        return native.cast(native.type.value_type).to_numpy()
    return series.to_numpy()


def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, pd.DataFrame):
        return data.to_numpy()
    if is_polars_dataframe(data):
        # Export methods in older Polars versions borrow the frame mutably.
        # A cheap buffer-sharing clone isolates simultaneous explanation calls.
        data = data.clone()
    if is_pyarrow_table(data) and any(
        hasattr(column.type, "index_type") for column in data.columns
    ):
        # Arrow dictionary.to_numpy can expose a null index's physical value
        # (e.g. an unused Enum category). Decode before crossing the boundary.
        return np.column_stack([column_values(data, name) for name in data.column_names])
    return as_eager_dataframe(data).to_numpy()


def column_names(data):
    """Feature labels, including backends whose .columns contains column arrays."""
    if isinstance(data, pd.DataFrame) or is_polars_dataframe(data):
        return data.columns
    if is_pyarrow_table(data):
        return data.column_names
    return as_eager_dataframe(data).columns


def iter_column_arrays(data, numeric_only=False):
    """Inspect features without materializing a full mixed/structured matrix."""
    if isinstance(data, pd.DataFrame):
        yield from (series.to_numpy() for _, series in data.items())
    else:
        frame = as_eager_dataframe(data)
        for name in frame.columns:
            column = frame.get_column(name)
            if numeric_only and not (column.dtype.is_numeric() or isinstance(
                column.dtype, (nw.Boolean, nw.Decimal, nw.Categorical, nw.Enum, nw.Object)
            )):
                # Datetime/nested/string features cannot equal a numeric target.
                # Avoid optional conversions unsupported by older backends.
                continue
            yield column_values(data, name) if is_pyarrow_table(data) else column.to_numpy()


def _row_positions(positions, size):
    """Validate positions before narrowing integers (uint64 must not wrap)."""
    index = np.asarray(positions)
    if index.ndim != 1 or (index.size and index.dtype.kind not in "iu"):
        raise TypeError("row positions must be a one-dimensional integer sequence")
    if np.any(index >= size) or (index.dtype.kind != "u" and np.any(index < -size)):
        raise IndexError("row position out of bounds")
    return index.astype(np.int64, copy=False)


def take_rows(data, positions):
    """Take positional rows (including repeats) while preserving native backend.

    Pandas .iloc is kept for its existing index-retention semantics. On other
    backends Narwhals gathers the requested positional order.
    """
    size = data.shape[0]
    index = _row_positions(positions, size)
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
    index = _row_positions(positions, data.shape[0])
    if isinstance(data, pd.DataFrame):
        other = data.loc[:, names].iloc[index, :]
        result = data.copy()
        # Extension arrays assign positionally without losing categorical/nullable
        # dtypes on older pandas, or aligning shuffled row labels.
        for name in names:
            result[name] = other[name].array
        return result
    frame = as_eager_dataframe(data)
    # Keep the Narwhals wrapper throughout the operation. Round-tripping the
    # selected native frame through take_rows used to validate/wrap it twice more.
    if not index.size:
        return frame.to_native()
    index = np.where(index < 0, index + len(frame), index)
    other = frame[index, names]
    return frame.with_columns(other.get_column(col) for col in names).to_native()


def to_pandas(data):
    """Materialize an explanation result/plot boundary; never use for model input."""
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if is_polars_dataframe(data):
        data = data.clone()
    return as_eager_dataframe(data).to_pandas()


def column_values(data, name):
    if isinstance(data, pd.DataFrame):
        return data[name].to_numpy()
    if is_pyarrow_table(data):
        column = data.column(name)
        if hasattr(column.type, "index_type"):
            return column.cast(column.type.value_type).to_numpy()
    return as_eager_dataframe(data).get_column(name).to_numpy()


def _temporal_counts(values, dtype):
    """Epoch counts avoid Python datetime's microsecond precision limit."""
    constructor = pd.Timestamp if isinstance(dtype, nw.Datetime) else pd.Timedelta
    kind = "datetime64" if isinstance(dtype, nw.Datetime) else "timedelta64"
    method = "to_datetime64" if isinstance(dtype, nw.Datetime) else "to_timedelta64"
    counts = []
    for value in values:
        if pd.isna(value):
            counts.append(None)
        else:
            native = getattr(constructor(value), method)()
            counts.append(int(native.astype(f"{kind}[{dtype.time_unit}]").view("i8")))
    return counts


def _restore_native_fields(result, reference, names, schema):
    """Keep metadata not represented by Narwhals' portable dtype classes."""
    if is_pyarrow_table(reference):
        for name in names:
            dtype = schema[name]
            field = reference.schema.field(name)
            column = result.column(name)
            if not dtype.is_numeric() or isinstance(dtype, nw.Decimal):
                column = column.cast(field.type)
            else:
                # Numeric grids may legitimately promote integer features to
                # floats; retain field metadata without truncating split points.
                field = field.with_type(column.type)
            result = result.set_column(result.schema.get_field_index(name), field, column)
        result = result.replace_schema_metadata(reference.schema.metadata)
    elif is_polars_dataframe(reference):
        for name in names:
            if isinstance(schema[name], nw.Decimal):
                # Decimal casting was not implemented by Narwhals 2.0. Use the
                # native reference dtype rather than infer a smaller precision.
                result = result.with_columns(result.get_column(name).cast(reference.schema[name]))
    return result


def set_columns(data, values, *, scalar=False, copy=True):
    """Replace columns positionally, copying pandas sources by default.

    ``scalar=True`` broadcasts observation cells, including list/struct cells.
    Otherwise sequences represent a full replacement column.
    ``copy=False`` can update an already-owned pandas work frame in place;
    native expression backends still return a new frame.
    """
    if isinstance(data, pd.DataFrame):
        result = data.copy() if copy else data
        for name, value in values.items():
            if isinstance(value, pd.Series):
                value = value.array  # replacements are positional, never index-aligned
            if scalar and isinstance(value, (list, tuple, dict, np.ndarray)):
                result[name] = pd.Series([value] * len(data), index=data.index, dtype=data[name].dtype)
            elif value is None or (isinstance(result[name].dtype, pd.CategoricalDtype) and not pd.api.types.is_scalar(value)):
                column = result[name].copy()
                column.iloc[:] = value
                result[name] = column.array
            else:
                try:
                    result.loc[:, name] = value
                except TypeError:
                    # pandas 3 rejects implicit integer -> float widening in
                    # .loc; counterfactual observations may be fractional.
                    result[name] = value
        return result
    frame = as_eager_dataframe(data)
    backend = nw.get_native_namespace(frame)
    # Only inspect columns being replaced. Collecting the entire portable schema
    # for each Shapley/break-down step creates quadratic work on wide frames.
    schema = {name: frame.get_column(name).dtype for name in values}
    expressions = []
    for name, value in values.items():
        dtype = schema[name]
        if scalar or pd.api.types.is_scalar(value) or isinstance(value, dict):
            if value is pd.NA or value is pd.NaT:
                value = None
            if (not dtype.is_numeric() or isinstance(dtype, nw.Decimal)) and pd.api.types.is_scalar(value) and pd.isna(value):
                # pandas represents missing categorical/date cells as NaN/NaT.
                # Native enum/dictionary casts require an actual null scalar.
                value = None
            if dtype.is_nested() and value is not None:
                expressions.append(nw.new_series(name, [value] * len(frame), dtype=dtype, backend=backend).cast(dtype))
                continue
            if isinstance(dtype, (nw.Datetime, nw.Duration)):
                value = _temporal_counts([value], dtype)[0]
            expression = nw.lit(value)
            if (value is None or not dtype.is_numeric()) and not isinstance(dtype, (nw.Decimal, nw.Unknown)):
                expression = expression.cast(dtype)
            expressions.append(expression.alias(name))
        else:
            if len(value) != len(frame):
                raise ValueError("replacement must have the same length as data")
            if isinstance(dtype, (nw.Datetime, nw.Duration)):
                series = nw.new_series(name, _temporal_counts(value, dtype), dtype=nw.Int64, backend=backend).cast(dtype)
            else:
                if isinstance(dtype, (nw.Categorical, nw.Enum, nw.String, nw.Boolean)):
                    value = [None if pd.isna(cell) else cell for cell in value]
                elif isinstance(value, np.ndarray) and value.dtype.kind == "O":
                    # Polars preserves a NumPy object array as Object, which
                    # cannot be cast back to Boolean/List/Struct. Infer from
                    # Python values while keeping genuine floating NaNs.
                    value = [None if cell is pd.NA or cell is pd.NaT else cell for cell in value]
                series = nw.from_dict({name: value}, backend=backend).get_column(name)
                if (not dtype.is_numeric() or series.null_count() == len(series)) and not isinstance(dtype, (nw.Decimal, nw.Unknown)):
                    series = series.cast(dtype)
            expressions.append(series)
    return _restore_native_fields(frame.with_columns(expressions).to_native(), data, values, schema)


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
        if is_polars_dataframe(data):
            data = data.clone()
        prediction = self.function(model, data)
        if isinstance(prediction, np.ndarray):
            return prediction
        return as_numpy_vector(prediction, "prediction")



class NativeResidualFunction:
    """Preserve the three-argument callback contract and normalize native vectors."""

    def __init__(self, function):
        self.function = function

    def __call__(self, model, data, y):
        if is_polars_dataframe(data):
            data = data.clone()
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
    backend = nw.get_native_namespace(frame)
    schema = {name: frame.get_column(name).dtype for name in names}
    columns_data = {}
    casts = []
    for i, name in enumerate(names):
        dtype = schema[name]
        values = [None if value is pd.NA or value is pd.NaT else value for value in array[:, i].tolist()]
        if isinstance(dtype, (nw.Datetime, nw.Duration)):
            columns_data[name] = nw.new_series(name, _temporal_counts(values, dtype), dtype=nw.Int64, backend=backend).cast(dtype).to_native()
        else:
            if isinstance(dtype, (nw.Categorical, nw.Enum, nw.String, nw.Boolean)):
                values = [None if pd.isna(value) else value for value in values]
            columns_data[name] = values
            if (not dtype.is_numeric() or all(value is None for value in values)) and not isinstance(dtype, (nw.Decimal, nw.Unknown)):
                casts.append(nw.col(name).cast(dtype))
    native = nw.from_dict(columns_data, backend=backend)
    result = native.with_columns(casts).to_native() if casts else native.to_native()
    return _restore_native_fields(result, reference, names, schema)
