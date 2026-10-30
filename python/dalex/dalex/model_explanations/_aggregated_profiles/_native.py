"""Narwhals profile reductions. Legacy display formatting lives at the boundary.

The input is a typed, long profile table. No pandas conversion is necessary for
means, Gaussian weighting, within-observation differences or accumulated effects.
The caller retains row-index and label-format metadata for its legacy public API.
"""
import numpy as np
import narwhals.stable.v2 as nw
from narwhals.dependencies import get_pandas
from ..._explanation import IndexedTable
from ...predict_explanations._ceteris_paribus._table import concat_compatible, export_frame


def _non_nan(expression):
    # Expr.fill_nan was introduced after Narwhals 2.0.
    return nw.when(expression.is_nan()).then(None).otherwise(expression)


def _ordered_aggregate(frame, keys, expressions):
    # Native group-by ordering is not a portable contract. Record first-observed
    # order explicitly, as pandas groupby(sort=False) does for result rows.
    ordinal = '__dalex_profile_position__'
    while ordinal in frame.columns:
        ordinal += '_'
    return (frame.with_row_index(ordinal).group_by(keys, drop_null_keys=False)
            .agg(*expressions, nw.col(ordinal).min()).sort(ordinal).drop(ordinal))


def _window(frame, keys, operation):
    expression = getattr(nw.col('_yhat_'), operation)().over(keys)
    namespace = nw.get_native_namespace(frame).__name__
    missing_groups = operation == 'cum_sum' and any(frame[key].null_count() for key in keys)
    if namespace != 'pyarrow' and not (namespace == 'pandas' and missing_groups):
        return frame.with_columns(expression)
    # Arrow supports elementary aggregate windows, but not diff/cumsum windows
    # (including at the declared Narwhals minimum). Use numeric vectors for this
    # operation only; the table and reductions retain the input backend. Older
    # Narwhals pandas windows also drop missing groups; the same fallback keeps
    # the legacy dropna=False cumulative-sum contract for those memberships.
    columns = frame.select(keys).to_dict(as_series=False)
    missing = object()
    positions = {}
    for i, cells in enumerate(zip(*(columns[key] for key in keys))):
        key = tuple(missing if value is None or isinstance(value, float) and np.isnan(value)
                    else value for value in cells)
        positions.setdefault(key, []).append(i)
    values = frame.get_column('_yhat_').to_numpy()
    output = np.empty_like(values)
    for key, rows in positions.items():
        chunk = values[rows]
        if operation == 'diff':
            output[rows] = 0. if missing in key else np.r_[0., np.diff(chunk)]
        else:
            absent = np.isnan(chunk)
            accumulated = np.cumsum(np.where(absent, 0., chunk))
            accumulated[absent] = np.nan
            output[rows] = accumulated
    return frame.with_columns(nw.new_series('_yhat_', output, backend=nw.get_native_namespace(frame)))


def reduce_profiles(data, kind, groups=(), *, span=.25):
    """Reduce one variable/model partition; retain typed group memberships.

    Returns a native result with ``_x_``, group columns and ``_yhat_``. Missing
    predictions follow legacy skip-NaN reductions, while missing grid keys remain
    keys. The caller rejects ambiguous legacy null/NaN bins before publication.
    """
    if kind not in ('partial', 'conditional', 'accumulated'):
        raise ValueError('Unknown profile type')
    frame = nw.from_native(data, eager_only=True)
    if not len(frame):
        raise ValueError('Cannot aggregate an empty profile partition')
    groups = list(groups)
    keys = ['_x_', *groups]
    numeric = frame.schema['_x_'].is_numeric()
    if kind == 'partial':
        frame = frame.with_columns(_non_nan(nw.col('_yhat_')).cast(frame.schema['_yhat_']).alias('_yhat_'))
        return _ordered_aggregate(frame, keys, [nw.col('_yhat_').mean()]).to_native()
    if numeric:
        grid = _non_nan(nw.col('_x_').cast(nw.Float64)).cast(nw.Float64)
        if kind == 'accumulated' and frame.select((grid.is_null()).any()).item():
            raise ValueError('Accumulated effects cannot order a missing value in a numeric grid')
        width = frame.select(grid.max() - grid.min()).item()
        if width == 0:
            width = 1.
        original = _non_nan(nw.col('_original_').cast(nw.Float64)).cast(nw.Float64)
        # Retain the legacy norm() operation order; backend reductions may
        # still differ by floating-point rounding, but avoid reassociation here.
        weights = (-1 * (((original - grid) / width) / span) ** 2 / 2).exp() / np.sqrt(2 * np.pi) / span
        if kind == 'conditional' and frame.select(grid.is_null().any()).item():
            weights = nw.when(original.is_null() & grid.is_null()).then(1.).otherwise(weights.fill_null(0.))
    else:
        weights = nw.col('_original_') == nw.col('_x_')
    frame = frame.with_columns(weights.cast(nw.Float64).alias('__weight__'))
    if kind == 'accumulated':
        frame = _window(frame, ['_ids_'], 'diff')
        frame = frame.with_columns(_non_nan(nw.col('_yhat_')).fill_null(0.).cast(nw.Float64).alias('_yhat_'))
    products = _non_nan(nw.col('_yhat_') * nw.col('__weight__')).cast(nw.Float64)
    frame = frame.with_columns(products.alias('__product__'))
    result = _ordered_aggregate(frame, keys, [nw.col('__product__').sum().alias('__numerator__'),
                                             nw.col('__weight__').sum().alias('__denominator__')])
    divisor = nw.when(nw.col('__denominator__') != 0).then(nw.col('__denominator__')).otherwise(None)
    result = result.with_columns((nw.col('__numerator__') / divisor).fill_null(0.).alias('_yhat_'))
    result = result.drop('__numerator__', '__denominator__')
    if kind == 'accumulated':
        if groups:
            result = _window(result, groups, 'cum_sum')
        else:
            result = result.with_columns(nw.col('_yhat_').cum_sum())
    return result.to_native()


def _missing(value):
    pandas = get_pandas()
    return value is None or (pandas is not None and (value is pandas.NA or value is pandas.NaT)) or (
        isinstance(value, (float, np.floating)) and np.isnan(value))


def _numeric_column(frame, name):
    dtype = frame.schema[name]
    if isinstance(dtype, nw.Decimal):
        return False  # Legacy Decimal object columns are categorical.
    if dtype.is_numeric() or isinstance(dtype, nw.Boolean):
        return True
    if isinstance(dtype, (nw.Object, nw.Unknown)):
        values = [value for value in frame[name].to_list() if not _missing(value)]
        return bool(values) and np.asarray(values).dtype.kind in 'biufc'
    return False


def _time_label(count, dtype):
    if count is None:
        return 'NaT'
    count = int(count) * {'s': 10**9, 'ms': 10**6, 'us': 1000, 'ns': 1}[dtype.time_unit]
    pandas = get_pandas()
    if pandas is not None:
        if isinstance(dtype, nw.Duration):
            return str(pandas.Timedelta(count, unit='ns'))
        value = pandas.Timestamp(count, unit='ns', tz='UTC') if dtype.time_zone else pandas.Timestamp(count, unit='ns')
        return str(value.tz_convert(dtype.time_zone) if dtype.time_zone else value)
    # Native-only installations still need precise scalar display labels.
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    if isinstance(dtype, nw.Duration):
        days, rest = divmod(count, 86400 * 10**9)
        seconds, fraction = divmod(rest, 10**9)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)
        text = f'{days} days ' + ('+' if days < 0 and rest else '') + f'{hours:02}:{minutes:02}:{seconds:02}'
        return text + (f'.{fraction:09}' if fraction else '')
    seconds, fraction = divmod(count, 10**9)
    value = datetime.fromtimestamp(seconds, timezone.utc)
    if dtype.time_zone:
        value = value.astimezone(ZoneInfo(dtype.time_zone))
    text = value.strftime('%Y-%m-%d %H:%M:%S')
    if fraction:
        text += f'.{fraction:09}' if fraction % 1000 else f'.{fraction // 1000:06}'
    offset = value.strftime('%z') if dtype.time_zone else ''
    return text + (offset[:3] + ':' + offset[3:] if offset else '')


def _keys_labels(frame, variable, integer_labels=False):
    column = frame.get_column(variable)
    dtype = column.dtype
    if isinstance(dtype, (nw.Datetime, nw.Duration)):
        absent = column.is_null().to_numpy()
        counts = column.cast(nw.Int64).to_list()
        values = [None if missing else int(value) for value, missing in zip(counts, absent)]
        return values, [_time_label(value, dtype) for value in values], 'NaT'
    values = column.to_list()
    missing_label = ('nan' if isinstance(dtype, (nw.Categorical, nw.Enum)) else
                     '<NA>' if isinstance(dtype, (nw.String, nw.Boolean)) or integer_labels else
                     'nan' if dtype.is_numeric() else 'None')
    labels = [missing_label if _missing(value) else str(int(value)) if integer_labels else str(value) for value in values]
    return values, labels, missing_label


def _integer_display(frames, variable):
    # CP's legacy convert_dtypes turns wholly integral floating columns into
    # nullable integers. Inspect all partitions, including unmodified cells.
    for frame in frames:
        if variable not in frame.columns:
            continue
        values = frame[variable].to_list()
        for value in values:
            if not _missing(value) and (not isinstance(value, (int, float, np.integer, np.floating))
                                        or isinstance(value, (bool, np.bool_)) or value != np.floor(value)):
                return False
    return True


class AggregatedTable(IndexedTable):
    """Format legacy group labels only at the public pandas boundary."""

    def __init__(self, data, index, groups, group_dtypes):
        super().__init__(data, index)
        self.groups = list(groups)
        self.group_dtypes = group_dtypes

    def _dalex_pandas_view(self):
        result = super()._dalex_pandas_view()
        for group in self.groups:
            dtype = self.group_dtypes.get(group)
            if dtype == 'object':
                result[group] = result[group].astype(object).where(result[group].notna(), np.nan)
            elif dtype is not None:
                result[group] = result[group].astype(dtype)
        if self.groups:
            result['_groups_'] = result.loc[:, self.groups].apply(lambda row: '_'.join(map(str, row)), axis=1)
            result = result.drop(columns=self.groups)
            result['_label_'] = result.loc[:, ['_label_', '_groups_']].apply(lambda row: '_'.join(row), axis=1)
        return result


def aggregate_ceteris(profiles, variables, variable_type, kind, groups, span, center):
    """Compute aggregated explanations directly from typed CP partitions."""
    entries = [(variable, nw.from_native(data)) for cp in profiles for variable, data in cp._native_profile_frames()]
    all_variables = list(dict.fromkeys(variable for variable, _ in entries))
    if variables is not None:
        selected = set(variables).intersection(all_variables)
        if not selected:
            raise ValueError('variables do not overlap with profiles')
    else:
        selected = set(all_variables)
    numerical = {variable for variable, frame in entries if _numeric_column(frame, variable)}
    selected &= numerical if variable_type == 'numerical' else selected if variables is not None else set(all_variables) - numerical
    if not selected:
        raise ValueError('There are no numerical variables' if variable_type == 'numerical' else 'There are no non-numerical variables')
    from .utils import check_native_missing_interventions
    check_native_missing_interventions(profiles, list(selected), kind)
    predictions = [nw.from_native(cp._native_observation_frame())['_yhat_'].to_numpy() for cp in profiles]
    mean_prediction = np.nanmean(np.concatenate(predictions))
    target = nw.get_native_namespace(entries[0][1]).__name__
    groups = list(groups)
    group_dtypes = {}
    for _, frame in entries:
        for group in groups:
            if group not in frame.columns:
                raise KeyError(group)
            dtype = frame.schema[group]
            if isinstance(dtype, (nw.Categorical, nw.Enum)):
                group_dtypes[group] = 'object'
            elif nw.get_native_namespace(frame).__name__ == 'pandas':
                group_dtypes.setdefault(group, frame.to_native()[group].dtype)
            elif isinstance(dtype, (nw.String, nw.Boolean, nw.Object, nw.Unknown)):
                group_dtypes.setdefault(group, 'object')
    # Category labels are global to a variable, even across model labels/CPs.
    category_maps = {}
    if variable_type == 'categorical':
        frames = [frame for _, frame in entries]
        for variable in selected:
            integer_labels = _integer_display(frames, variable)
            levels, missing_labels, nonmissing = {}, [], set()
            for name, frame in entries:
                if name != variable:
                    continue
                values, labels, missing_label = _keys_labels(frame, variable, integer_labels)
                for value, label in zip(values, labels):
                    if _missing(value):
                        missing_labels.append(missing_label)
                    else:
                        levels[value] = label
                        nonmissing.add(label)
            missing_label = missing_labels[0] if missing_labels else '<missing>'
            while missing_label in nonmissing:
                missing_label += ' (missing)'
            category_maps[variable] = (levels, missing_label, integer_labels)
    partitions = {}
    for variable, frame in entries:
        if variable not in selected:
            continue
        if variable_type == 'numerical':
            frame = frame.with_columns(frame.get_column(variable).cast(nw.Float64).alias('_x_'))
        else:
            levels, missing_label, integer_labels = category_maps[variable]
            values, labels, _ = _keys_labels(frame, variable, integer_labels)
            originals, original_labels, _ = _keys_labels(frame, '_original_', False)
            labels = [missing_label if _missing(value) else label for value, label in zip(values, labels)]
            original_labels = [missing_label if _missing(value) else levels.get(value, label)
                               for value, label in zip(originals, original_labels)]
            namespace = nw.get_native_namespace(frame)
            frame = frame.with_columns(nw.new_series('_x_', labels, backend=namespace),
                                       nw.new_series('_original_', original_labels, backend=namespace))
        for group in groups:
            dtype = frame.schema[group]
            if isinstance(dtype, (nw.Categorical, nw.Enum)):
                frame = frame.with_columns(nw.col(group).cast(nw.String))
            elif isinstance(dtype, (nw.Float32, nw.Float64)):
                frame = frame.with_columns(_non_nan(nw.col(group)).cast(dtype).alias(group))
        keep = list(dict.fromkeys(['_vname_', '_label_', '_x_', '_yhat_', '_ids_', '_original_', *groups]))
        frame = frame.select(keep)
        for key, split in frame.group_by(['_vname_', '_label_'], drop_null_keys=True):
            key = tuple(value.as_py() if hasattr(value, 'as_py') else value for value in key)
            partitions.setdefault(key, []).append(export_frame(split.to_native(), target))
    ordered = list(partitions)
    if kind != 'partial':
        ordered.sort(key=lambda key: tuple((isinstance(value, str), value) for value in key))
    results, indexes = [], []
    for variable, label in ordered:
        joined = concat_compatible(partitions[(variable, label)])
        reduced = nw.from_native(reduce_profiles(joined, kind, groups, span=span))
        reduced = reduced.with_columns(nw.lit(variable).alias('_vname_'), nw.lit(label).alias('_label_'),
                                        nw.lit(0).cast(nw.Int64).alias('_ids_'))
        results.append(reduced.select(['_vname_', '_label_', '_x_', *groups, '_yhat_', '_ids_']).to_native())
        indexes.extend(range(len(reduced)))
    combined = nw.from_native(concat_compatible(results))
    if kind == 'accumulated' and center:
        average = combined.select(_non_nan(nw.col('_yhat_')).mean()).item()
        combined = combined.with_columns((nw.col('_yhat_') - average + mean_prediction).alias('_yhat_'))
    index = None if kind == 'partial' else np.asarray(indexes)
    return AggregatedTable(combined.to_native(), index, groups, group_dtypes), mean_prediction
