import multiprocessing as mp

import numpy as np
import pandas as pd
import narwhals.stable.v2 as nw
from ..._dataframe import as_eager_dataframe, column_values, set_columns, take_rows, to_pandas
from tqdm import tqdm


def calculate_ceteris_paribus(explainer,
                              new_observation,
                              variable_splits,
                              y,
                              processes,
                              verbose=True):
    """
    Inner method that calculates ceteris paribus and some additional fields.

    Set profile and new_profile fields to actual values.

    :return: None
    """

    profiles = calculate_variable_profile(explainer.predict_function,
                                          explainer.model,
                                          new_observation,
                                          variable_splits,
                                          processes,
                                          verbose)

    profiles.loc[:, '_label_'] = explainer.label

    # add points of interests
    predictions = explainer.predict(new_observation)

    new_observation = to_pandas(new_observation)
    new_observation.loc[:, '_yhat_'] = predictions

    new_observation.loc[:, '_label_'] = explainer.label

    new_observation.loc[:, '_ids_'] = new_observation.index.values

    if y is not None:
        new_observation.loc[:, '_y_'] = y

    return profiles, new_observation


def calculate_variable_profile(predict_function,
                               model,
                               data,
                               variable_splits,
                               processes,
                               verbose=True):
    """
    Inner function that calculates ceteris paribus for all variables.

    :param explainer:
    :param data: new observations
    :param variable_splits: dictionary of split points
    :return: ceteris paribus profile for all variable
    """

    if processes == 1:
        profile = []
        for variable in tqdm(variable_splits, desc="Calculating ceteris paribus", disable=not verbose):
            split_points = variable_splits[variable]
            profile.append(single_variable_profile(predict_function, model, data, variable, split_points))
    else:
        with mp.get_context('spawn').Pool(processes) as pool:

            profile = pool.starmap_async(single_variable_profile, [(predict_function, model, data, variable, split_points)
                                                                   for variable, split_points in
                                                                   variable_splits.items()]).get()

    profiles = pd.concat(profile)
    # convert the variable types
    # convert_floating=False since pandas v1.2 seem to have issues
    converted = profiles.loc[:, list(variable_splits)].convert_dtypes(convert_floating=False)
    for variable, column in converted.items():
        # Explicit positional assignment supports pandas 3's strict dtype rules
        # and duplicated profile indexes without index alignment.
        profiles[variable] = column.array

    return profiles


def single_variable_profile(predict,
                            model,
                            data,
                            variable,
                            split_points):
    """
    Inner function for calculating variable profile. This function is iterated over all variables.

    :param predict:
    :param data: new observations
    :param variable: considered variable
    :param split_points: dataset is split over these points
    :return: ceteris paribus profile for one variable
    """
    if not isinstance(data, pd.DataFrame):
        positions = np.repeat(np.arange(data.shape[0], dtype=np.int64), len(split_points))
        expanded = take_rows(data, positions)
        original = column_values(expanded, variable)
        replacements = np.tile(split_points, data.shape[0])
        if as_eager_dataframe(data).get_column(variable).dtype.is_numeric():
            # Match the legacy pandas profile path, which deliberately widens
            # numeric interventions to float even for integer split points.
            # Keep explicit native null interventions separate from NaN.
            if replacements.dtype.kind == "O" and any(
                value is None or value is pd.NA for value in replacements
            ):
                replacements = np.array([
                    None if value is None or value is pd.NA else float(value)
                    for value in replacements
                ], dtype=object)
            else:
                replacements = replacements.astype(float)
        native = set_columns(expanded, {variable: replacements})
        yhat = predict(model, native)
        result = to_pandas(native)
        result.index = positions
        return result.assign(_original_=original, _yhat_=yhat, _vname_=variable, _ids_=positions)

    # remember ids of selected points
    ids = np.repeat(data.index.values, split_points.shape[0])
    new_data = data.loc[ids, :]
    original = new_data.loc[:, variable].copy()
    if new_data[variable].dtype.kind in 'iufc':
        try:
            new_data[variable] = new_data[variable].astype('float')
        except TypeError:
            if not isinstance(new_data[variable].dtype, pd.ArrowDtype):
                raise
            new_data[variable] = np.asarray(new_data[variable].array.__arrow_array__().to_numpy(), dtype=float)
    new_data.loc[:, variable] = np.tile(split_points, data.shape[0])

    yhat = predict(model, new_data)

    new_data.loc[:, '_original_'] = original
    new_data.loc[:, '_yhat_'] = yhat
    new_data.loc[:, '_vname_'] = variable
    new_data.loc[:, '_ids_'] = ids

    return new_data


def native_numeric_missing_states(data, variable_splits):
    """Keep validity provenance outside the backward-compatible result table."""
    if isinstance(data, pd.DataFrame):
        return {}
    frame = as_eager_dataframe(data)
    states = {}
    for variable, splits in variable_splits.items():
        if not any(value is None or value is pd.NA or value is pd.NaT or
                   (isinstance(value, (float, np.floating)) and np.isnan(value)) for value in splits):
            continue
        column = frame.get_column(variable)
        kinds = set()
        if column.dtype.is_numeric():
            if column.null_count():
                kinds.add('null')
            if isinstance(column.dtype, (nw.Float32, nw.Float64)) and column.is_nan().any():
                kinds.add('nan')
        states[variable] = kinds
    return states


def calculate_variable_split(data,
                             variables,
                             grid_points,
                             variable_splits_type='uniform',
                             variable_splits_with_obs=False,
                             new_observation=None):
    """
    Calculate points for splitting the dataset

    :param data: dataset to split
    :param variables: variables to calculate ceteris paribus
    :param grid_points: how many points should split the dataset
    :param variable_splits_type: {'uniform', 'quantiles'}, optional way of calculating
        `variable_splits`. Set 'quantiles' for percentiles.
    :param variable_splits_with_obs: bool, optional add variable values of `new_observation`
        data to the `variable_splits`.
    :param new_observation: pd.DataFrame or np.ndarray, Observations for which predictions
        need to be explained.
    :return: dict, dictionary of split points for all variables
    """
    variable_splits = {}
    # grid points might be larger than the number of unique values
    probs = np.linspace(0, 1, grid_points)

    for variable in variables:
        column = as_eager_dataframe(data).get_column(variable)
        dtype = column.dtype
        # pandas 1.5 does not classify numeric ArrowDtype via is_numeric_dtype.
        numeric = data[variable].dtype.kind in 'iufc' if isinstance(data, pd.DataFrame) else (
            dtype.is_numeric() and not isinstance(dtype, nw.Decimal))
        if numeric:
            # Reference datasets can contain millions of rows even when only
            # a small CP sample is explained. Keep missing filtering and range
            # reductions native; exporting/scanning Python cells made grid
            # construction dominate the entire sampled explanation.
            observed = column.drop_nulls()
            if dtype.is_float():
                observed = observed.filter(~observed.is_nan())
            if not len(observed):
                # Native float validity distinguishes null from NaN. An
                # all-missing column can therefore have two interventions.
                if not isinstance(data, pd.DataFrame):
                    count = column.null_count()
                    levels = ([None] if count else []) + ([np.nan] if count < len(column) else [])
                    column_splits = np.array(levels, dtype=object if count else float)
                else:
                    column_splits = np.array([np.nan])
            elif variable_splits_type == 'uniform':
                column_splits = np.linspace(observed.min(),
                                            observed.max(),
                                            grid_points)
            else:
                column_splits = np.unique(np.quantile(observed.to_numpy(), probs))
            if variable_splits_with_obs:
                values = np.concatenate((column_splits, column_values(new_observation, variable, preserve_nulls=True)))
                nulls = [value is None or value is pd.NA or value is pd.NaT for value in values]
                if any(nulls):
                    numeric = np.unique(np.asarray([value for value, absent in zip(values, nulls) if not absent], dtype=float))
                    if isinstance(data, pd.DataFrame):
                        column_splits = np.unique(np.append(numeric, np.nan))
                    else:
                        missing_nan = np.isnan(numeric)
                        column_splits = np.array(numeric[~missing_nan].tolist() + [None]
                                                 + ([np.nan] if missing_nan.any() else []), dtype=object)
                else:
                    column_splits = np.sort(np.unique(values), kind='mergesort')

            variable_splits[variable] = column_splits
        else:
            # Narwhals keeps the first-observed order without a pandas Series
            # allocation. Use the exact scalar export for dictionary/temporal
            # columns rather than a backend's lossy generic NumPy conversion.
            unique = column.unique(maintain_order=True).to_frame().to_native()
            variable_splits[variable] = column_values(unique, variable, preserve_nulls=True)

    return variable_splits
