import numpy as np
import pandas as pd
from tqdm import tqdm


def aggregate_profiles(all_profiles, mean_prediction, type, groups, center, span, verbose=True):
    if type == 'accumulated' and pd.api.types.is_numeric_dtype(all_profiles['_x_']) and all_profiles['_x_'].isna().any():
        raise ValueError(
            "Accumulated effects cannot order a missing value in a numeric grid; "
            "use a finite grid, partial profiles or conditional profiles."
        )
    for group in groups:
        if isinstance(all_profiles[group].dtype, pd.CategoricalDtype):
            # pandas 1.5 ignores categorical null groups even with dropna=False;
            # unused category levels also create fictitious empty aggregates.
            # Only observed memberships belong in the result (groups are later
            # serialized to labels, so categorical storage is not part of API).
            all_profiles[group] = all_profiles[group].astype(object)
    if type == 'partial':
        aggregated_profiles = \
            all_profiles.groupby(['_vname_', '_label_', '_x_'] + groups, sort=False, dropna=False)['_yhat_'].mean().reset_index()

    else:
        # split all_profiles into groups
        tqdm.pandas(desc='Calculating accumulated dependency', disable=not verbose) if type == 'accumulated' else tqdm.pandas(
            desc="Calculating conditional dependency", disable=not verbose)
        aggregated_profiles = \
            all_profiles. \
                loc[:, ["_vname_", "_label_", "_x_", "_yhat_", "_ids_", "_original_"] + groups]. \
                groupby(['_vname_', '_label_'])[["_x_", "_yhat_", "_ids_", "_original_"] + groups]. \
                progress_apply(lambda split_profile: split_over_variables_and_labels(split_profile.copy(deep=True),
                                                                                     type, groups, span)). \
                reset_index(level=[0, 1])  # remove level_2
        # deepcopy due to https://github.com/ModelOriented/DALEX/issues/278

    aggregated_profiles.loc[:, '_ids_'] = 0

    if type == 'accumulated' and center:
        aggregated_profiles.loc[:, '_yhat_'] = aggregated_profiles.loc[:, '_yhat_'] - \
                                               aggregated_profiles.loc[:, '_yhat_'].mean() + \
                                               mean_prediction

    # postprocessing
    if len(groups) != 0:
        aggregated_profiles['_groups_'] = aggregated_profiles.loc[:, groups].apply(
            lambda row: '_'.join(map(str, row)), axis=1)
        aggregated_profiles = aggregated_profiles.drop(columns=groups)

        aggregated_profiles.loc[:, '_label_'] = \
            aggregated_profiles.loc[:, ['_label_', '_groups_']].apply(lambda row: '_'.join(row), axis=1)

    return aggregated_profiles


def split_over_variables_and_labels(split_profile, type, groups, span):
    """
    Inner function that calculates actual conditional profiles for one variable only. Iterated over each variable and group.

    :param split_profile: pandas.DataFrame, one group of the dataset (with only one variable)
    :param groups: str, name of grouping variable
    :return: pd.DataFrame, dataframe with calculated conditional profile for only one variable
    """

    if split_profile.shape[0] == 0:
        return None

    if pd.api.types.is_numeric_dtype(split_profile['_x_']):
        # for continuous variables we will calculate weighted average
        # where weights come from gaussian kernel and distance between points
        # scaling factor, range if the range i > 0
        split_profile['_original_'] = split_profile['_original_'].astype('float')
        range_x = split_profile['_x_'].max() - split_profile['_x_'].min()

        if range_x == 0:
            range_x = 1

        # scalled differences
        diffs = (split_profile['_original_'] - split_profile['_x_']) / range_x

        split_profile['_w_'] = norm(diffs, 0, span)
        if type == 'conditional' and split_profile['_x_'].isna().any():
            # Missing interventions form a separate bin: numeric distance to a
            # missing original is undefined, while two missing values match.
            both_missing = split_profile['_original_'].isna() & split_profile['_x_'].isna()
            split_profile['_w_'] = split_profile['_w_'].fillna(0).mask(both_missing, 1.)

    else:
        # for categorical variables we will calculate weighted average
        # but weights are 0-1, 1 if it's the same level and 0 otherwise
        split_profile['_w_'] = split_profile['_original_'] == split_profile['_x_']

    if type == 'accumulated':
        # diffs
        split_profile['_yhat_'] = split_profile. \
            groupby('_ids_')['_yhat_']. \
            transform(lambda column: column.diff())

        # diff causes NaNs at the beginning of each group
        split_profile.loc[np.isnan(split_profile['_yhat_']), '_yhat_'] = 0

    par_profile = split_profile.groupby(['_x_'] + groups, sort=False, dropna=False)[['_yhat_', '_w_']]. \
        apply(lambda point: (point['_yhat_'] * point['_w_']).sum() / point['_w_'].sum() \
        if point['_w_'].sum() != 0 else 0)

    par_profile.name = '_yhat_'
    par_profile = par_profile.reset_index()

    if type == 'accumulated':
        if len(groups) == 0:
            par_profile['_yhat_'] = par_profile['_yhat_'].cumsum()
        else:
            par_profile['_yhat_'] = par_profile.groupby(groups, sort=False, dropna=False)['_yhat_'].transform(
                lambda column: column.cumsum())

    return par_profile


def norm(x, loc, scale):
    return np.exp(-1 * ((x - loc) / scale) ** 2 / 2) / np.sqrt(2 * np.pi) / scale


def prepare_numerical_categorical(all_profiles, variables, variable_type):
    # variables to use
    all_variables = all_profiles['_vname_'].dropna().unique()  # variables do not need to be string

    if variables is not None:
        all_variables_intersect = set(all_variables).intersection(set(variables))
        if len(all_variables_intersect) == 0:
            raise ValueError("variables do not overlap with " + all_variables)
        all_variables = np.array(list(all_variables_intersect))

    # only numerical or only factors?
    is_numeric = np.empty_like(all_variables, bool)
    for i, var in enumerate(all_variables):
        is_numeric[i] = pd.api.types.is_numeric_dtype(all_profiles[var])

    if variable_type == 'numerical':
        vnames = all_variables[is_numeric]
        if vnames.shape[0] == 0:
            raise ValueError("There are no numerical variables")

        all_profiles['_x_'] = 0.0

    else:
        vnames = all_variables[~is_numeric]
        if variables is not None:
            # take all if the user wants to convert numerical into categorical
            vnames = all_variables
        elif vnames.shape[0] == 0:
            raise ValueError("There are no non-numerical variables")

        all_profiles['_x_'] = ""

    return all_profiles, vnames


def prepare_x(all_profiles, variable_type):
    if variable_type == 'categorical':
        labels, original_labels = categorical_profile_labels(all_profiles)
        # Complete-column assignment also supports pandas 3's StringDtype.
        all_profiles['_x_'] = labels
        all_profiles['_original_'] = original_labels
        return all_profiles
    # create _x_
    for variable in all_profiles['_vname_'].unique():
        where_variable = all_profiles['_vname_'] == variable
        # Normalize nullable masks before assigning into the float grid. Pandas
        # 3 rejects putting an IntegerArray containing pd.NA into that column.
        all_profiles.loc[where_variable, '_x_'] = all_profiles.loc[
            where_variable, variable
        ].to_numpy(dtype=float, na_value=np.nan)

    return all_profiles


def categorical_profile_labels(profiles):
    """Match categorical levels without merging missing and literal labels.

    Numeric codes and nullable booleans can have different pandas dtypes in the
    intervention/original columns. Missing values share the grid's label, with
    a suffix only if a real category already uses that display string.
    """
    names = profiles['_vname_'].to_numpy()
    original = profiles['_original_'].to_numpy()
    labels = np.empty(len(profiles), dtype=object)
    original_labels = np.empty(len(profiles), dtype=object)

    def missing(value):
        return pd.api.types.is_scalar(value) and pd.isna(value)

    for variable in pd.unique(names):
        positions = np.flatnonzero(names == variable)
        values = profiles[variable].to_numpy()[positions]
        strings = [str(value) for value in values]
        missing_mask = [missing(value) for value in values]
        nonmissing = {label for label, absent in zip(strings, missing_mask) if not absent}
        # Numeric interventions can become nullable integers while originals
        # remain floats. Match their values before choosing display strings.
        level_labels = {value: label for value, label, absent in zip(values, strings, missing_mask) if not absent}
        missing_label = next((label for label, absent in zip(strings, missing_mask) if absent), '<missing>')
        while missing_label in nonmissing:
            missing_label += ' (missing)'
        labels[positions] = [missing_label if absent else label for label, absent in zip(strings, missing_mask)]
        original_labels[positions] = [missing_label if missing(value) else level_labels.get(value, str(value))
                                      for value in original[positions]]
    return labels, original_labels
