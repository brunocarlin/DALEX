import numpy as np
import pandas as pd
from .._dataframe import to_pandas
import warnings

from . import checks
from .. import _global_checks

def calculate_cat_assoc_matrix(data, categorical_variables, n):
    from scipy.stats import chi2_contingency
    cat_assoc_matrix = pd.DataFrame(
        index=categorical_variables, columns=categorical_variables, dtype=float
    )
    for i, variable_name_a in enumerate(categorical_variables):
        for j, variable_name_b in enumerate(categorical_variables):
            if i > j:
                continue   # matrix is symmetric
            elif i == j:
                cramers_V = 1
            else:
                # calculate Cramér’s V with bias correction
                cont_tab = pd.crosstab(data[variable_name_a], data[variable_name_b])
                r = cont_tab.shape[0]
                c = cont_tab.shape[1]
                chi2_test = chi2_contingency(cont_tab)
                phi = max(0, (chi2_test[0] / n) - (((r - 1) * (c - 1)) / (n - 1)))
                r = r - ((r - 1) ** 2 / (n - 1))
                c = c - ((c - 1) ** 2 / (n - 1))
                cramers_V = np.sqrt(phi / (min(c, r) - 1))
                cramers_V = checks.check_assoc_value(cramers_V)
            cat_assoc_matrix.loc[variable_name_a, variable_name_b] = cramers_V
            cat_assoc_matrix.loc[variable_name_b, variable_name_a] = cramers_V
    return cat_assoc_matrix


def calculate_cat_num_assoc_matrix(data, categorical_variables, numerical_variables, n):
    from scipy.stats import kruskal
    cat_num_assoc_matrix = pd.DataFrame(
        index=categorical_variables, columns=numerical_variables, dtype=float
    )
    for cat_variable_name in categorical_variables:
        for num_variable_name in numerical_variables:
            # calculate eta-squared
            samples = [
                gr_data[num_variable_name].values
                for gr_name, gr_data in data.groupby(cat_variable_name, as_index=False)
            ]
            kruskal_wallis_test = kruskal(*samples, nan_policy="omit")
            n_samples = len(samples)
            eta_squared = (kruskal_wallis_test.statistic - n_samples + 1) / (n - n_samples)
            eta_squared = checks.check_assoc_value(eta_squared)
            cat_num_assoc_matrix.loc[cat_variable_name, num_variable_name] = eta_squared
    return cat_num_assoc_matrix


def calculate_assoc_matrix(data, corr_method, *, native=False):
    from ._native_dependence import associations
    table = associations(data, corr_method)
    return table if native else table._dalex_pandas_view()


def calculate_pps_matrix(data, agg_method):
    # PPS is an optional pandas-only statistics library, deliberately excluded
    # from the full extra because its dependency bounds can conflict with pandas 2.
    try:
        import ppscore as pps
    except ImportError as exc:
        raise ImportError("Install optional 'ppscore' for PPS dependence matrices") from exc
    pps_result = pps.matrix(to_pandas(data), sample=None)
    pps_matrix = pps_result[["x", "y", "ppscore"]].pivot(
        columns="x", index="y", values="ppscore"
    )
    pps_matrix.rename_axis(None, axis=1, inplace=True)
    pps_matrix.rename_axis(None, axis=0, inplace=True)
    # aggregate values to make symmetric matrix
    if agg_method == "max":
        pps_matrix = np.maximum(pps_matrix, pps_matrix.transpose())
    if agg_method == "min":
        pps_matrix = np.minimum(pps_matrix, pps_matrix.transpose())
    if agg_method == "mean":
        pps_matrix = (pps_matrix + pps_matrix.transpose()) / 2
    return pps_matrix


def calculate_depend_matrix(data, depend_method, corr_method, agg_method, *, native=False):
    from ._native_dependence import normalize_missing
    if depend_method == "assoc":
        table = calculate_assoc_matrix(data, corr_method, native=True)
    elif depend_method == "pps":
        table = calculate_pps_matrix(data, agg_method)
    elif callable(depend_method):
        try:
            table = depend_method(data)
        except Exception as exc:
            raise ValueError("You have passed wrong callable in depend_method argument. "
                             "'depend_method' is the callable to use for calculating dependency matrix.") from exc
    else:
        raise ValueError('Unknown dependence method')
    table = normalize_missing(table)
    if native:
        return table
    return table._dalex_pandas_view() if hasattr(type(table), '_dalex_pandas_view') else table


def calculate_linkage_matrix(depend_matrix, clust_method):
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import squareform
    from ._native_dependence import matrix_values
    values, names = matrix_values(depend_matrix)
    dissimilarity = 1 - abs(values)
    ## https://www.kaggle.com/sgalella/correlation-heatmaps-with-hierarchical-clustering?scriptVersionId=24045077&cellId=10
    linkage_matrix = linkage(squareform(dissimilarity), clust_method)

    return linkage_matrix


def get_dendrogram_aspects_ordered(hierarchical_clustering_dendrogram, depend_matrix, *, native=False):
    # names of variables on axis
    tick_dict = dict(
        zip(
            hierarchical_clustering_dendrogram.layout.yaxis.tickvals,
            [[var] for var in hierarchical_clustering_dendrogram.layout.yaxis.ticktext]
        )
    )
    from ._native_dependence import matrix_values
    _, names = matrix_values(depend_matrix)
    d = {k:v for v,k in enumerate(names)} # original order of columns
    
    # get names of variables for dendrogram traces
    _aspects_dendrogram_order = []
    for scatter in hierarchical_clustering_dendrogram.data:
        vars_list = tick_dict[scatter.y[1]] + tick_dict[scatter.y[2]]
        vars_list.sort(key=d.get)
        tick_dict[np.mean(scatter.y[1:3])] = vars_list
        _aspects_dendrogram_order.append(
            vars_list
        )
    
    # get min dependency between grouped variables
    _vars_min_depend, _min_depend = get_min_depend_from_matrix(depend_matrix, _aspects_dendrogram_order)

    if not native:
        return pd.DataFrame({"variable_names": _aspects_dendrogram_order,
                             "min_depend": _min_depend, "vars_min_depend": _vars_min_depend})
    import narwhals.stable.v2 as nw
    from .._explanation import IndexedTable
    reference = depend_matrix.data if isinstance(depend_matrix, IndexedTable) else depend_matrix
    backend = nw.get_native_namespace(nw.from_native(reference, eager_only=True))
    return IndexedTable(nw.from_dict({"variable_names": _aspects_dendrogram_order,
                                      "min_depend": _min_depend,
                                      "vars_min_depend": [x.tolist() for x in _vars_min_depend]},
                                     backend=backend).to_native())


def get_min_depend_from_matrix(depend_matrix, variables_list):
    from ._native_dependence import matrix_values
    values, names = matrix_values(depend_matrix)
    positions = {name: i for i, name in enumerate(names)}
    minima, pairs = [], []
    for variables in variables_list:
        if isinstance(variables, (list, tuple, np.ndarray)):
            ids = [positions[name] for name in variables]
            sub = abs(values[np.ix_(ids, ids)])
            minimum = sub.min()
            i, j = np.argwhere(sub == minimum)[0]
            pairs.append(np.asarray([variables[i], variables[j]]))
            minima.append(minimum)
        else:
            pairs.append(None)
            minima.append(None)
    return pairs, minima


def calculate_min_depend(
    variables_list,
    data,
    depend_method="assoc",
    corr_method="spearman",
    agg_method="max",
): 
    depend_matrix = calculate_depend_matrix(data, depend_method, corr_method, agg_method, native=True)
    return get_min_depend_from_matrix(depend_matrix, variables_list)
