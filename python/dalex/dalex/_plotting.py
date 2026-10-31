"""Native table preparation shared by explanation plots.

An explicitly read or edited pandas compatibility view remains authoritative.
Plotting must not create that view as a side effect of inspecting native results.
"""
import narwhals.stable.v2 as nw
import numpy as np

from ._explanation import export_result_frame


def descending_order(values):
    """Legacy pandas quicksort order, with missing numeric values last."""
    values = np.asarray(values)
    missing = np.isnan(values)
    positions = np.flatnonzero(~missing)[::-1]
    order = positions[np.argsort(values[positions], kind='quicksort')][::-1]
    return np.concatenate([order, np.flatnonzero(missing)])


def result_frame(explanation, *, table='result'):
    cached = explanation.__dict__.get('_compatibility_' + table)
    if cached is not None:
        return nw.from_native(cached, eager_only=True)
    original = explanation.__dict__.get(table)
    if callable(getattr(type(original), '_dalex_plot_data', None)):
        return nw.from_native(original._dalex_plot_data(), eager_only=True)
    return nw.from_native(explanation.result_as(table=table), eager_only=True)


def concat_frames(frames, *, how='vertical'):
    """Combine plot tables, converting only when their backends differ."""
    frames = list(frames)
    # Prefer an existing native backend for mixed-backend plots, so a pandas
    # companion never forces the native explanation through pandas.
    backend = next((nw.get_native_namespace(frame).__name__ for frame in frames
                    if nw.get_native_namespace(frame).__name__ != 'pandas'), 'pandas')
    frames = [nw.from_native(export_result_frame(frame.to_native(), backend), eager_only=True)
              if nw.get_native_namespace(frame).__name__ != backend else frame for frame in frames]
    return nw.concat(frames, how=how)
