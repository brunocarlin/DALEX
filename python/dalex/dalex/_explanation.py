from abc import ABC, abstractmethod
import narwhals.stable.v2 as nw
from narwhals.dependencies import is_pandas_dataframe


class CompatibilityTable:
    """Materialize legacy pandas tables only when their public attribute is read.

    Store under the historical attribute name so older pickles remain readable.
    A materialized pandas view is authoritative: callers have always been able
    to mutate ``explanation.result`` in place.
    """

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        value = instance.__dict__.get(self.name)
        if value is None or is_pandas_dataframe(value):
            return value
        key = '_compatibility_' + self.name
        if key not in instance.__dict__:
            try:
                frame = nw.from_native(value, eager_only=True)
            except TypeError:
                return value  # SHAP arrays and other non-tabular results
            if not isinstance(frame, nw.DataFrame):
                return value
            instance.__dict__[key] = frame.to_pandas()
        return instance.__dict__[key]

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value
        instance.__dict__.pop('_compatibility_' + self.name, None)

class Explanation(ABC):
    result = CompatibilityTable()

    def result_as(self, backend='native', *, table='result'):
        """Export an independent result table without changing the legacy API.

        ``native`` retains the table's computation backend. Explicit backends
        are ``pandas``, ``polars`` and ``pyarrow``. pandas indexes remain indexes
        on pandas exports; they are not silently added as feature columns on
        other backends. In-place edits to the public pandas table are respected.
        Non-tabular results can only be exported with ``backend='native'``.
        """
        if backend not in ('native', 'pandas', 'polars', 'pyarrow'):
            raise ValueError("backend must be 'native', 'pandas', 'polars' or 'pyarrow'")
        if not isinstance(getattr(type(self), table, None), CompatibilityTable):
            raise ValueError(f'No exportable table {table!r}')
        original = self.__dict__.get(table)
        value = self.__dict__.get('_compatibility_' + table, original)
        if value is None:
            return None
        try:
            frame = nw.from_native(value, eager_only=True)
        except TypeError:
            if backend != 'native':
                raise TypeError('This explanation result is not a dataframe') from None
            return value.copy() if hasattr(value, 'copy') else value
        if not isinstance(frame, nw.DataFrame):
            raise TypeError('This explanation result is not a dataframe')
        namespace = nw.get_native_namespace(frame)
        if backend == 'native':
            namespace = nw.get_native_namespace(nw.from_native(original, eager_only=True))
        elif namespace.__name__ != backend:
            namespace = backend
        if namespace == nw.get_native_namespace(frame):
            return frame.clone().to_native()
        return nw.from_dict(frame.to_dict(as_series=False), backend=namespace).to_native()

    @abstractmethod
    def fit(self):
        pass
    @abstractmethod
    def plot(self):
        pass
