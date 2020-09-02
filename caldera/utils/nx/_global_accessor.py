"""Adds the following to nx.Graph.

`get_global()`
"""
import functools

import networkx as nx


_global_key = "_global_data"


class GlobalDict(dict):
    def __init__(self, data, obj):
        super().__init__(data)
        self._obj = obj
        self._obj_set = False

    def update(self, d):
        if not self._obj_set:
            setattr(self._obj, _global_key, self)
        super().update(d)

    def setdefault(self, k, v):
        if not self._obj_set:
            setattr(self._obj, _global_key, self)
        super().__setitem__(k, v)

    def __setitem__(self, k, v):
        if not self._obj_set:
            setattr(self._obj, _global_key, self)
        super().__setitem__(k, v)


class GlobalAccess:
    def __get__(self, obj, objtype):
        if not hasattr(obj, _global_key):
            return GlobalDict({}, obj)
        return getattr(obj, _global_key)

    def __set__(self, obj, val):
        setattr(obj, _global_key, val)


def get_global(self, global_key: str = None, default_key: str = None):
    """Get.

    :param self:
    :param global_key:
    :param default_key:
    :return:
    """
    if global_key is None:
        global_key = default_key
    return getattr(self, global_key)


class GraphWithGlobal(nx.Graph):
    def get_global_key(self, global_key: str = None, default_key: str = None):
        """Retrieve the global_key for the graph. If `global_key` is provided,
        just return `global_key`. Else, use the provided default.

        Default key is set on `caldera` import (see `caldera._setup`) to be the value found
        in `caldera.defaults.CalderaDefaults.nx_global_key`.

        :param global_key:
        :param default_key:
        :return:
        """
        if global_key is None:
            global_key = default_key
        return global_key

    def get_global(self, global_key: str = None):
        """Get global attribute for the graph. If global_key not provided, will
        use the provided default_key.

        Default key is set on `caldera` import (see `caldera._setup`) to be the value found
        in `caldera.defaults.CalderaDefaults.nx_global_key`.

        :param global_key:
        :param default_key:
        :return:
        """
        gk = self.get_global_key(global_key)
        try:
            return getattr(self, gk)
        except AttributeError as e:
            msg = "'{}' object is missing attribute '{}' during call of `get_global`".format(
                self.__class__.__name__, gk
            )
            if global_key is None:
                msg += ". Global key was not explicitly provided."
            raise AttributeError(msg) from e

    def set_global(self, value: dict, global_key: str = None):
        """Set global attribute for the graph. If global_key not provided, will
        use the provided default_key.

        Default key is set on `caldera` import (see `caldera._setup`) to be the value found
        in `caldera.defaults.CalderaDefaults.nx_global_key`.

        :param global_key:
        :param default_key:
        :return:
        """
        if not isinstance(value, dict):
            raise TypeError("Value must be a `dict`")
        return setattr(self, self.get_global_key(global_key), value)
