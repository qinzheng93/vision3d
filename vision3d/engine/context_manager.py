import os.path as osp
import warnings

import torch
from torch import Tensor


class ContextManager:
    def __init__(self):
        self._data_dict = {}
        self._scope = []

    def register(self, name, value, retain_grad=True):
        name = osp.join(*self._scope, name)
        if isinstance(value, Tensor) and retain_grad:
            if value.requires_grad:
                value.retain_grad()
        self._data_dict[name] = value

    def get(self, name=None, full_name=None):
        assert name is not None or full_name is not None
        if full_name is None:
            full_name = osp.join(*self._scope, name)
        if full_name not in self._data_dict:
            warnings.warn(f"Key '{full_name}' not found in ContextManager.")
            return None
        return self._data_dict[full_name]

    def clear(self):
        self._data_dict.clear()
        self._scope.clear()

    def enter_scope(self, scope_name):
        self._scope.append(scope_name)
        return self

    def exit_scope(self):
        if self._scope:
            self._scope.pop()

    @property
    def scope(self):
        if not self._scope:
            return None
        return osp.join(*self._scope)

    @property
    def data_dict(self):
        return self._data_dict

    @property
    def data_keys(self):
        return [x for x in self._data_dict.keys()]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit_scope()

    def __repr__(self):
        format_string = "ContextManager(\n"
        format_string += f"  scope='{self.scope}',\n"
        format_string += f"  names={self.data_keys}\n"
        format_string += ")"
        return format_string


_CONTEXT_MANAGER = ContextManager()


def get_context_manager():
    return _CONTEXT_MANAGER


def clear_context_manager():
    _CONTEXT_MANAGER.clear()
