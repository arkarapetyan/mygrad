# value.py
import numpy as np

from .node import GraphNode


class Value(GraphNode):
    def __init__(self, value, name, requires_grad=False, function_id=None):
        super(Value, self).__init__(name, requires_grad, function_id)
        self.value = np.array(value)
        self.dtype = self.value.dtype
        self.shape = self.value.shape
        self.ndim = self.value.ndim
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Value(data={self.value}, shape={self.shape}, requires_grad={self.requires_grad}), dtype={self.dtype}"
