# value.py
import numpy as np

from node import GraphNode


class Value(GraphNode):
    def __init__(self, value: np.ndarray, name, requires_grad=False,):
        super(Value, self).__init__(name, requires_grad)
        self.value = value
        self.dtype = value.dtype
        self.shape = value.shape
        self.ndim = value.ndim
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"Value(data={self.value}, shape={self.shape}, requires_grad={self.requires_grad}), dtype={self.dtype}"
