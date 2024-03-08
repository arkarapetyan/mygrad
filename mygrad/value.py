# value/value.py
import numpy as np

from mygrad.node import GraphNode


class Value(GraphNode):
    def __init__(self, value, name, requires_grad=False, function_id=None):
        super(Value, self).__init__(name, requires_grad, function_id)
        self.value = np.array(value).squeeze()
        self.dtype = self.value.dtype
        self.shape = self.value.shape
        self.ndim = self.value.ndim
        self.requires_grad = requires_grad

    def __repr__(self):
        return (f"Value(data={self.value}, shape={self.shape}, name={self.name}, requires_grad={self.requires_grad}), "
                f"dtype={self.dtype}")
