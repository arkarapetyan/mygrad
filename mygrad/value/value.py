# value/value.py
import numpy as np

from mygrad.value.node import GraphNode


class Value(GraphNode):
    def __init__(self, data, name, requires_grad=False, function_id=None):
        super(Value, self).__init__(name, requires_grad, function_id)
        self.data = np.array(data).squeeze()
        self.dtype = self.data.dtype
        self.shape = self.data.shape
        self.ndim = self.data.ndim
        self.requires_grad = requires_grad

    def set_data(self, new_data):
        self.data = np.array(new_data).squeeze()

    def __repr__(self):
        return (f"Value(data={self.data}, shape={self.shape}, name={self.name}, requires_grad={self.requires_grad}), "
                f"dtype={self.dtype}")
