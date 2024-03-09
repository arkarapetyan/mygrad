# nn/module.py
import numpy as np

from mygrad import value
import mygrad.functional as F


class Module(object):
    def __init__(self, name):
        self.name = name

    def forward(self, X):
        pass

    def backward(self, ):
        pass

    def parameters(self, ):
        pass


class Linear(Module):
    def __init__(self, name, in_shape, out_shape, init="xavier"):
        super(Linear, self).__init__(name)

        self.matmul1 = F.matmul()
        self.add1 = F.add()

        self.W = value(np.zeros(in_shape, out_shape), f"{self.name}.W", requires_grad=True)
        self.b = value(np.zeros(out_shape), f"{self.name}.b", requires_grad=True)

    def forward(self, X):
        return self.add1.forward(self.matmul1.forward(X, self.W), self.b)

    def backward(self, ):
        return self.add1.backward()

    def parameters(self, ):
        return [self.W, self.b]
