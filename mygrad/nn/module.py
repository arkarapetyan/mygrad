# nn/module.py
import numpy as np

from mygrad import value
import mygrad.functional as F
import mygrad.nn as nn


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
    def __init__(self, name, in_shape, out_shape, init="xavier_normal", **kwargs):
        super(Linear, self).__init__(name)

        self.matmul1 = F.matmul()
        self.add1 = F.add()

        self.W = value(np.zeros((in_shape, out_shape)), f"{self.name}.W", requires_grad=True)
        self.b = value(np.zeros(out_shape), f"{self.name}.b", requires_grad=True)

        if init is not None:
            activation = kwargs.get("activation", None)
            gain = kwargs.get("gain", None)
            param = kwargs.get("LeakyReLU_alpha", None)

            if init == "xavier_normal":
                nn.xavier_normal_initialization(self.W, gain=gain, activation=activation, param=param)
            elif init == "xavier_uniform":
                nn.xavier_uniform_initialization(self.W, gain=gain, activation=activation, param=param)
            elif init == "he_normal":
                nn.he_normal_initialization(self.W, gain=gain, activation=activation, param=param)
            elif init == "he_uniform":
                nn.he_uniform_initialization(self.W, gain=gain, activation=activation, param=param)

    def forward(self, X):
        return self.add1.forward(self.matmul1.forward(X, self.W), self.b)

    def backward(self, ):
        return self.add1.backward()

    def parameters(self, ):
        return [self.W, self.b]
