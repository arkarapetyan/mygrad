# nn/optim.py
import numpy as np


class Optimizer:
    def __init__(self, params, lr, param_optim_step):
        self.params = {param.name: param for param in params}
        self.lr = lr
        self.__param_optim_step = param_optim_step

    def step(self, ):
        for name, param in self.params.items():
            param.data -= self.lr * self.__param_optim_step(name)

    def zero_grad(self, ):
        for name, param in self.params.items():
            if param.requires_grad:
                param.zero_grad()


class SGD(Optimizer):
    def __init__(self, params, lr):
        super(SGD, self).__init__(params, lr, self.__param_sgd_step)

    def __param_sgd_step(self, param_name):
        return self.params[param_name].grad


class RMSProp(Optimizer):
    def __init__(self, params, lr=0.01, beta=0.99, eps=1e-08):
        super(RMSProp, self).__init__(params, lr, self.__param_rms_step)
        self.beta = beta
        self.eps = eps
        self.v = {}

    def __param_rms_step(self, param_name):
        if param_name not in self.v.keys():
            self.v[param_name] = np.zeros_like(self.params[param_name].grad)

        self.v[param_name] = self.beta * self.v[param_name] + (1 - self.beta) * (self.params[param_name].grad ** 2)
        step = self.params[param_name].grad / np.sqrt(self.v[param_name] + self.eps)

        return step


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.99, eps=1e-08):
        super(Adam, self).__init__(params, lr, self.__param_adam_step)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {}
        self.v = {}

    def __param_adam_step(self, param_name):
        if param_name not in self.m.keys():
            self.m[param_name] = np.zeros_like(self.params[param_name].grad)

        if param_name not in self.v.keys():
            self.v[param_name] = np.zeros_like(self.params[param_name].grad)

        self.m[param_name] = self.beta1 * self.m[param_name] + (1 - self.beta1) * self.params[param_name].grad
        self.v[param_name] = self.beta2 * self.v[param_name] + (1 - self.beta2) * (self.params[param_name].grad ** 2)

        m_hat = self.m[param_name] / (1 - self.beta1)
        v_hat = self.v[param_name] / (1 - self.beta2)

        step = m_hat / np.sqrt(v_hat + self.eps)

        return step
