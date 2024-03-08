# function.py
import numpy as np

import function_factory as factory
from .value import Value


class Function(object):
    def __init__(self, name, forward_func, backward_func, n_var):
        self.name = name
        self.forward_func = forward_func
        self.backward_func = backward_func
        self.n_var = n_var
        self.grad_info = None
        self.out = None

    def forward(self, *args):
        if len(args) != self.n_var:
            raise Exception(f"Invalid number of parameters in function {self.name}. Expected {self.n_var}" ""
                            ", got {len(args)}")

        self.out = self.forward_func(*args[:self.n_var])
        for i in range(self.n_var):
            self.out.attach_node(args[i])

        if self.out.requires_grad:
            self._save_grad_info(args[:self.n_var])

        return self.out

    def backward(self, ):
        if self.out is None:
            raise Exception(f"Function not Computed, call forward() first")

        self.out.add_grad(np.ones(self.out.shape))
        self.__update_grad()

    def __update_grad(self, ):
        if self.grad_info is None:
            raise Exception(f"Failed to get grad info, make sure that parameters require grad and call forward() first")
        grads = self.backward_func()

        for name, node in self.out.nodes.items():
            if not node.requires_grad:
                continue

            node.add_grad(grads[name])
            if node.function_id is None:
                continue
            func = factory.FunctionFactory().get_active_function_by_name(node.function_id)
            func.__update_grad()

    def _save_grad_info(self, args):
        pass


class Add(Function):
    def __init__(self, name):
        super(Add, self).__init__(name, self.__add_forward, self.__add_backward, 2)

    def __add_forward(self, a, b):
        s = (a.value + b.value).squeeze()
        self.out = Value(s, f"({a.name}+{b.name})", function_id=self.name)

        return self.out

    def __add_backward(self,):
        grads = {}

        for name, shape in self.grad_info.items():
            dx = self.out.grad * np.ones(shape)
            if dx.ndim != len(shape):   # Broadcasting case
                dx = np.mean(dx, axis=0)
            grads[name] = dx

        return grads

    def _save_grad_info(self, args):
        self.grad_info = {}

        if args[0].requires_grad:
            self.grad_info[args[0].name] = args[0].shape
        if args[1].requires_grad:
            self.grad_info[args[1].name] = args[1].shape


class Mul(Function):
    def __init__(self, name):
        super(Mul, self).__init__(name, self.__mul_forward, self.__mul_backward, 2)

    def __mul_forward(self, a, b):
        m = (a.value * b.value).squeeze()
        self.out = Value(m, f"{a.name}*{b.name}", function_id=self.name)

        return self.out

    def __mul_backward(self, ):
        grads = {}

        for name, (shape, dx) in self.grad_info.items():
            dx *= self.out.grad
            if dx.ndim != len(shape):   # Broadcasting case
                dx = np.mean(dx, axis=0)
            grads[name] = dx

        return grads

    def _save_grad_info(self, args):
        self.grad_info = {}

        if args[0].requires_grad:
            self.grad_info[args[0].name] = (args[0].shape, args[1].value)
        if args[1].requires_grad:
            self.grad_info[args[1].name] = (args[1].shape, args[0].value)


class Matmul(Function):
    def __init__(self, name):
        super(Matmul, self).__init__(name, self.__matmul_forward, self.__matmul_backward, 2)

    def __matmul_forward(self, A, B):
        A = A.reshape(A.shape[0], -1)
        B = B.reshape(B.shape[0], -1)
        M = np.matmul(A, B).squeeze()
        self.out = Value(M, f"{A.name}@{B.name}", function_id=self.name)

        return self.out

    def __matmul_backward(self, ):
        grads = {}

        dy = self.out.grad
        dy = dy.reshape(dy.shape[0], -1)

        for name, (mul_side, dx) in self.grad_info.items():
            if mul_side == "L":
                dx = (dy @ dx.T).squeeze()
            elif mul_side == "R":
                dx = (dx.T @ dy).squeeze()

            grads[name] = dx

        return grads

    def _save_grad_info(self, args):
        self.grad_info = {}

        if args[0].requires_grad:
            self.grad_info[args[0].name] = ("L", args[1].value)
        if args[1].requires_grad:
            self.grad_info[args[1].name] = ("R", args[0].value)


class Exp(Function):
    def __init__(self, name):
        super(Exp, self).__init__(name, self.__exp_forward, self.__exp_backward, 1)

    def __exp_forward(self, x):
        e = np.exp(x.value).squeeze()
        self.out = Value(e, f"exp({x.name})", function_id=self.name)

        return self.out

    def __exp_backward(self, ):
        grads = {}

        for name, dx in self.grad_info.items():
            dx *= self.out.grad
            grads[name] = dx

        return grads

    def _save_grad_info(self, args):
        self.grad_info = {}

        if args[0].requires_grad:
            self.grad_info[args[0].name] = self.out.value


class Sigmoid(Function):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name, self.__sigmoid_forward, self.__sigmoid_backward, 1)

    def __sigmoid_forward(self, x):
        s = 1 / (1 + np.exp(-x.value)).squeeze()
        self.out = Value(s, f"sigmoid({x.name})", function_id=self.name)

        return self.out

    def __sigmoid_backward(self, ):
        grads = {}

        for name, dx in self.grad_info.items():
            dx *= self.out.grad
            grads[name] = dx

        return grads

    def _save_grad_info(self, args):
        self.grad_info = {}

        if args[0].requires_grad:
            self.grad_info[args[0].name] = self.out.value


class Linear(Function):
    def __init__(self, name):
        super(Linear, self).__init__(name, self.__linear_forward, self.__linear_backward, 3)

    def __linear_forward(self, X, W, b):
        X_val = X.value.reshape(X.shape[0], -1)
        W_val = W.value.reshape(W.shape[0], -1)
        b_val = b.value
        lin = ((X_val @ W_val).squeeze() + b_val).squeeze()
        self.out = Value(lin, f"({X.name}@{W.name}+{b.name})", function_id=self.name)

        return self.out

    def __linear_backward(self, ):
        grads = {}

        dy = self.out.grad
        dy = dy.reshape(dy.shape[0], -1)

        for name, (p, dx, shape) in self.grad_info.items():
            if p == "X":
                dx = (dy @ dx.T).squeeze()
            elif p == "W":
                dx = (dx.T @ dy).squeeze()
            elif p == "b":
                dx = self.out.grad * np.ones(shape)
                if dx.ndim != len(shape):  # Broadcasting case
                    dx = np.mean(dx, axis=0)

            grads[name] = dx

        return grads

    def _save_grad_info(self, args):
        if args[0].requires_grad:
            self.grad_info[args[0].name] = ("X", args[1].value, args[0].shape)
        if args[1].requires_grad:
            self.grad_info[args[1].name] = ("W", args[0].value, args[1].shape)
        if args[2].requires_grad:
            self.grad_info[args[2].name] = ("b", args[2].value, args[2].shape)


class BCELossWithLogits(Function):
    def __init__(self, name):
        super(BCELossWithLogits, self).__init__(name, self.__bce_forward, self.__bce_backward, 2)
        self.__sigmoid = None

    def __bce_forward(self, y, y_true):
        s = 1 / (1 + np.exp(-y.value)).squeeze()
        loss = -np.mean(y_true.value * np.log(s) + (1 - y_true.value) * np.log(1 - s))
        self.out = Value(loss, f"BCELossWithLogits({y.name})", function_id=self.name)
        self.__sigmoid = s

        return self.out

    def __bce_backward(self, ):
        grads = {}

        for name, (s,  in self.grad_info.items():

        [(name1, node1), (name2, node2)] = self.out.nodes.items()

        if not node1.requires_grad:
            return

        s, y_true = self.grad_info[name1]
        dy = (s - y_true) * self.out.grad
        node1.add_grad(dy)

        super().__update_grad()

    def _save_grad_info(self, args):
        if args[0].requires_grad:
            self.grad_info[args[0]] = (self.out.value, self.__sigmoid)


class MSELoss(Function):
    def __init__(self, name):
        super(MSELoss, self).__init__(name)

    def forward(self, *args):
        y = args[0].value
        y_true = args[1].value
        loss = np.mean((y_true - y)**2)
        self.out = Value(loss, f"MSELoss({args[0].name})", function_id=self.name)

        key1 = self.out.attach_node(args[0])
        key2 = self.out.attach_node(args[1])

        if self.out.requires_grad:
            self.grad_info[key1] = y, y_true

        return self.out

    def backward(self, ):
        self.out.add_grad(np.ones(self.out.shape))
        self.__update_grad()

    def __update_grad(self, ):
        [(name1, node1), (name2, node2)] = self.out.nodes.items()

        if not node1.requires_grad:
            return

        y, y_true = self.grad_info[name1]
        dy = 2 * (y - y_true) * self.out.grad
        node1.add_grad(dy)

        super().__update_grad()


def add(a, b, return_func=False):
    func = factory.FunctionFactory().get_new_function_of_type(Add)
    if func is None:
        print(f"Failed to get function of type _Add")
        return

    c = func.forward(a, b)
    if return_func:
        return c, func

    return c


def mul(a, b, return_func=False):
    func = factory.FunctionFactory().get_new_function_of_type(Mul)
    if func is None:
        print(f"Failed to get function of type _Mul")
        return

    c = func.forward(a, b)
    if return_func:
        return c, func

    return c


def matmul(A, B, return_func=False):
    func = factory.FunctionFactory().get_new_function_of_type(Matmul)
    if func is None:
        print(f"Failed to get function of type _Matmul")
        return

    c = func.forward(A, B)
    if return_func:
        return c, func

    return c


def exp(x, return_func=False):
    func = factory.FunctionFactory().get_new_function_of_type(Exp)
    if func is None:
        print(f"Failed to get function of type _Exp")
        return

    y = func.forward(x)
    if return_func:
        return y, func

    return y


def sigmoid(x, return_func=False):
    func = factory.FunctionFactory().get_new_function_of_type(Sigmoid)
    if func is None:
        print(f"Failed to get function of type _Sigmoid")
        return

    y = func.forward(x)
    if return_func:
        return y, func

    return y


def linear(X, W, b, return_func=False):
    func = factory.FunctionFactory().get_new_function_of_type(Linear)
    if func is None:
        print(f"Failed to get function of type _Linear")
        return

    y = func.forward(X, W, b)
    if return_func:
        return y, func

    return y


def bce_loss_with_logits(y, y_true):
    loss = factory.FunctionFactory().get_new_function_of_type(BCELossWithLogits)
    if loss is None:
        print(f"Failed to get BCE loss")
        return

    y = loss.forward(y, y_true)
    return y, loss


def mse_loss(y, y_true):
    loss = factory.FunctionFactory().get_new_function_of_type(MSELoss)
    if loss is None:
        print(f"Failed to get MSE loss")
        return

    y = loss.forward(y, y_true)
    return y, loss
