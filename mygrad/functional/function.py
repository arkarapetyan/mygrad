# functional/function.py
import numpy as np

import mygrad.functional.function_factory as factory
from mygrad.value import Value


class Function(object):
    def __init__(self, name, forward_func, backward_func, n_var, *args):
        self.name = name
        self.forward_func = forward_func
        self.backward_func = backward_func
        self.n_var = n_var
        self.forward_cache = {}   # This should be refactored
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
            self.grad_info = {}
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
        s = (a.data + b.data).squeeze()

        return Value(s, f"({a.name}+{b.name})", function_id=self.name)

    def __add_backward(self,):
        grads = {}

        for name, shape in self.grad_info.items():
            dx = self.out.grad * np.ones(shape)
            if dx.ndim != len(shape):   # Broadcasting case
                dx = np.mean(dx, axis=0)
            grads[name] = dx

        return grads

    def _save_grad_info(self, args):
        if args[0].requires_grad:
            self.grad_info[args[0].name] = args[0].shape
        if args[1].requires_grad:
            self.grad_info[args[1].name] = args[1].shape


class Mul(Function):
    def __init__(self, name):
        super(Mul, self).__init__(name, self.__mul_forward, self.__mul_backward, 2)

    def __mul_forward(self, a, b):
        m = (a.data * b.data).squeeze()

        return Value(m, f"{a.name}*{b.name}", function_id=self.name)

    def __mul_backward(self, ):
        grads = {}

        for name, (shape, dx) in self.grad_info.items():
            dx *= self.out.grad
            if dx.ndim != len(shape):   # Broadcasting case
                dx = np.mean(dx, axis=0)
            grads[name] = dx

        return grads

    def _save_grad_info(self, args):
        if args[0].requires_grad:
            self.grad_info[args[0].name] = (args[0].shape, args[1].data)
        if args[1].requires_grad:
            self.grad_info[args[1].name] = (args[1].shape, args[0].data)


class Matmul(Function):
    def __init__(self, name):
        super(Matmul, self).__init__(name, self.__matmul_forward, self.__matmul_backward, 2)

    def __matmul_forward(self, A, B):
        A_val = A.data.reshape(A.shape[0], -1)
        B_val = B.data.reshape(B.shape[0], -1)
        M = np.matmul(A_val, B_val).squeeze()

        return Value(M, f"{A.name}@{B.name}", function_id=self.name)

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
        if args[0].requires_grad:
            self.grad_info[args[0].name] = ("L", args[1].data)
        if args[1].requires_grad:
            self.grad_info[args[1].name] = ("R", args[0].data)


class Exp(Function):
    def __init__(self, name):
        super(Exp, self).__init__(name, self.__exp_forward, self.__exp_backward, 1)

    def __exp_forward(self, x):
        e = np.exp(x.data).squeeze()
        self.forward_cache["exp"] = e

        return Value(e, f"exp({x.name})", function_id=self.name)

    def __exp_backward(self, ):
        grads = {}

        for name, dx in self.grad_info.items():
            dx *= self.out.grad
            grads[name] = dx

        return grads

    def _save_grad_info(self, args):
        if args[0].requires_grad:
            self.grad_info[args[0].name] = self.forward_cache["exp"]


class Sigmoid(Function):
    def __init__(self, name):
        super(Sigmoid, self).__init__(name, self.__sigmoid_forward, self.__sigmoid_backward, 1)

    def __sigmoid_forward(self, x):
        s = 1 / (1 + np.exp(-x.data)).squeeze()
        self.forward_cache["sigmoid"] = s

        return Value(s, f"sigmoid({x.name})", function_id=self.name)

    def __sigmoid_backward(self, ):
        grads = {}

        for name, dx in self.grad_info.items():
            dx *= self.out.grad
            grads[name] = dx

        return grads

    def _save_grad_info(self, args):
        if args[0].requires_grad:
            self.grad_info[args[0].name] = self.forward_cache["sigmoid"]


class Linear(Function):
    def __init__(self, name):
        super(Linear, self).__init__(name, self.__linear_forward, self.__linear_backward, 3)

    def __linear_forward(self, X, W, b):
        X_val = X.data.reshape(X.shape[0], -1)
        W_val = W.data.reshape(W.shape[0], -1)
        b_val = b.data
        lin = ((X_val @ W_val).squeeze() + b_val).squeeze()

        return Value(lin, f"({X.name}@{W.name}+{b.name})", function_id=self.name)

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
            self.grad_info[args[0].name] = ("X", args[1].data, args[0].shape)
        if args[1].requires_grad:
            self.grad_info[args[1].name] = ("W", args[0].data, args[1].shape)
        if args[2].requires_grad:
            self.grad_info[args[2].name] = ("b", args[2].data, args[2].shape)


class BCELossWithLogits(Function):
    def __init__(self, name):
        super(BCELossWithLogits, self).__init__(name, self.__bce_forward, self.__bce_backward, 2)

    def __bce_forward(self, y, y_true):
        s = 1 / (1 + np.exp(-y.data)).squeeze()
        loss = -np.mean(y_true.data * np.log(s) + (1 - y_true.data) * np.log(1 - s)).squeeze()
        self.forward_cache["sigmoid"] = s
        self.forward_cache["y_true"] = y_true.data

        return Value(loss, f"BCELossWithLogits({y.name})", function_id=self.name)

    def __bce_backward(self, ):
        grads = {}

        for name, (s, y_true) in self.grad_info.items():
            dx = (s - y_true) * self.out.grad
            grads[name] = dx

        return grads

    def _save_grad_info(self, args):
        if args[0].requires_grad:
            self.grad_info[args[0].name] = (self.forward_cache["sigmoid"], self.forward_cache["y_true"])


class MSELoss(Function):
    def __init__(self, name):
        super(MSELoss, self).__init__(name, self.__mse_forward, self.__mse_backward, 2)

    def __mse_forward(self, y, y_true):
        loss = np.mean((y_true.data - y.data) ** 2).squeeze()

        return Value(loss, f"MSELoss({y.name})", function_id=self.name)

    def __mse_backward(self, ):
        grads = {}

        for name, (y, y_true) in self.grad_info.items():
            dx = 2 * (y - y_true) * self.out.grad
            grads[name] = dx

        return grads

    def _save_grad_info(self, args):
        if args[0].requires_grad:
            self.grad_info[args[0].name] = (args[0].data, args[1].data)
