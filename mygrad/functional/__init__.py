# functional/__init__.py

import function as F
from function_factory import FunctionFactory


def add(a, b, return_func=False):
    func = FunctionFactory().get_new_function_of_type(F.Add)
    if func is None:
        print(f"Failed to get function of type _Add")
        return

    c = func.forward(a, b)
    if return_func:
        return c, func

    return c


def mul(a, b, return_func=False):
    func = FunctionFactory().get_new_function_of_type(F.Mul)
    if func is None:
        print(f"Failed to get function of type _Mul")
        return

    c = func.forward(a, b)
    if return_func:
        return c, func

    return c


def matmul(A, B, return_func=False):
    func = FunctionFactory().get_new_function_of_type(F.Matmul)
    if func is None:
        print(f"Failed to get function of type _Matmul")
        return

    c = func.forward(A, B)
    if return_func:
        return c, func

    return c


def exp(x, return_func=False):
    func = FunctionFactory().get_new_function_of_type(F.Exp)
    if func is None:
        print(f"Failed to get function of type _Exp")
        return

    y = func.forward(x)
    if return_func:
        return y, func

    return y


def sigmoid(x, return_func=False):
    func = FunctionFactory().get_new_function_of_type(F.Sigmoid)
    if func is None:
        print(f"Failed to get function of type _Sigmoid")
        return

    y = func.forward(x)
    if return_func:
        return y, func

    return y


def linear(X, W, b, return_func=False):
    func = FunctionFactory().get_new_function_of_type(F.Linear)
    if func is None:
        print(f"Failed to get function of type _Linear")
        return

    y = func.forward(X, W, b)
    if return_func:
        return y, func

    return y


def bce_loss_with_logits(y, y_true):
    loss = FunctionFactory().get_new_function_of_type(F.BCELossWithLogits)
    if loss is None:
        print(f"Failed to get BCE loss")
        return

    y = loss.forward(y, y_true)
    return y, loss


def mse_loss(y, y_true):
    loss = FunctionFactory().get_new_function_of_type(F.MSELoss)
    if loss is None:
        print(f"Failed to get MSE loss")
        return

    y = loss.forward(y, y_true)
    return y, loss
