# functional/__init__.py
from mygrad.functional.function_factory import FunctionFactory
from mygrad.functional.function import *


def add():
    func = FunctionFactory().get_new_function_of_type(Add)
    if func is None:
        print(f"Failed to get function of type _Add")
        return

    return func


def mul():
    func = FunctionFactory().get_new_function_of_type(Mul)
    if func is None:
        print(f"Failed to get function of type _Mul")
        return

    return func


def matmul():
    func = FunctionFactory().get_new_function_of_type(Matmul)
    if func is None:
        print(f"Failed to get function of type _Matmul")
        return

    return func


def exp():
    func = FunctionFactory().get_new_function_of_type(Exp)
    if func is None:
        print(f"Failed to get function of type _Exp")
        return

    return func


def sigmoid():
    func = FunctionFactory().get_new_function_of_type(Sigmoid)
    if func is None:
        print(f"Failed to get function of type _Sigmoid")
        return

    return func


def tanh():
    func = FunctionFactory().get_new_function_of_type(Tanh)
    if func is None:
        print(f"Failed to get function of type _Sigmoid")
        return

    return func


def relu():
    func = FunctionFactory().get_new_function_of_type(ReLU)
    if func is None:
        print(f"Failed to get function of type _Sigmoid")
        return

    return func


def linear():
    func = FunctionFactory().get_new_function_of_type(Linear)
    if func is None:
        print(f"Failed to get function of type _Linear")
        return

    return func


def bce_loss_with_logits():
    loss = FunctionFactory().get_new_function_of_type(BCELossWithLogits)
    if loss is None:
        print(f"Failed to get BCE loss")
        return

    return loss


def mse_loss():
    loss = FunctionFactory().get_new_function_of_type(MSELoss)
    if loss is None:
        print(f"Failed to get MSE loss")
        return

    return loss
