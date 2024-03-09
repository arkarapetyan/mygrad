# functional/__init__.py
import mygrad as F
from mygrad import FunctionFactory


def add():
    func = FunctionFactory().get_new_function_of_type(F.Add)
    if func is None:
        print(f"Failed to get function of type _Add")
        return

    return func


def mul():
    func = FunctionFactory().get_new_function_of_type(F.Mul)
    if func is None:
        print(f"Failed to get function of type _Mul")
        return

    return func


def matmul():
    func = FunctionFactory().get_new_function_of_type(F.Matmul)
    if func is None:
        print(f"Failed to get function of type _Matmul")
        return

    return func


def exp():
    func = FunctionFactory().get_new_function_of_type(F.Exp)
    if func is None:
        print(f"Failed to get function of type _Exp")
        return

    return func


def sigmoid():
    func = FunctionFactory().get_new_function_of_type(F.Sigmoid)
    if func is None:
        print(f"Failed to get function of type _Sigmoid")
        return

    return func


def linear():
    func = FunctionFactory().get_new_function_of_type(F.Linear)
    if func is None:
        print(f"Failed to get function of type _Linear")
        return

    return func


def bce_loss_with_logits():
    loss = FunctionFactory().get_new_function_of_type(F.BCELossWithLogits)
    if loss is None:
        print(f"Failed to get BCE loss")
        return

    return loss


def mse_loss():
    loss = FunctionFactory().get_new_function_of_type(F.MSELoss)
    if loss is None:
        print(f"Failed to get MSE loss")
        return

    return loss
