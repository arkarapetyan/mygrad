# value/__init__.py
from mygrad.value.value import *


def value(data, name, requires_grad=False):
    return Value(data, name, requires_grad=requires_grad)