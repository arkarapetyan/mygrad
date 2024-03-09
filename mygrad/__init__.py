# mygrad/__init__.py
from mygrad.value import Value


def value(v, name, requires_grad=False):
    return Value(v, name, requires_grad=requires_grad)
