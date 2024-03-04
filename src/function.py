# function.py
import numpy as np

from .value import Value


class _SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class _FunctionFactory(metaclass=_SingletonMeta):
    _active_instances = {}
    _free_instances = {}

    def get_new_function_of_type(self, f_type):
        if len(self._free_instances) != 0:
            name, func = self._free_instances[f_type].popitem()
        else:
            name, func = self._create_function_of_type(f_type)

        if f_type not in self._active_instances:
            self._active_instances[f_type] = {}

        self._active_instances[f_type][name] = func

        return func

    def get_active_function_by_name(self, name):
        func = None
        for t in self._active_instances.keys():
            if name not in self._active_instances[t]:
                continue

            func = self._active_instances[t][name]

        if func is None:
            print(f"Active Function not Found: {name}")

        return func

    def _create_function_of_type(self, f_type):
        num = 0
        if f_type in self._active_instances.keys():
            num += len(self._active_instances[f_type])
        if f_type in self._free_instances.keys():
            num += len(self._free_instances[f_type])

        print(f"Creating Function of type {f_type} with name", end=" ")
        if f_type == _Add:
            name = f"add_{num}"
            print(name)
            return name, _Add(name)
        elif f_type == _Mul:
            name = f"mul_{num}"
            print(name)
            return name, _Mul(name)
        elif f_type == _Matmul:
            name = f"matmul_{num}"
            print(name)
            return name, _Matmul(name)
        elif f_type == _Exp:
            name = f"exp_{num}"
            print(name)
            return name, _Exp(name)
        elif f_type == _Sigmoid:
            name = f"sigmoid_{num}"
            print(name)
            return name, _Sigmoid(name)
        elif f_type == _Linear:
            name = f"linear_{num}"
            print(name)
            return name, _Linear(name)


class _Function(object):
    def __init__(self, name):
        self.name = name
        self.grad_info = {}
        self.out = None

    def forward(self, *args):
        pass

    def backward(self, ):
        for node in self.out.nodes.values():
            if node.function_id is not None:
                func = _FunctionFactory().get_active_function_by_name(node.function_id)
                func.update_grad()

    def update_grad(self, ):
        pass


class _Add(_Function):
    def __init__(self, name):
        super(_Add, self).__init__(name)

    def forward(self, *args):
        s = (args[0].value + args[1].value).squeeze()
        self.out = Value(s, f"({args[0].name}+{args[1].name})", function_id=self.name)

        key1 = self.out.attach_node(args[0])
        key2 = self.out.attach_node(args[1])

        if self.out.requires_grad:
            self.grad_info[key1] = args[0].shape
            self.grad_info[key2] = args[1].shape

        return self.out

    def backward(self, ):
        self.out.add_grad(np.ones(self.out.shape))
        self.update_grad()
        super().backward()

    # This should be called from a backward of a higher function
    def update_grad(self, ):
        for name, node in self.out.nodes.items():
            if not node.requires_grad:
                continue

            dx = self.out.grad * np.ones(self.grad_info[name])
            if dx.ndim != len(self.grad_info[name]):  # Broadcasting case
                dx = np.mean(dx, axis=0)
            node.add_grad(dx)


class _Mul(_Function):
    def __init__(self, name):
        super(_Mul, self).__init__(name)

    def forward(self, *args):
        m = (args[0].value * args[1].value).squeeze()
        self.out = Value(m, f"{args[0].name}*{args[1].name}", function_id=self.name)

        key1 = self.out.attach_node(args[0])
        key2 = self.out.attach_node(args[1])

        if self.out.requires_grad:
            self.grad_info[key1] = args[1].value
            self.grad_info[key2] = args[0].value

        return self.out

    def backward(self, ):
        self.out.add_grad(np.ones(self.out.shape))
        self.update_grad()
        super().backward()

    # This should be called from a backward of a higher function
    def update_grad(self, ):
        [(name1, node1), (name2, node2)] = self.out.nodes.items()

        dnode1 = self.grad_info[name1]
        dnode2 = self.grad_info[name2]

        dnode1 *= self.out.grad
        if dnode1.ndim > self.out.ndim:
            dnode1 = np.mean(dnode1, axis=0)
        dnode2 *= self.out.grad
        if dnode2.ndim > self.out.ndim:
            dnode2 = np.mean(dnode2, axis=0)

        node1.add_grad(dnode1)
        node2.add_grad(dnode2)


class _Matmul(_Function):
    def __init__(self, name):
        super(_Matmul, self).__init__(name)

    def forward(self, *args):
        m = np.matmul(args[0].value, args[1].value).squeeze()
        self.out = Value(m, f"{args[0].name}@{args[1].name}", function_id=self.name)

        key1 = self.out.attach_node(args[0])
        key2 = self.out.attach_node(args[1])

        if self.out.requires_grad:
            self.grad_info[key1] = args[1].value
            self.grad_info[key2] = args[0].value

        return self.out

    def backward(self, ):
        if self.out.ndim > 1:
            return

        self.out.add_grad(np.ones(self.out.shape))
        self.update_grad()
        super().backward()

    # This should be called from a backward of a higher function
    def update_grad(self, ):
        [(name1, node1), (name2, node2)] = self.out.nodes.items()

        dnode1 = self.grad_info[name1]
        dnode2 = self.grad_info[name2]

        dnode1 = self.out.grad @ dnode1.T
        dnode2 = dnode2.T @ self.out.grad

        node1.add_grad(dnode1)
        node2.add_grad(dnode2)


class _Exp(_Function):
    def __init__(self, name):
        super(_Exp, self).__init__(name)

    def forward(self, *args):
        e = np.exp(args[0].value).squeeze()
        self.out = Value(e, f"exp({args[0].name})", function_id=self.name)

        key = self.out.attach_node(args[0])

        if self.out.requires_grad:
            self.grad_info[key] = e

        return self.out

    def backward(self, ):
        self.out.add_grad(np.ones(self.out.shape))
        self.update_grad()
        super().backward()

    # This should be called from a backward of a higher function
    def update_grad(self, ):
        [(name, node)] = self.out.nodes.items()

        dx = self.grad_info[name]
        dx *= self.out.grad

        node.add_grad(dx)


class _Sigmoid(_Function):
    def __init__(self, name):
        super(_Sigmoid, self).__init__(name)

    def forward(self, *args):
        s = 1 / (1 + np.exp(-args[0].value)).squeeze()
        self.out = Value(s, f"sigma({args[0].name})", function_id=self.name)

        key = self.out.attach_node(args[0])

        if self.out.requires_grad:
            self.grad_info[key] = s

        return self.out

    def backward(self, ):
        self.out.add_grad(np.ones(self.out.shape))
        self.update_grad()
        super().backward()

    def update_grad(self, ):
        [(name, node)] = self.out.nodes.items()

        dx = self.grad_info[name] * (1 - self.grad_info[name])
        dx *= self.out.grad

        node.add_grad(dx)


class _Linear(_Function):
    def __init__(self, name):
        super(_Linear, self).__init__(name)

    def forward(self, *args):
        lin = (args[0].value @ args[1].value + args[2].value).squeeze()
        self.out = Value(lin, f"({args[0].name}@{args[1].name}+{args[2].name})", function_id=self.name)

        key1 = self.out.attach_node(args[0])
        key2 = self.out.attach_node(args[1])
        key3 = self.out.attach_node(args[2])

        if self.out.requires_grad:
            self.grad_info[key1] = args[1].value
            self.grad_info[key2] = args[0].value
            self.grad_info[key3] = args[2].shape

        return self.out

    def backward(self, ):
        if self.out.ndim > 1:
            return

        self.out.add_grad(np.ones(self.out.shape))
        self.update_grad()
        super().backward()

    # This should be called from a backward of a higher function
    def update_grad(self, ):
        [(name1, node1), (name2, node2), (name3, node3)] = self.out.nodes.items()

        dnode1 = self.grad_info[name1]
        dnode2 = self.grad_info[name2]

        dnode1 = self.out.grad @ dnode1.T
        dnode2 = dnode2.T @ self.out.grad
        dnode3 = self.out.grad
        if dnode3.ndim != len(self.grad_info[name3]):  # Broadcasting case
            dnode3 = np.mean(dnode3, axis=0)

        node1.add_grad(dnode1)
        node2.add_grad(dnode2)
        node3.add_grad(dnode3)


def add(a, b, return_func=False):
    func = _FunctionFactory().get_new_function_of_type(_Add)
    if func is None:
        print(f"Failed to get function of type _Add")
        return

    c = func.forward(a, b)
    if return_func:
        return c, func

    return c


def mul(a, b, return_func=False):
    func = _FunctionFactory().get_new_function_of_type(_Mul)
    if func is None:
        print(f"Failed to get function of type _Mul")
        return

    c = func.forward(a, b)
    if return_func:
        return c, func

    return c


def matmul(A, B, return_func=False):
    func = _FunctionFactory().get_new_function_of_type(_Matmul)
    if func is None:
        print(f"Failed to get function of type _Matmul")
        return

    c = func.forward(A, B)
    if return_func:
        return c, func

    return c


def exp(x, return_func=False):
    func = _FunctionFactory().get_new_function_of_type(_Exp)
    if func is None:
        print(f"Failed to get function of type _Exp")
        return

    y = func.forward(x)
    if return_func:
        return y, func

    return y


def sigmoid(x, return_func=False):
    func = _FunctionFactory().get_new_function_of_type(_Sigmoid)
    if func is None:
        print(f"Failed to get function of type _Sigmoid")
        return

    y = func.forward(x)
    if return_func:
        return y, func

    return y


def linear(X, W, b, return_func=False):
    func = _FunctionFactory().get_new_function_of_type(_Linear)
    if func is None:
        print(f"Failed to get function of type _Linear")
        return

    y = func.forward(X, W, b)
    if return_func:
        return y, func

    return y
