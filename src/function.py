# function.py
import numpy as np

from value import Value


class Function(object):
    def __init__(self, name):
        self.name = name
        self.grad_info = {}
        self.out = None

    def forward(self, *args):
        pass

    def backward(self, ):
        pass

    def _update_grad(self, ):
        pass


class Add(Function):
    def __init__(self, name):
        super(Add, self).__init__(name)

    def forward(self, *args):
        s = args[0].value + args[1].value
        self.out = Value(s, f"({args[0].name}+{args[1].name})")

        key1 = self.out.attach_node(args[0])
        key2 = self.out.attach_node(args[1])

        if self.out.requires_grad:
            self.grad_info[key1] = args[0].shape
            self.grad_info[key2] = args[1].shape

        return self.out

    def backward(self,):
        self.out.add_grad(np.ones(self.out.shape))
        for name, node in self.out.nodes.items():
            if node.requries_grad:
                self._compute_node_grad(node)
            if node.function_id is not None:
                # Hanel funckian, kanchel ira backwardy

    # This should be called from a backward of a higher function
    def _update_grad(self,):
        for name, node in self.out.nodes.items():
            if node.requires_grad:
                self._compute_node_grad(node)

    def _compute_node_grad(self, node):
        # TODO
        dx = self.out.grad
        if dx.ndim != node.ndim:  # Broadcasting case
            dx = np.mean(dx, axis=0)
        node.add_grad(dx)


class Matmul(Function):
    def __init__(self, name):
        super(Add, self).__init__(name)

    def forward(self, *args):
        m = np.matmul(args[0].value, args[1].value)
        if m.shape[-1] == 1:
            m = m.squeeze()

        self.out = Value(m, f"{args[0].name}@{args[1].name}")

        key1 = self.out.attach_node(args[0])
        key2 = self.out.attach_node(args[1])

        if self.out.requires_grad:
            self.grad_info[key1] = args[0].value
            self.grad_info[key2] = args[1].value

        return self.out

    def backward(self,):
        if self.out.ndim > 1:
            return

        self.out.add_grad(np.ones(self.out.shape))
        # TODO

    # This should be called from a backward of a higher function
    def _update_grad(self,):
        (name1, node1), (name2, node2) = self.out.nodes.items()

        dnode1 = self.grad_info[name1]
        dnode2 = self.grad_info[name2]

        dnode1 = self.out.grad @ dnode1.T
        dnode2 = self.out.grad @ dnode2.T

        node1.add_grad(dnode1)
        node2.add_grad(dnode2)

