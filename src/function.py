# function.py
import numpy as np
from value import Value
from node import GraphNode


class Function(object):
    def __init__(self, name):
        self.name = name
        self.grad_info = {}
        self.out = None

    def forward(self, *args):
        pass

    def backward(self,):
        topo = self.build_topo()

        for node in reversed(topo):
            if node.requires_grad:
                node.update_grad()


class Add(Function):
    def __init__(self, name):
        super(Add, self).__init__(name)

    def forward(self, *args):
        s = args[0].value + args[1].value
        self.out = Value(s, f"{args[0].name}+{args[1].name}")

        key1 = self.out.attach_node(args[0])
        key2 = self.out.attach_node(args[1])

        if self.out.requires_grad:
            self.grad_info[key1] = args[0].shape
            self.grad_info[key2] = args[1].shape

        return self.out

    def update_grad(self, dout):
        for name, node in self.out.nodes.items():
            if node.requires_grad is False:
                continue
            dx = np.full(self.grad_info[name], self.out.grad)
            node.update_grad(dx)

    def backward(self,):
        self.out.update_grad(np.ones(self.out.shape))
        super().backward()
