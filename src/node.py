# node.py


class GraphNode(object):
    def __init__(self, name, requires_grad=False, function_id=None):
        self.nodes = {}
        self.name = name

        self.requires_grad = requires_grad
        self.grad = None
        self.active_grad = None
        self.requires_grad_(requires_grad)
        self.function_id = function_id

    def attach_node(self, node):
        if node.name in self.nodes.keys():
            return node.name

        self.nodes[node.name] = node
        if node.requires_grad and (not self.requires_grad):
            self.requires_grad_(True)

        return node.name

    def detach_node_by_name(self, name):
        if name not in self.nodes.keys():
            return
        self.nodes.pop(name)

    def detach_node(self, node):
        self.detach_node_by_name(node.name)

    def requires_grad_(self, val):
        if val:
            self.requires_grad = True
            self.grad = 0.0
            self.active_grad = False
        else:
            self.requires_grad = False

    def add_grad(self, dx):
        if self.requires_grad is False:
            return

        self.grad += dx
        self.active_grad = True

    def zero_grad(self):
        self.grad = 0.0
        self.active_grad = False

        for node in self.nodes.values():
            if node.requires_grad and node.active_grad:
                node.zero_grad()
