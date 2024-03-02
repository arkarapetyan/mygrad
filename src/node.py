# node.py


class GraphNode(object):
    def __init__(self, name, requires_grad=False):
        self.nodes = {}
        self.name = name

        self.requires_grad = requires_grad
        self.grad = None
        self.requires_grad_(requires_grad)

    def attach_node(self, node):
        key = node.name
        if key is None:
            key = f"node{len(self.nodes)}"
        self.nodes[key] = node

        if node.requires_grad:
            self.requires_grad_(True)

        return key

    def detach_node_by_name(self, name):
        if name not in self.nodes.keys():
            return
        self.nodes.pop(name)

    def detach_node(self, node):
        self.detach_node_by_name(node.name)

    def build_topo(self):
        topo = []
        visited = set()

        def topology(x):
            if x not in visited:
                visited.add(x)
                for child in x.nodes.values():
                    topology(child)
                topo.append(x)

        topology(self)

        return topo

    def requires_grad_(self, val):
        if val:
            self.requires_grad = True
            self.grad = 0.0
        else:
            self.requires_grad = False

    def update_grad(self, dx):
        if self.requires_grad is False:
            return

        self.grad = dx
