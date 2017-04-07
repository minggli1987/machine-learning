

class Node(object):

    instances = list()

    def __init__(self, value):
        self.value = value
        self.edges = []
        self.is_origin = None
        self.__class__.instances.append(self)

    def is_connected(self, other):

        self.is_origin = True

        if other in self.edges:
            self.is_origin = None
            return True
        else:
            for edge in self.edges:
                if edge.is_origin:
                    continue
                if edge.is_connected(other):
                    self.is_origin = None
                    return True
            return False

    def connect(self, other):
        if other not in self.edges:
            self.edges.insert(0, other)

    def __repr__(self):
        return 'Node({0})'.format(self.value)

    def __eq__(self, other):
        pass

    def __gt__(self, other):
        pass

    def __ge__(self, other):
        pass

    @classmethod
    def show(cls):
        return [i for i in cls.instances]


x = Node(3)
y = Node(7)
z = Node(11)
a = Node(14)
b = Node(21)
c = Node(23)
d = Node(27)
e = Node(31)
f = Node(38)
g = Node(42)
h = Node(45)

x.connect(y)
y.connect(z)
z.connect(a)
a.connect(b)
b.connect(c)
c.connect(d)
d.connect(e)
e.connect(f)
f.connect(g)
g.connect(h)
# h.connect(x)

print([i.value for i in x.edges])
print([i.value for i in y.edges])
print([i.value for i in z.edges])
print([i.value for i in a.edges])
print([i.value for i in b.edges])
print([i.value for i in c.edges])
print([i.value for i in d.edges])
print([i.value for i in e.edges])
print([i.value for i in f.edges])
print([i.value for i in g.edges])
print([i.value for i in h.edges])


print(x.is_connected(y))
print(x.is_connected(h))

print(h.show())
print(c.show())
