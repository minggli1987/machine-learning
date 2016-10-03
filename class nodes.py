__author__ = 'Ming'


class Node(object):

    __origin__ = None
    __nextid = 1
    __inst = dict()

    def __init__(self, n):
        self.edges = []
        self.value = n
        self.inst_id = self.__class__.__nextid
        self.__class__.__inst[self.inst_id] = self
        self.__class__.__nextid += 1
        pass

    def is_connected(self, other):

        if self.__class__.__origin__ is None:
            self.__class__.__origin__ = self

        if other in self.edges:
            self.__class__.__origin__ = None
            return True
        else:
            for edge in self.edges:
                if edge == self.__class__.__origin__:
                    continue
                if edge.is_connected(other):
                    self.__class__.__origin__ = None
                    return True
        return False

    def connect(self, other):
        if other not in self.edges:
            self.edges.append(other)

    def __eq__(self, other):
        pass

    def __gt__(self, other):
        pass

    def __ge__(self, other):
        pass

    def __repr__(self):
        return str({k: 'Node({0})'.format(v.value) for k, v in self.__class__.__inst.items()})


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

print(h)

def fibonacci(n):
    assert n > 0 and isinstance(n, int), 'please input a valid integer'
    return 0 if n == 1 else 1 if n == 3 or n == 2 else fibonacci(n - 1) + fibonacci(n - 2)

seq = [fibonacci(i) for i in range(1, 20)]

print(seq)
