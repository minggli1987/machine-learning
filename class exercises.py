__author__ = 'Ming'


class NumberFormatted(object):

    def __init__(self, n):

        assert isinstance(n, (float, int)), 'please input a numeric value'
        self.value = float('{0:.2f}'.format(n))

print(NumberFormatted(234.123).value + 1)


class Node(object):

    __origin__ = None
    __nextid = 0
    __inst = {}

    def __init__(self, n):
        self.edges = []
        self.value = n
        self.id = self.__class__.__nextid
        self.__class__.__inst[self.id] = self
        self.__class__.__nextid += 1

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
        return self.value == other.value

    def __gt__(self, other):
        return self.value > other.value

    def __ge__(self, other):
        return self.value >= other.value

    # def __del__(self):
    #     self.__class__.__inst.remove(self)
    #     print('Node({0})'.format(self.value), 'has been deleted.')

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
b.connect(x)
x.connect(c)
c.connect(d)
d.connect(e)
x.connect(f)
x.connect(g)
f.connect(h)

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

print(h.value)


# def fibonacci(n):
#     assert n > 0 and isinstance(n, int), 'please input a valid integer'
#     return 0 if n == 1 else 1 if n == 3 or n == 2 else fibonacci(n - 1) + fibonacci(n - 2)
#
# seq = [fibonacci(i) for i in range(1, 20)]
#
# print(seq)


class A(object):
    instances = []

    def __init__(self, foo):
        self.foo = foo
        A.instances.append(self)

i = A(1)
o = A(2)
print(A.instances)

