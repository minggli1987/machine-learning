__author__ = 'Ming'


class NumberFormatted(object):

    def __init__(self, n):

        assert isinstance(n, (float, int)), 'please input a numeric value'
        self.value = float('{0:.2f}'.format(n))

assert isinstance(32.123, (float, int)), 'please input a numeric value'

print(NumberFormatted(234.123).value)


class Node(object):

    def __init__(self, n):
        self.edges = []
        self.value = n

    def is_connected(self, other):

        if origin is None:
            origin = self

        if other in self.edges:
            return True
        else:
            for edge in self.edges:
                if edge == origin:
                    continue
                if edge.is_connected(other):
                    return True
        return False

    def connect(self, other):
        if other not in self.edges:
            self.edges.append(other)

    def __eq__(self, other):
        return self.value == other.value

x = Node(3)
y = Node(7)
z = Node(11)
a = Node(14)
b = Node(21)
c = Node(23)
d = Node(27)

x.connect(y)
y.connect(z)
z.connect(a)
a.connect(b)
b.connect(x)
x.connect(c)
c.connect(d)

print(origin)

print([i.value for i in x.edges])
print([i.value for i in y.edges])
print([i.value for i in z.edges])
print([i.value for i in a.edges])
print([i.value for i in b.edges])
print([i.value for i in c.edges])

print(x.is_connected(y))
print(x.is_connected(z))
print(x.is_connected(d))


def fibonacci(n):
    assert n > 0 and isinstance(n, int), 'please input a valid integer'
    return 0 if n == 1 else 1 if n == 3 or n == 2 else fibonacci(n - 1) + fibonacci(n - 2)

seq = [fibonacci(i) for i in range(1, 20)]

print(seq)
