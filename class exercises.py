__author__ = 'Ming'


class NumberFormatted(object):

    def __init__(self, n):

        assert isinstance(n, (float, int)), 'please input a numeric value'
        self.value = float('{0:.2f}'.format(n))

assert isinstance(32.123, (float, int)), 'Please input a numeric value'

print(NumberFormatted(234.123).value)


class Nodes(object):

    def __init__(self, n):
        self.edges = []
        self.value = n

    def is_connected(self, other):
        return
