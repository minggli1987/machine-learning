import ast

class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
    def __add__(self, other):
        # no overlap
        if other.start > self.end or self.start > other.end:
            return [self, other]
        # overlap and other ends later
        if other.start <= self.end and other.end >= self.end:
            return self.__class__(self.start, other.end)
        
        # overlap and self ends later
        if self.start <= other.end and self.end >= other.end:
            return self.__class__(other.start, self.end)
    def __repr__(self):
        return f"[{self.start}, {self.end}]"
    def eval(self):
        return ast.literal_eval(repr(self))


intervals = [Interval(1, 2), Interval(4, 7), Interval(9, 10)]
ins = Interval(2, 5)
intervals.append(ins)
intervals.sort(key=lambda x: x.start)

while True:
    n = len(intervals)
    for i in range(n):
        try:
            s = intervals[i] + intervals[i + 1]
            if not hasattr(s, "__iter__"):
                intervals[i + 1] = s
                intervals.pop(i)
        except IndexError:
            pass
    if len(intervals) == n:
        break

print(intervals)
