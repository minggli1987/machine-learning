import functools


def fibonacci1(n):
    """recursively sum up fibonacci numbers"""
    assert n == int(n) and n > 0
    if n in [1, 2]:
        return 1
    return fibonacci1(n-1) + fibonacci1(n-2)


def fibonacci_wrapper(func):
    @functools.wraps(func)
    def wrapper(*args):
        n = args[0]
        g = func(n)
        sequence = [next(g) for _ in range(n)]
        return sequence.pop()
    return wrapper


@fibonacci_wrapper
def fibonacci2(max):
    a, b = 1, 1
    while True:
        yield a
        a, b = b, a + b


x = 0b1001111
print(fibonacci1(x))
print(fibonacci2(x))
