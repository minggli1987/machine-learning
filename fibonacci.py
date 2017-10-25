import time
from functools import wraps, lru_cache as cache


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        rv = func(*args, **kwargs)
        end = time.time()
        print("function {0} took {1:.4f} ms.".format(func.__name__, 1e3 * (end - start)))
        return rv
    return wrapper


def generator_wrapper(func):
    @wraps(func)
    def wrapper(*args):
        n,  = args
        g = func(n)
        sequence = [next(g) for _ in range(n)]
        return sequence.pop()
    return wrapper


def fibonacci0(n):
    """recursively sum up fibonacci numbers"""
    assert n == int(n) and n > 0
    if n in [1, 2]:
        return 1
    return fibonacci0(n-1) + fibonacci0(n-2)


@timeit
def fibonacci1(n):
    return fibonacci0(n)


@timeit
def fibonacci2(n):
    @cache(maxsize=128, typed=False)
    def fibonacci0(n):
        """recursively sum up fibonacci numbers"""
        assert n == int(n) and n > 0
        if n in [1, 2]:
            return 1
        return fibonacci0(n-1) + fibonacci0(n-2)
    return fibonacci0(n)


@timeit
@generator_wrapper
def fibonacci3(max):
    a, b = 1, 1
    while True:
        yield a
        a, b = b, a + b


@timeit
@cache(maxsize=128, typed=False)
@generator_wrapper
def fibonacci4(max):
    a, b = 1, 1
    while True:
        yield a
        a, b = b, a + b


x = 35
print(fibonacci1(x))
print(fibonacci2(x))
print(fibonacci3(x))
print(fibonacci4(x))
