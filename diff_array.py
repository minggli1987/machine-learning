"""
store information using difference array

so that for range update, there is linear complexity.
"""
from timeit import default_timer as timer
from functools import wraps


def time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = timer()
        output = func(*args, **kwargs)
        end = timer()
        print(f"function {func.__name__} took {round((end - start) * 1e9)} ns.")
        return output
    return wrapper


@time
def array_manipulation_quadratic(n, queries):
    normal_arr = [0] * n

    for (a, b, k) in queries:
        # O(M)
        a -= 1
        b -= 1
        normal_arr[a: b + 1] = map(lambda x: x + k, normal_arr[a: b + 1])

    return normal_arr, max(normal_arr)


@time
def array_manipulation_linear(n, queries):
    diff_arr = [0] * (n + 1)

    for (a, b, k) in queries:
        # O(M)
        a -= 1
        b -= 1
        diff_arr[a] += k
        diff_arr[b + 1] -= k

    normal_arr = [0] * n
    cumsum = 0
    for idx, l in enumerate(diff_arr):
        # O(N)
        cumsum += l
        try:
            normal_arr[idx] = cumsum
        except IndexError:
            pass

    return normal_arr, max(normal_arr)


if __name__ == "__main__":

    i = 5, [[1, 3, 3], [2, 3, 4]]

    arr_1, max_1 = array_manipulation_linear(*i)
    arr_2, max_2 = array_manipulation_quadratic(*i)
    assert arr_1 == arr_2
    assert max_1 == max_2


