
def fibonacci(n):
    return 1 if n <= 2 else fibonacci(n-1) + fibonacci(n-2)


def is_prime(num):
    if num in [0, 1]:
        return False
    for x in range(2, num):
        if num % x == 0:
            return False
    else:
        return True


test = list(filter(is_prime, range(0, 1000)))
print(test)
print(test.index(571))


def solution(X):
    if X == 2:
        return 3
    elif X == 3:
        return 6

    else:
        return test[2 * (fibonacci(X) - 3)]


print(solution(10))
