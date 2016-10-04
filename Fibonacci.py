def fibonacci(n):
    assert n == int(n) and n > 0
    if n == 1 or n == 2:
        return 1
    return fibonacci(n-1) + fibonacci(n-2)

x =  10 #change the number

print(fibonacci(x))