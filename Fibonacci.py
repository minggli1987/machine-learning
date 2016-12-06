def fibonacci(n):
    assert n == int(n) and n > 0
    if n == 1 or n == 2:
        return 1
    return fibonacci(n-1) + fibonacci(n-2)

x =  7 #change the number

print(fibonacci(x))



def power_of_two(x):
    return x == 1 if x < 2 else power_of_two(x/2)


print(power_of_two(8))