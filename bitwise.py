
string = '100000000'

a = bin(1)

b = bin(2)

c = bin(5)

d = bin(8)

e = bin(16)


def power_of_two_bitwise(n):
    """bitwise power of two check."""
    return n > 0 and n & (n - 1) == 0


def power_of_two_bitstring(n):
    """only has 1 in binary string."""
    return n > 0 and bin(n).count('1') == 1


def power_of_two_recursive(n):
    return n == 1 if n < 2 else power_of_two_recursive(n/2)


print(power_of_two_recursive(32))


def even_number(n):
    """last bit will be 0."""
    return n & 1 == 0


def odd_number(n):
    """odd number has 1 in last bit."""
    return n & 1 == 1


print(odd_number(3), even_number(3))
