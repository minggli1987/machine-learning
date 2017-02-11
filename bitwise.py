
string = '100000000'

a = bin(1)

b = bin(2)

c = bin(5)

d = bin(8)

e = bin(16)

for i in range(33):
	print(str(i), bin(i))


def power_of_two(n):
	print(bin(n & (n - 1)))
	return n & (n - 1) == 0 if n > 0 else False

print(power_of_two(7))

def even_number(n):
	return n & 1 == 0

def odd_number(n):
	return n & 1 == 1

print(odd_number(3), even_number(3))
