testlist = list(range(1, 10, 2))

print(testlist)

def listsum(list):
    if len(list) == 1:
        return list[0]
    else:
        return list[0] + listsum(list[1:])

print(listsum(testlist))

def toStr(n, base):
    convertString = '0123456789ABCDEF'
    if n < base:
        return convertString[n]
    else:
        return toStr(n // base, base) + convertString[n%base]

print(toStr(1453 ,2))

