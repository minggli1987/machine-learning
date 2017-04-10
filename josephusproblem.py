def joseph(friends):
    """starting from one"""
    n = len(friends)
    if n > 1:
        print(friends)
        friends = friends[::2]
        if n % 2 != 0:
            friends.insert(0, friends.pop())
        return joseph(friends)
    elif n == 1:
        return friends[0]


friends = list(range(1, 77))
a = joseph(friends)
print(a)
