
def joseph(friends):
    """starting from one"""
    n = len(friends)
    if n > 1:
        print(friends)
        friends = friends[::2]
        if n % 2 != 0:
            friends.insert(0, friends.pop())
        joseph(friends)
    elif n == 1:
        print(friends)


friends = list(range(1, 95))

joseph(friends)
