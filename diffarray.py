"""
store information using difference array

so that for range update, there is linear complexity.

"""

def array_manipulation(n, queries):
    difference_lst = [0] * (n + 1)

    for (a, b, k) in queries:
        # O(M)
        a -= 1
        b -= 1
        difference_lst[a] += k
        difference_lst[b + 1] -= k

    maximum = 0
    cumsum = 0
    for l in difference_lst:
        # O(N})
        cumsum += l
        if cumsum > maximum:
            maximum = cumsum

    return maximum


if __name__ == "__main__":
    m = array_manipulation(5, [[1, 3, 3], [2, 3, 4]])
    print(m)
