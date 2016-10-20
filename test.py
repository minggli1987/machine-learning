import numpy as np

p = np.random.randint(50, 100, 24 * 60)
ix = range(1, 24 * 60 + 1, 1)

prices = dict()

for k, v in zip(ix, p):
    prices[k] = v

buying_positions = prices.items()

max_p_diff = 0
optimal_positions = list()
increase = list()

for i in buying_positions:
    for j in buying_positions:
        # buying position must appear before selling position and have lower price than selling
        if i[0] < j[0] and j[1] - i[1] >= max_p_diff:
            increase.append(j[1] - i[1] - max_p_diff)
            max_p_diff = j[1] - i[1]
            optimal_positions = (i, j)


for i in range(1, 101, 1):
    if i / 15 == i // 15:
        print('fizzbuzz')
    elif i / 3 == i // 3:
        print('fizz')
    elif i / 5 == i // 5:
        print('buzz')
    else:
        print(i)