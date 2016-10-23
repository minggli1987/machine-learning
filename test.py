import numpy as np
from operator import itemgetter

p = np.random.randint(50, 100, 24 * 60)
ix = range(1, 24 * 60 + 1, 1)

prices = dict()

for k, v in zip(ix, p):
    prices[k] = v  # prices in dictionary with key being range and values being prices

buying_positions = sorted(prices.items(), key=itemgetter(1))
selling_positions = sorted(prices.items(), key=itemgetter(1), reverse=True)

max_p_diff = 0
optimal_positions = list()
margins = list()

for i in buying_positions:
    for j in selling_positions:
        # buying position must appear before selling position in timing and have lower price than selling
        if i[0] < j[0] and j[1] - i[1] >= max_p_diff:
            if j[1] - i[1] - max_p_diff > 0:
                margins.append(j[1] - i[1] - max_p_diff)
            max_p_diff = j[1] - i[1]
            optimal_positions.append((i, j))

print(optimal_positions)


def conditional_print():

    for i in range(1, 101, 1):
        if i % 15 == 0:
            print('fizzbuzz')
        elif i % 3 == 0:
            print('fizz')
        elif i % 5 == 0:
            print('buzz')
        else:
            print(i)

