import numpy as np
from functools import partial



class RoundedSet(set):
    
    def __init__(self, *args, decimals=99):
        self._decimals = decimals
        self._round = partial(np.around, decimals=self._decimals)
        super(RoundedSet, self).__init__(*map(self._round, args))

    def add(self, a):
    	super(RoundedSet, self).add(self._round(a))

    def update(self, iterable):
        super(RoundedSet, self).update(map(self._round, iterable))

    def __contains__(self, a):
        return True if super(RoundedSet, self).__contains__(self._round(a)) else False


if __name__ == "__main__":
	w = RoundedSet([3.1415926535, 9.34234], decimals=4)
	print(w)
	assert 3.14159 in w

	w.update([5.555555, 6.66666])
	print(w)

	w.add(9.9999)
	print(w)
