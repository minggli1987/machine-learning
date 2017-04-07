import numpy as np


class Board(object):

    def __init__(self, m=8, n=8):
        self.board = np.zeros(shape=(m, n))

    def cast(self, chesspiece):
        self.board[chesspiece.row, chesspiece.col] = 2
        self._eval(chesspiece)

    @property
    def avail(self):
        return [(i, j) for i in range(self.board.shape[0])
                for j in range(self.board.shape[1]) if self.board[i, j] == 0]

    def reset(self):
        self.board = np.zeros(shape=self.board.shape)

    def _eval(self, chesspiece):
        for pos in self.avail:
            if chesspiece.captures(chesspiece.__class__(pos[0], pos[1])):
                self.board[pos[0], pos[1]] = 1

    def __repr__(self):
        print(self.board)
        return ''


class Queen(object):

    def __init__(self, row, col):
        self.row = row
        self.col = col

    def captures(self, other):
        if any([self.row == other.row,
                self.col == other.col,
                abs(self.col - other.col) == abs(self.row - other.row)]
               ):
            return True
        else:
            return False

    def __repr__(self):
        return 'Queen located at ({0}, {1})'.format(self.row, self.col)


def timeit(func):
    import time

    def wrapper(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        end = time.time()
        print('function {0} took {1:0.3f} ms'.format(
              func.__name__, (end - start) * 1000))
        return output
    return wrapper


@timeit
def fit_queens(game, num_queens=0):
    queens_fit = 0
    while queens_fit < num_queens:
        try:
            """randomly pick an unattacked position"""
            pos = np.random.permutation(game.avail)[0]
            game.cast(Queen(pos[0], pos[1]))
            queens_fit += 1
        except IndexError:
            # no more available positions left on the chessboard before
            # reaching 8 queens.
            game.reset()
            queens_fit = 0
            continue
    return game


if __name__ == '__main__':
    print(fit_queens(game=Board(8, 8), num_queens=8))
