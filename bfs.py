#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
brute force search
"""
import sys

from itertools import product, chain
from string import printable
from multiprocessing import Pool


def _cartesian_product_generator(charset, length=1):
    for p in product(charset, repeat=length):
        yield ''.join(p)


def search(charset, max_length, product_func=_cartesian_product_generator):
    """return a generator filling entire search space."""
    lst_generators = [product_func(charset, i) for i in range(1, max_length+1)]
    for attempt in chain(*lst_generators):
        yield attempt


def check_single_password(attempt, true=r"bbbbbb"):
    return attempt, attempt == true


if __name__ == '__main__':
    try:
        charset = str(sys.argv[1])
        if charset.isdigit() and len(charset) < 3:
            raise ValueError
    except (IndexError, ValueError) as e:
        charset = printable[:94]

    try:
        max_length = int(sys.argv[2])
    except (IndexError, TypeError) as e:
        max_length = 10

    print(f'searching up to {max_length} characters among {charset}')
    attempts = search(charset, max_length)

    with open('correct_password.txt', 'w') as correct_password, \
        open('wrong_passwords.txt', 'w') as wrong_passwords:

        with Pool() as p:
            n = 0
            for k, (pwd, correct) in enumerate(p.imap(check_single_password, attempts)):
                if len(pwd) > n:
                    n = len(pwd)
                    print(f'start searching passwords of {n} characters. ')

                if k > 0 and k % 10000 == 0:
                    print(f'for example {pwd}')
                if not correct:
                    wrong_passwords.write(pwd + '\n')
                    wrong_passwords.flush()
                elif correct:
                    correct_password.write(pwd + '\n')
                    correct_password.close()
                    break
