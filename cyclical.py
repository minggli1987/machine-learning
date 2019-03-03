"""
cyclical


encoding of cyclical variables like time
"""

import numpy as np


def cyclical_transform(x, units_cycle=24):
    radius = x / units_cycle
    sin = np.sin(2 * np.pi * radius)
    cos = np.cos(2 * np.pi * radius)
    return sin, cos

if __name__ == "__main__":
    s, c = cyclical_transform(6)
    print(s, c)

