"""
cyclical


encoding of cyclical variables like time
"""

import numpy as np


def cyclical_transform(x, units_cycle=24):
    radians = x / units_cycle
    sin = np.sin(2 * np.pi * radians)
    cos = np.cos(2 * np.pi * radians)
    return sin, cos

if __name__ == "__main__":
    s, c = cyclical_transform(6)
    print(s, c)

