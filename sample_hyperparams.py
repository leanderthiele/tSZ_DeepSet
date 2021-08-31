from sys import argv

import numpy as np

SEED = int(argv[1])
RNG = np.random.default_rng(SEED)


def draw_float(a, b, log=True) :
    if not log :
        return RNG.uniform(a, b)
    else :
        return a * (b/a)**RNG.random()


def draw_int(a, b, log=True) :
    if not log :
        return RNG.integers(a, b, endpoint=True)
    else :
        return int(round(a * (b/a)**RNG.random()))


def draw_binary() :
    return RNG.choice([True, False])


def draw_discrete(options) :
    return RNG.choice(options)
