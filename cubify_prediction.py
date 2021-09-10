"""
Exports the function CubifyPrediction(path) which takes path as a binary file
of flattened Pth values in canonical order and returns them on a regularized
grid with un-specified values set to zero
"""

import numpy as np

import cfg


def CubifyPrediction(path) :

    p = np.fromfile(path, dtype=np.float32)

    N = cfg.TNG_RESOLUTION

    # here we store our final result
    box = np.zeros(N*N*N, dtype=np.float32)

    # coordinates of box cells
    x = np.mgrid[-N//2:N//2, -N//2:N//2, -N//2:N//2].transpose(1,2,3,0).reshape((N*N*N, 3)).astype(np.float32)
    x /= N # now in the range [-0.5, 0.5]

    # filter cells where we don't have any particles
    r = np.linalg.norm(x, axis=-1).flatten()
    mask1 = r < 0.5

    # filter by radius outside of which we don't predict
    mask2 = r[mask1] < (cfg.RMAX / 5.0)

    # numpy is a bit funny here, but this works
    p1 = np.zeros(np.count_nonzero(mask1), dtype=np.float32)
    p1[mask2] = p

    # now fill the box
    box[mask1] = p1

    return box.reshape((N, N, N))
