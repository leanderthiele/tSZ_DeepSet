import os.path

import numpy as np

from voxelize import Voxelize

ROOT = '/scratch/gpfs/lthiele/tSZ_DeepSet_halos/rockstar/TNG'

N = 128 # box sidelength

F = 2.5 # out to which radius (in units of R200c) we have particles

idx = 0

with Voxelize(use_gpu=True) as v :
        
    while True :

        print(idx)

        fname = lambda s : '%s_%d_%s.bin'%(ROOT, idx, s)
        
        if not os.path.isfile(fname('globals')) :
            print('Did %d boxes'%idx)
            break

        halo_globals = eval(open(fname('globals'), 'r').read())
        pos = halo_globals['pos']
        R200c = halo_globals['R200c']

        x = np.fromfile(fname('coords'), dtype=np.float32)
        x = x.reshape((len(x)//3, 3))

        x -= pos - F*R200c

        m = np.fromfile(fname('masses'), dtype=np.float32)
        d = np.fromfile(fname('densities'), dtype=np.float32)
        r = np.cbrt(3 * m / 4 / np.pi / d).astype(np.float32)
        del m
        del d

        p = np.fromfile(fname('Pth'), dtype=np.float32)

        box = np.zeros((N, N, N), dtype=np.float32)

        box = v(2*F*R200c, x, r, p, box).flatten().astype(np.float32)

        # save the original box to file so we can visually inspect interesting objects
        box.tofile(fname('box_%d_cube_Pth'%N))

        # coordinates of the box cells
        # NOTE the following works for both even and odd N
        x = np.mgrid[-N//2:N//2, -N//2:N//2, -N//2:N//2].transpose(1,2,3,0).reshape((N*N*N, 3)).astype(np.float32)
        x /= N # get in the range [-0.5, 0.5]

        # filter cells where we don't have particles
        r = np.linalg.norm(x, axis=-1)
        mask = r < 0.5

        x = x[mask]
        box = box[mask]

        # now fix the coordinate units
        x = (pos + x * 2*F*R200c).astype(np.float32)

        x.tofile(fname('box_%d_coords'%N))
        box.tofile(fname('box_%d_Pth'%N))

        idx += 1
