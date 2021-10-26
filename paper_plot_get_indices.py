import os.path

import numpy as np


def GetIndices(n) :
    # returns n indices from the testing set that we can use to
    # make representative plots

    ID = 'vae64_200epochs_usekld_onelatent_nr2074_ontestingset'

    all_indices = []
    for idx in range(470) :
        if os.path.isfile('/scratch/gpfs/lthiele/tSZ_DeepSet_results/predictions_testing_%s_idx%d.bin'%(ID, idx)) :
            all_indices.append(idx)

    all_masses = [eval(open('/scratch/gpfs/lthiele/tSZ_DeepSet_halos/rockstar/TNG_%d_globals.bin'%idx, 'r').read())['M200c']
                  for idx in all_indices]

    all_indices = np.array(all_indices)
    all_masses = np.array(all_masses)

    sorter = np.argsort(all_masses)

    all_indices = all_indices[sorter]
    all_masses = all_masses[sorter]

    N = len(all_indices)

    out = *[all_indices[int(f*N)] for f in np.linspace(0.0, 1.0, num=n+1)[1:-1]], all_indices[-1]

    return out
