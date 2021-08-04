import numpy as np
from matplotlib import pyplot as plt

nets = ['encoder', 'local', 'deformer', 'vae', ]

scalars = dict()
desc = dict()

with np.load('normalization_data.npz') as f :
    for s in nets :
        scalars[s] = f['scalars_%s'%s]
        desc[s] = str(f['desc_%s'%s])

print(max(a.shape[-1] for a in scalars.keys()))
