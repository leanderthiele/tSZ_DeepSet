"""
Small script to get the inputs to our networks normalized to
standard normal.

Uses the create_scalars methods of various networks to see directly
what they would look like.

When calling this script, it is advised tp set the entire network architecture
as well as all global vectors and scalars active so that we can gather all
normalization factors we could ever want.
"""

import torch
import numpy as np

from network_encoder import NetworkEncoder
from network_local import NetworkLocal
from network_deformer import NetworkDeformer
from network_vae import NetworkVAE

from data_loader import DataLoader
from data_batch import DataBatch
from data_modes import DataModes

from init_proc import InitProc
from mpi_env_types import MPIEnvTypes

import cfg

InitProc(0)

if cfg.mpi_env_type is MPIEnvTypes.NOGPU :
    torch.set_num_threads(5)


# construct the network instances
# -- the specific architecture is not important
encoder = NetworkEncoder(1)
print('Constructed Encoder')
local = NetworkLocal(1)
print('Constructed Local')
deformer = NetworkDeformer()
print('Constructed Deformer')
vae = NetworkVAE()
print('Constructed VAE')

# construct the data loader
loader = DataLoader(DataModes.TRAINING)
print('Constructed loader')

# the descriptive strings
desc_encoder = None
desc_local = None
desc_deformer = None
desc_vae = None

# the data arrays
scalars_encoder = []
scalars_local = []
scalars_deformer = []
scalars_vae = []

def append_samples(l, s) :
    # append scalars from s (last dimension) to list l
    assert isinstance(l, list)
    assert isinstance(s, torch.Tensor)
    l.extend(s.cpu().detach().numpy().reshape(-1, s.shape[-1]))

length = len(loader)

for t, data in enumerate(loader) :

    print('%d / %d'%(t, length))

    assert isinstance(data, DataBatch)

#    print(data.TNG_coords.cpu().detach().numpy())
    scalars_encoder_, desc_encoder_ \
        = encoder.create_scalars(data.DM_coords, data.DM_vels, data.u, data.basis)
#    print(data.TNG_coords.cpu().detach().numpy())
    scalars_local_, desc_local_ \
        = local.create_scalars(data.TNG_coords, data.DM_coords_local, data.DM_N_local,
                               data.basis, data.DM_vels_local)
#    print(data.TNG_coords.cpu().detach().numpy())
    scalars_deformer_, desc_deformer_ \
        = deformer.create_scalars(data.TNG_coords, data.TNG_radii, data.u, data.basis)
    scalars_vae_, desc_vae_ \
        = vae.create_scalars(data.TNG_residuals)

    if t == 0 :
        desc_encoder = desc_encoder_
        desc_local = desc_local_
        desc_deformer = desc_deformer_
        desc_vae = desc_vae_
    else :
        assert desc_encoder == desc_encoder_
        assert desc_local == desc_local_
        assert desc_deformer == desc_deformer_
        assert desc_vae == desc_vae_

    append_samples(scalars_encoder, scalars_encoder_)
    append_samples(scalars_local, scalars_local_)
    append_samples(scalars_deformer, scalars_deformer_)
    append_samples(scalars_vae, scalars_vae_)

np.savez('normalization_data_memmap.npz',
         scalars_encoder=np.array(scalars_encoder),
         scalars_local=np.array(scalars_local),
         scalars_deformer=np.array(scalars_deformer),
         scalars_vae=np.array(scalars_vae),
         desc_encoder=desc_encoder,
         desc_local=desc_local,
         desc_deformer=desc_deformer,
         desc_vae=desc_vae)
