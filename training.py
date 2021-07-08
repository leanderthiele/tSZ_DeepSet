import os.path

import numpy as np

import torch
import torch.nn as nn

from mpi_env_types import MPIEnvTypes
from network import Network
from network_batt12 import NetworkBatt12
from data_loader import DataLoader
from data_modes import DataModes
from training_loss import TrainingLoss
from data_batch import DataBatch
from init_proc import InitProc
import cfg

InitProc(0)

if cfg.mpi_env_type is MPIEnvTypes.NOGPU :
    torch.set_num_threads(5)
    from matplotlib import pyplot as plt

model = Network().to_device()

batt12 = NetworkBatt12().to_device() # use this to compute the reference loss
batt12.eval()

# solution adopted from
# https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348/4
# We need to do this carefully in order to avoid regularizing the layernorm weights
wd_params = list()
no_wd_params = list()
for n, p in model.named_parameters() :
    if 'linear' in n and 'weight' in n :
        wd_params.append(p)
    else :
        no_wd_params.append(p)

optimizer = torch.optim.Adam([{'params': no_wd_params, 'weight_decay': 0},
                              {'params': wd_params, 'weight_decay': cfg.WEIGHT_DECAY}])

loss_fn = TrainingLoss()

training_loader = DataLoader(mode=DataModes.TRAINING)
validation_loader = DataLoader(mode=DataModes.VALIDATION)

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **cfg.ONE_CYCLE_LR_KWARGS,
                                                steps_per_epoch=len(training_loader),
                                                epochs=cfg.EPOCHS)

# store all the losses here
training_loss_arr = []
training_guess_loss_arr = []
training_logM_arr = []
training_idx_arr = []

validation_loss_arr = []
validation_guess_loss_arr = []
validation_logM_arr = []
validation_idx_arr = []

for epoch in range(cfg.EPOCHS) :

    model.train()
    print('epoch %d'%epoch)

    this_training_loss_arr = []
    this_training_guess_loss_arr = []
    this_training_logM_arr = []
    this_training_idx_arr = []

    for t, data in enumerate(training_loader) :

        assert isinstance(data, DataBatch)

        optimizer.zero_grad()
        data = data.to_device()

        with torch.no_grad() :
            guess = batt12(data.M200c, data.TNG_radii, R200c=data.R200c if not cfg.NORMALIZE_COORDS else None)

        _, loss_list_guess = loss_fn(guess, data.TNG_Pth, w=None)

        prediction = model(data)

        if cfg.mpi_env_type is MPIEnvTypes.NOGPU :
            # FIXME diagnostics on head node
            if epoch > 0 :
                r_npy = data.TNG_radii.cpu().detach().numpy()[0, ...].squeeze()
                g_npy = guess.cpu().detach().numpy()[0, ...].squeeze()
                p_npy = prediction.cpu().detach().numpy()[0, ...].squeeze()
                t_npy = data.TNG_Pth.cpu().detach().numpy()[0, ...].squeeze()
                plt.loglog(r_npy, t_npy, linestyle='none', marker='o', label='target')
                plt.loglog(r_npy, g_npy, linestyle='none', marker='o', label='guess')
                plt.loglog(r_npy, p_npy, linestyle='none', marker='o', label='prediction')
                plt.legend()
                plt.show()
        else :
            # FIXME diagnostics to file
            if False and (epoch in [30, 80, 250] and t < 10) :
                r_npy = data.TNG_radii.cpu().detach().numpy()[0, ...].squeeze()
                g_npy = guess.cpu().detach().numpy()[0, ...].squeeze()
                p_npy = prediction.cpu().detach().numpy()[0, ...].squeeze()
                t_npy = data.TNG_Pth.cpu().detach().numpy()[0, ...].squeeze()
                np.savez('test_%s_%d_%d.npz'%(cfg.ID, epoch, t),
                         r=r_npy, g=g_npy, p=p_npy, t=t_npy)

        loss, loss_list = loss_fn(prediction, data.TNG_Pth, w=None)

        this_training_loss_arr.extend([l.item() for l in loss_list])
        this_training_guess_loss_arr.extend([l.item() for l in loss_list_guess])
        this_training_logM_arr.extend(np.log(data.M200c.cpu().detach().numpy()))
        this_training_idx_arr.extend(data.idx)

        loss.backward()
        
        if cfg.GRADIENT_CLIP is not None :
            # TODO when scaling with P200c, we can consider scaling the max_norm here
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRADIENT_CLIP)

        optimizer.step()
        scheduler.step()


    # put the losses for this epoch in the global arrays
    training_loss_arr.append(this_training_loss_arr)
    training_guess_loss_arr.append(this_training_guess_loss_arr)
    training_logM_arr.append(this_training_logM_arr)
    training_idx_arr.append(this_training_idx_arr)


    model.eval()
    
    this_validation_loss_arr = []
    this_validation_guess_loss_arr = []
    this_validation_logM_arr = []
    this_validation_idx_arr = []

    for t, data in enumerate(validation_loader) :
        
        data = data.to_device()

        with torch.no_grad() :
            guess = batt12(data.M200c, data.TNG_radii, R200c=data.R200c if not cfg.NORMALIZE_COORDS else None)
            prediction = model(data)

            _, loss_list = loss_fn(prediction, data.TNG_Pth, w=None)
            _, loss_list_guess = loss_fn(guess, data.TNG_Pth, w=None)

        this_validation_loss_arr.extend([l.item() for l in loss_list])
        this_validation_guess_loss_arr.extend([l.item() for l in loss_list_guess])
        this_validation_logM_arr.extend(np.log(data.M200c.cpu().detach().numpy()))
        this_validation_idx_arr.extend(data.idx)

    # put the validation losses in the global arrays
    validation_loss_arr.append(this_validation_loss_arr)
    validation_guess_loss_arr.append(this_validation_guess_loss_arr)
    validation_logM_arr.append(this_validation_logM_arr)
    validation_idx_arr.append(this_validation_idx_arr)


    # save all the losses so far to file 
    np.savez(os.path.join(cfg.RESULTS_PATH, 'loss_%s.npz'%cfg.ID),
             training=np.array(training_loss_arr),
             training_guess=np.array(training_guess_loss_arr),
             training_logM=np.array(training_logM_arr),
             training_idx=np.array(training_idx_arr, dtype=int),
             validation=np.array(validation_loss_arr),
             validation_guess=np.array(validation_guess_loss_arr),
             validation_logM=np.array(validation_logM_arr),
             validation_idx=np.array(validation_idx_arr, dtype=int))

# save the network to file
torch.save(model.state_dict(), os.path.join(cfg.RESULTS_PATH, 'model_%s.pt'%cfg.ID))
