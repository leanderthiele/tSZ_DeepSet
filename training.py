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
from training_optimizer import TrainingOptimizer
from training_loss_record import TrainingLossRecord
from data_batch import DataBatch
from init_proc import InitProc
from init_model import InitModel
from archive_cfg import ArchiveCfg
import cfg

InitProc(0)

ArchiveCfg()

if cfg.mpi_env_type is MPIEnvTypes.NOGPU :
    torch.set_num_threads(5)
    from matplotlib import pyplot as plt

model = Network().to_device()

InitModel(model)

batt12 = NetworkBatt12().to_device() # use this to compute the reference loss
batt12.eval()

loss_fn = TrainingLoss()

training_loader = DataLoader(mode=DataModes.TRAINING)
validation_loader = DataLoader(mode=DataModes.VALIDATION)

optimizer = TrainingOptimizer(model, steps_per_epoch=len(training_loader))

# store all the losses here
loss_record = TrainingLossRecord()

for epoch in range(cfg.EPOCHS) :

    model.train()
    print('epoch %d'%epoch)

    for t, data in enumerate(training_loader) :

        assert isinstance(data, DataBatch)

        optimizer.zero_grad()
        data = data.to_device()

        with torch.no_grad() :
            guess = batt12(data.M200c, data.TNG_radii, data.P200c)

        _, loss_list_guess, _ = loss_fn(guess, data.TNG_Pth, w=None)

        prediction, KLD = model(data)

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

        loss, loss_list, KLD_list = loss_fn(prediction, data.TNG_Pth, KLD, w=None, epoch=epoch)

        loss_record.add_training_loss(loss_list, KLD_list, loss_list_guess, 
                                      np.log(data.M200c.cpu().detach().numpy()), data.idx)

        loss.backward()
        
        if cfg.GRADIENT_CLIP is not None :
            # TODO when scaling with P200c, we can consider scaling the max_norm here
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.GRADIENT_CLIP)

        optimizer.step()
        optimizer.lr_step()


    model.eval()
    
    for t, data in enumerate(validation_loader) :
        
        data = data.to_device()

        with torch.no_grad() :
            guess = batt12(data.M200c, data.TNG_radii, data.P200c)
            prediction, KLD = model(data)

            _, loss_list, KLD_list = loss_fn(prediction, data.TNG_Pth, KLD, w=None, epoch=epoch)
            _, loss_list_guess, _ = loss_fn(guess, data.TNG_Pth, w=None)

        loss_record.add_validation_loss(loss_list, KLD_list, loss_list_guess,
                                        np.log(data.M200c.cpu().detach().numpy()), data.idx)

    # gather losses and save to file
    loss_record.end_epoch()

# save the network to file
torch.save(model.state_dict(), os.path.join(cfg.RESULTS_PATH, 'model_%s.pt'%cfg.ID))
