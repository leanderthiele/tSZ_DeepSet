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

def Training(training_loader=None, validation_loader=None, call_after_epoch=None) :
    """
    The entire training process. If training_loader and validation_loader are passed,
    there are certain restrictions on the changes that can be made in cfg.
    (usually not important)

    call_after_epoch ... will be called with the TrainingLossRecord instance as argument.

    Returns a TrainingLossRecord instance gathering all the training losses.
    """
#{{{
    InitProc(0)

    ArchiveCfg()

    if cfg.mpi_env_type is MPIEnvTypes.NOGPU :
        torch.set_num_threads(5)

    model = Network().to_device()

    InitModel(model)

    # use B12 model (with updated parameters) as reference
    batt12 = NetworkBatt12().to_device()
    batt12.eval()

    loss_fn = TrainingLoss()

    if training_loader is None :
        training_loader = DataLoader(mode=DataModes.TRAINING)
    
    if validation_loader is None :
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

        # if user requested, call their callback function
        if call_after_epoch is not None :
            assert callable(call_after_epoch)
            call_after_epoch(loss_record)

    # save the network to file
    torch.save(model.state_dict(), os.path.join(cfg.RESULTS_PATH, 'model_%s.pt'%cfg.ID))

    return loss_record
#}}}


if __name__ == '__main__' :
    
    Training()
