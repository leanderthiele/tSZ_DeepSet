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
from testing_loss_record import TestingLossRecord
from data_batch import DataBatch
from init_proc import InitProc
from init_model import InitModel
from archive_cfg import ArchiveCfg
import cfg


def Testing(loader=None) :

#{{{
    InitProc(0)

    ArchiveCfg()

    if cfg.mpi_env_type is MPIEnvTypes.NOGPU :
        torch.set_num_threads(5)

    model = Network().to_device()

    InitModel(model)

    model.eval()

    # use B12 model (with updated parameters) as reference
    batt12 = NetworkBatt12().to_device()
    batt12.eval()

    loss_fn = TrainingLoss()

    if loader is None :
        loader = DataLoader(mode=DataModes.VALIDATION if cfg.TEST_ON_VALIDATION else DataModes.TESTING)

    loss_record = TestingLossRecord()

    T = len(loader)

    this_idx = -1
    all_predictions = np.empty(0, dtype=np.float32)
    save_rng = np.random.default_rng(cfg.TESTING_SEED)

    for t, data in enumerate(loader) :

        assert isinstance(data, DataBatch)

        if data.idx[0] != this_idx \
           and cfg.TEST_ON_ALL \
           and cfg.TEST_SAVE_PROB is not None \
           and len(all_predictions) != 0 :
            # save according to probability
            if save_rng.random() < cfg.TEST_SAVE_PROB :
                all_predictions.tofile(os.path.join(cfg.RESULTS_PATH,
                                                    'predictions_%s_idx%d.bin'%(cfg.ID, this_idx)))
            # reset the array
            all_predictions = np.empty(0, dtype=np.float32)

        # batch size is 1 for validation / testing
        assert len(data) == 1
        this_idx = data.idx[0]
        print('%d / %d, idx = %d'%(t, T, this_idx))

        data = data.to_device()

        with torch.no_grad() :
            guess = batt12(data.M200c, data.TNG_radii, data.P200c)
            prediction, KLD = model(data)

        _, loss_list_guess, _ = loss_fn(guess, data.TNG_Pth, w=None)
        _, loss_list, KLD_list = loss_fn(prediction, data.TNG_Pth, KLD, w=None)

        loss_record.add_loss(loss_list, KLD_list, loss_list_guess,
                             np.log(data.M200c.cpu().detach().numpy()), data.idx,
                             # this works both for lists and tensors
                             [len(a) for a in data.TNG_Pth])

        all_predictions = np.concatenate((all_predictions, prediction.cpu().detach().numpy()))

    loss_record.save()

    return loss_record
#}}}


if __name__ == '__main__' :

    Testing()
