"""
This is the driver code for optuna. The idea is that we specify as the first command
line arguments a string IDENT,
such that generate_cl_<IDENT>.py exports a function generate_cl(trial)
which returns the command line arguments for a specific run (as a list ['ARG=value', ...]).

Second command line argument is an integer which if non-zero means that any previous runs
with the same IDENT should be discarded and the resulting data products (SQL database, loss curves, ...)
should be removed
"""

import os
import os.path
from sys import argv
from glob import glob
import sys
import logging
import importlib
import subprocess
from time import time

import numpy as np
import optuna
from optuna.samplers import TPESampler

from data_modes import DataModes
from data_loader import DataLoader
from training import Training
from training_loss_record import TrainingLossRecord
import cfg

IDENT = argv[1]
generate_cl_lib = importlib.import_module('generate_cl_%s'%IDENT)

REMOVE_PREVIOUS = bool(int(argv[2]))

INDEX = 0

TRAINING_LOADER = None
VALIDATION_LOADER = None

def objective(trial) :

    global INDEX
    global TRAINING_LOADER
    global VALIDATION_LOADER

    # generate our command line arguments
    cl = generate_cl_lib.generate_cl(trial)

    # the ID for this run (use the nr to be fairly sure we really only have the number afterwards)
    ID = 'optuna_%s_nr%d'%(IDENT, INDEX)

    # set the environment variables that will be used by the training process
    os.environ['TSZ_DEEP_SET_CFG'] = ' '.join('--%s'%s for s in cl)

    start_time = time()

    # run the training loop, making sure we catch any errors that may occur
    # (we expect errors to be mostly OOM, which is ok, we just cannot run this model then)
    try :
        loss_record = Training(training_loader=TRAINING_LOADER, validation_loader=VALIDATION_LOADER)
    except Exception as e :
        # make sure we abort if the user really wants it
        if isinstance(e, KeyboardInterrupt) :
            raise e from None
        print('WARNING Training() finished with error. Will continue!')
        print(e)
        return 20.0

    end_time = time()

    print('***One training loop (#%d) took %f seconds'%(INDEX, end_time-start_time))

    assert isinstance(loss_record, TrainingLossRecord)
    loss_curve = np.median(np.array(loss_record.validation_loss_arr)
                           / np.array(loss_record.validation_guess_loss_arr),
                           axis = -1)

    # take the mean of the last few losses to make sure we are not in some spurious 
    # local minimum
    final_loss = np.mean(loss_curve[-5:])

    INDEX += 1

    return final_loss


# main driver code
if __name__ == '__main__' :
    
    if REMOVE_PREVIOUS :
        print('REMOVE_PREVIOUS==True, will delete all previously generated data with same IDENT')
        os.remove('%s.db'%IDENT)
        for prefix, suffix in [('loss', 'npz'), ('cfg', 'py'), ('model', 'pt')] :
            # this is not ideal because glob doesn't know regular expressions, but should be
            # safe unless we do something stupid like taking ..._nr as IDENT
            fnames = glob(os.path.join(cfg.RESULTS_PATH, '%s_optuna_%s_nr[0-9]*.%s'%(prefix, IDENT, suffix)))
            for fname in fnames :
                os.remove(fname)

    # set up some logging (according to online example)
    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))

    # set up our study (loads from data base if exists)
    study = optuna.create_study(sampler=TPESampler(n_startup_trials=20),
                                study_name=IDENT,
                                storage='sqlite:///%s.db'%IDENT,
                                load_if_exists=True)

    # we should adapt the global index if trials have already been run
    INDEX = len(study.trials)

    # construct the data loaders
    TRAINING_LOADER = DataLoader(DataModes.TRAINING)
    VALIDATION_LOADER = DataLoader(DataModes.VALIDATION)

    # run the optimization loop
    study.optimize(objective)
