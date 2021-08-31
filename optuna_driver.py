"""
This is the driver code for optuna. The idea is that we specify as the first command
line arguments a string ident,
such that generate_cl_<ident>.py exports a function generate_cl(trial)
which returns the command line arguments for a specific run (as a list ['ARG=value', ...]).
"""

import os.path
from sys import argv
import sys
import logging
import importlib
import subprocess
from time import time

import numpy as np
import optuna

import cfg

IDENT = argv[1]
generate_cl_lib = importlib.import_module('generate_cl_%s'%IDENT)

INDEX = 0

def objective(trial) :

    global INDEX

    # generate our command line arguments
    cl = generate_cl_lib.generate_cl(trial)

    # the ID for this run
    ID = 'optuna_%s_%d'%(IDENT, INDEX)

    start_time = time()

    # run the training loop
    subprocess.run(['python', '-u', 'training.py', '--ID="%s"'%ID,
                    *['--%s'%s for s in cl]],
                   check=True)

    end_time = time()

    print('***One training loop (#%d) took %f seconds'%(INDEX, end_time-start_time))

    # retrieve our final loss
    with np.load(os.path.join(cfg.RESULTS_PATH, 'loss_%s.npz'%ID)) as f :
        validation_loss = f['validation']
        validation_guess_loss = f['validation_guess']

    loss_curve = np.median(validation_loss/validation_guess_loss, axis=-1)
    final_loss = loss_curve[-1]

    INDEX += 1

    return final_loss


# main driver code
if __name__ == '__main__' :
    # set up some logging (according to online example)
    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))

    # set up our study (loads from data base if exists)
    study = optuna.create_study(study_name=IDENT,
                                storage='sqlite:///%s.db'%IDENT,
                                load_if_exists=True)

    # we should adapt the global index if trials have already been run
    INDEX = len(study.trials)

    # run the optimization loop
    study.optimize(objective)
