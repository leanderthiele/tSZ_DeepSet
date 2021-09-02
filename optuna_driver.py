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
from optuna import TrialPruned
from optuna.pruners import BasePruner

from data_modes import DataModes
from data_loader import DataLoader
from training import Training
from training_loss_record import TrainingLossRecord
import cfg

IDENT = argv[1]
generate_cl_lib = importlib.import_module('generate_cl_%s'%IDENT)

REMOVE_PREVIOUS = bool(int(argv[2]))


class MyPruner(BasePruner) :

    def __init__(self, warmup_steps=10, min_cut=2.0) :
        """
        warmup_steps ... number of steps before which we do not consider the loss curve
        min_cut ... if after warmup_steps//2 all losses are above min_cut, we prune
        """
    #{{{
        self.warmup_steps = warmup_steps
        self.min_cut = min_cut
    #}}}


    def prune(self, study, trial) :
    #{{{
        step = trial.last_step

        if step is not None : # =False if no scores reported yet

            loss_curve = np.array(list(trial.intermediate_values[ii] for ii in range(step+1)))

            if not np.all(np.isfinite(loss_curve)) :
                # we have nan/inf in the loss curve, should abort
                return True
            
            if step < self.warmup_steps :
                # we are in early phase of training, cannot tell if it is promising
                return False

            if np.min(loss_curve[self.warmup_steps//2:]) > self.min_cut :
                return True

        return False
    #}}}



class CallAfterEpoch :
    
    def __init__(self, trial) :
    #{{{
        self.trial = trial
    #}}}


    def __call__(self, loss_record) :
    #{{{
        assert isinstance(loss_record, TrainingLossRecord)
        loss_curve = np.median(np.array(loss_record.validation_loss_arr)
                               / np.array(loss_record.validation_guess_loss_arr),
                               axis = -1)

        self.trial.report(loss_curve[-1], len(loss_curve)-1)

        if self.trial.should_prune() :
            raise TrialPruned
    #}}}



class Objective :
    
    def __init__(self) :
    #{{{
        self.training_loader = DataLoader(DataModes.TRAINING)
        self.validation_loader = DataLoader(DataModes.VALIDATION)
    #}}}


    def __call__(self, trial) :
    #{{{
        # generate our command line arguments
        cl = generate_cl_lib.generate_cl(trial)

        # the ID for this run (use the nr to be fairly sure we really only have the number afterwards)
        ID = 'optuna_%s_nr%d'%(IDENT, trial.number)

        # set the environment variables that will be used by the training process
        os.environ['TSZ_DEEP_SET_CFG'] = ' '.join('--%s'%s for s in cl + ['ID="%s"'%ID, ])

        # get our callable which will be executed after each epoch
        call_after_epoch = CallAfterEpoch(trial)

        print('***Starting training loop (#%d) with parameters:'%trial.number)
        print('\n'.join(os.environ['TSZ_DEEP_SET_CFG'].split()))

        start_time = time()

        # run the training loop, making sure we catch any errors that may occur
        # (we expect errors to be mostly OOM, which is ok, we just cannot run this model then)
        try :
            loss_record = Training(training_loader=self.training_loader,
                                   validation_loader=self.validation_loader,
                                   call_after_epoch=call_after_epoch)
        except Exception as e :
            # there are some exceptions where we know we really should abort
            if isinstance(e, (KeyboardInterrupt, AssertionError, TrialPruned)) :
                raise e from None
            print('WARNING Training() finished with error. Will continue!')
            print(e)
            return 20.0

        end_time = time()

        print('***One training loop (#%d) took %f seconds'%(trial.number, end_time-start_time))

        assert isinstance(loss_record, TrainingLossRecord)
        loss_curve = np.median(np.array(loss_record.validation_loss_arr)
                               / np.array(loss_record.validation_guess_loss_arr),
                               axis = -1)

        # take the mean of the last few losses to make sure we are not in some spurious 
        # local minimum
        final_loss = np.mean(loss_curve[-5:])

        return final_loss
    #}}}


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
                                pruner=MyPruner(),
                                study_name=IDENT,
                                storage='sqlite:///%s.db'%IDENT,
                                load_if_exists=True)

    # construct our objective callable
    objective = Objective()

    # run the optimization loop
    study.optimize(objective)
