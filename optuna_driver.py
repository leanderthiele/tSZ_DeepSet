"""
This is the driver code for optuna. The idea is that we specify as the first command
line arguments a string <ident>,
such that generate_cl_<IDENT>.py exports a function generate_cl(trial)
which returns the command line arguments for a specific run (as a list ['ARG=value', ...]).
"""

import os
import os.path
from glob import glob
import sys
import logging
import importlib
from time import time
from argparse import ArgumentParser

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

ARG_PARSER = ArgumentParser()
ARG_PARSER.add_argument('ident', type=str,
                        help='identifier for the current run. A file generate_cl_<ident>.py has to exist.')
ARG_PARSER.add_argument('-rm', '--remove-previous', action='store_true',
                        help='whether data produced by previous runs with same <ident> should be removed.')
ARG_PARSER.add_argument('-es', '--early-stopping', action='store_true',
                        help='whether some early stopping should be applied (probably not usable at the moment)')
ARG_PARSER.add_argument('-nr', '--num-random', type=int, default=20,
                        help='number of initial random trials')
ARG_PARSER.add_argument('-vae', '--has-vae', action='store_true',
                        help='set if architecture has probabilistic component, requiring multi-objective optimization.')
ARG_PARSER.add_argument('-kld', '--use-kld', action='store_true',
                        help='to include KL divergence instead of mean gaussian loss '\
                             'in multi-objective optimization, only relevant if has-vae.')

ARGS = ARG_PARSER.parse_args()

# give some print output to have a record
print('**** optuna_driver.py starting with settings ****')
print(ARGS)


# import the library containing the function we use to generate trial command lines
generate_cl_lib = importlib.import_module('generate_cl_%s'%ARGS.ident)


class MyPruner(BasePruner) :

    def __init__(self, warmup_steps=20) :
        """
        warmup_steps ... number of steps before which we do not consider the loss curve
        """
    #{{{
        self.warmup_steps = warmup_steps
    #}}}


    @staticmethod
    def should_prune(loss_curve, step, warmup_steps) :
    #{{{
        if not np.all(np.isfinite(loss_curve)) :
            # we have nan/inf in the loss curve, should abort
            return True
        
        if step < warmup_steps :
            # we are in early phase of training, cannot tell if it is promising
            return False

        if step > 0.7 * cfg.EPOCHS :
            # now we have spent so much time on this, let us finish the run
            return False
        
        if np.min(loss_curve) < 1.0 :
            # we have made some progress, so it is clear we should continue
            # NOTE this is something we should probably play with later.
            #      Currently it only serves to exclude the origin training from pruning
            #      as we expect it to be fairly noisy initially for good results.
            return False

        if np.median(loss_curve[warmup_steps//2:]) > np.median(loss_curve[:warmup_steps//2]) :
            # we are not making progress
            return True

        return False
    #}}}


    def prune(self, study, trial) :
    #{{{
        step = trial.last_step

        if step is not None : # =False if no scores reported yet

            loss_curve = np.array(list(trial.intermediate_values[ii] for ii in range(step+1)))

            return MyPruner.should_prune(loss_curve, step, self.warmup_steps)

        return False
    #}}}


class TrainingAbort(Exception) :
    pass    


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

        if MyPruner.should_prune(loss_curve, len(loss_curve)-1, 20) :
            raise TrainingAbort

#        self.trial.report(loss_curve[-1], len(loss_curve)-1)
#        if self.trial.should_prune() :
#            raise TrialPruned
    #}}}



class Objective :
    
    def __init__(self) :
    #{{{
        self.training_loader = DataLoader(DataModes.TRAINING)
        self.validation_loader = DataLoader(DataModes.VALIDATION)

        self.n_trials = 0
    #}}}


    def __call__(self, trial) :
    #{{{
        # advance our counter
        self.n_trials += 1

        # generate our command line arguments
        cl = generate_cl_lib.generate_cl(trial)

        # the ID for this run (use the nr to be fairly sure we really only have the number afterwards)
        ID = 'optuna_%s_nr%d'%(ARGS.ident, trial.number)

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
                                   call_after_epoch=call_after_epoch if ARGS.early_stopping else None)
        except TrainingAbort :
            print('WARNING Training() finished prematuraly because loss curve did not look good.')
            return 20.0
        except (AssertionError, TrialPruned) as e :
            raise e from None
        except Exception as e :
            if self.n_trials == 1 :
                # we are in the first run overall, it is highly unlikely that we encountered something
                # like OOM for which we would want to continue.
                raise e from None
            print('WARNING Training() finished with error. Will continue!')
            print(e)
            return 20.0

        end_time = time()

        print('***One training loop (#%d) took %f seconds'%(trial.number, end_time-start_time))

        assert isinstance(loss_record, TrainingLossRecord)

        loss_curve = np.median(np.array(loss_record.validation_loss_arr)
                               / np.array(loss_record.validation_guess_loss_arr),
                               axis=-1)

        # take the mean of the last few losses to make sure we are not in some spurious 
        # local minimum
        final_loss = np.mean(loss_curve[-5:])

        if ARGS.has_vae :
            if ARGS.use_kld :
                kld_curve = np.mean(np.array(loss_record.validation_KLD_arr), axis=-1)

                final_kld = np.mean(kld_curve[-5:])
            else :
                # we first average the Gaussian loss over random seeds, then take the median
                loss_curve_gaussian = np.median(np.mean(np.array(loss_record.validation_gauss_loss_arr), axis=-1)
                                                / np.array(loss_record.validation_guess_loss_arr),
                                                axis=-1)
                final_gaussian_loss = np.mean(loss_curve_gaussian[-5:])


        return (final_loss, final_kld if ARGS.use_kld else final_gaussian_loss) if ARGS.has_vae else final_loss 
    #}}}


# main driver code
if __name__ == '__main__' :
    
    if ARGS.remove_previous :
        print('remove-previous==True, will delete all previously generated data with same <ident>')
        # the SQL data base where optuna puts its stuff
        db_fname = '%s.db'%ARGS.ident
        if os.path.isfile(db_fname) :
            os.remove(db_fname)
        for prefix, suffix in [('loss', 'npz'), ('cfg', 'py'), ('model', 'pt')] :
            # this is not ideal because glob doesn't know regular expressions, but should be
            # safe unless we do something stupid like taking ..._nr as ident
            fnames = glob(os.path.join(cfg.RESULTS_PATH, '%s_optuna_%s_nr[0-9]*.%s'%(prefix, ARGS.ident, suffix)))
            for fname in fnames :
                os.remove(fname)

    # set up some logging (according to online example)
    optuna.logging.get_logger('optuna').addHandler(logging.StreamHandler(sys.stdout))

    # set up our study (loads from data base if exists)
    study = optuna.create_study(sampler=TPESampler(n_startup_trials=ARGS.num_random),
                                study_name=ARGS.ident,
                                storage='sqlite:///%s.db'%ARGS.ident,
                                directions=["minimize", ] * (1+ARGS.has_vae),
                                load_if_exists=True)

    # construct our objective callable
    objective = Objective()

    # run the optimization loop
    study.optimize(objective)
