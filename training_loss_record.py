import numpy as np

import cfg

class TrainingLossRecord :

    def __init__(self) :
    #{{{
        # holds all training losses etc
        self.training_loss_arr = []
        self.training_KLD_arr = []
        self.training_guess_loss_arr = []
        self.training_logM_arr = []
        self.training_idx_arr = []

        # holds all validation losses etc
        self.validation_loss_arr = []
        self.validation_KLD_arr = []
        self.validation_guess_loss_arr = []
        self.validation_logM_arr = []
        self.validation_idx_arr = []

        # holds training losses for this epoch
        self.this_training_loss_arr = []
        self.this_training_KLD_arr = []
        self.this_training_guess_loss_arr = []
        self.this_training_logM_arr = []
        self.this_training_idx_arr = []

        # holds validation losses for this epoch
        self.this_validation_loss_arr = []
        self.this_validation_KLD_arr = []
        self.this_validation_guess_loss_arr = []
        self.this_validation_logM_arr = []
        self.this_validation_idx_arr = []
    #}}}


    def add_training_loss(self, loss_list, KLD_list, loss_list_guess, logM_list, idx_list) :
    #{{{
        self.this_training_loss_arr.extend(loss_list)
        self.this_training_KLD_arr.extend(KLD_list)
        self.this_training_guess_loss_arr.extend(loss_list_guess)
        self.this_training_logM_arr.extend(logM_list)
        self.this_training_idx_arr.extend(idx_list)
    #}}}


    def add_validation_loss(self, loss_list, KLD_list, loss_list_guess, logM_list, idx_list) :
    #{{{
        self.this_validation_loss_arr.extend(loss_list)
        self.this_validation_KLD_arr.extend(KLD_list)
        self.this_validation_guess_loss_arr.extend(loss_list_guess)
        self.this_validation_logM_arr.extend(logM_list)
        self.this_validation_idx_arr.extend(idx_list)
    #}}}

    
    def end_epoch(self) :
    #{{{
        # put into global training array
        self.training_loss_arr.append(self.this_training_loss_arr)
        self.training_KLD_arr.append(self.this_training_KLD_arr)
        self.training_guess_loss_arr.append(self.this_training_guess_loss_arr)
        self.training_logM_arr.append(self.this_training_logM_arr)
        self.training_idx_arr.append(self.this_training_idx_arr)

        # put into global validation array
        self.validation_loss_arr.append(self.this_validation_loss_arr)
        self.validation_KLD_arr.append(self.this_validation_KLD_arr)
        self.validation_guess_loss_arr.append(self.this_validation_guess_loss_arr)
        self.validation_logM_arr.append(self.this_validation_logM_arr)
        self.validation_idx_arr.append(self.this_validation_idx_arr)

        # holds training losses for this epoch
        self.this_training_loss_arr = []
        self.this_training_KLD_arr = []
        self.this_training_guess_loss_arr = []
        self.this_training_logM_arr = []
        self.this_training_idx_arr = []

        # holds validation losses for this epoch
        self.this_validation_loss_arr = []
        self.this_validation_KLD_arr = []
        self.this_validation_guess_loss_arr = []
        self.this_validation_logM_arr = []
        self.this_validation_idx_arr = []

        # save to file
        np.savez(os.path.join(cfg.RESULTS_PATH, 'loss_%s.npz'%cfg.ID),
                 training=np.array(self.training_loss_arr),
                 training_KLD=np.array(self.training_KLD_arr),
                 training_guess=np.array(self.training_guess_loss_arr),
                 training_logM=np.array(self.training_logM_arr),
                 training_idx=np.array(self.training_idx_arr, dtype=int),
                 validation=np.array(self.validation_loss_arr),
                 validation_KLD=np.array(self.validation_KLD_arr),
                 validation_guess=np.array(self.validation_guess_loss_arr),
                 validation_logM=np.array(self.validation_logM_arr),
                 validation_idx=np.array(self.validation_idx_arr, dtype=int))
    #}}}
