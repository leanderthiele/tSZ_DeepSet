import os.path

import numpy as np

import cfg


class TestingLossRecord :
    
    def __init__(self) :
    #{{{
        self.loss_arr = []
        self.KLD_arr = []
        self.guess_loss_arr = []
        self.logM_arr = []
        self.idx_arr = []
    #}}}


    def add_loss(self, loss_list, KLD_list, loss_list_guess, logM_list, idx_list) :
    #{{{
        self.loss_arr.extend(loss_list)
        self.KLD_arr.extend(KLD_list)
        self.guess_loss_arr.extend(loss_list_guess)
        self.logM_arr.extend(logM_list)
        self.idx_arr.extend(idx_list)
    #}}}


    def save(self) :
    #{{{
        np.savez(os.path.join(cfg.RESULTS_PATH, 'loss_%s.npz'%cfg.ID),
                 loss=np.array(self.loss_arr),
                 KLD=np.array(self.KLD_arr),
                 guess_loss=np.array(self.guess_loss_arr),
                 logM=np.array(self.logM_arr),
                 idx=np.array(self.idx_arr, dtype=int))
    #}}}
