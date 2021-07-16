import torch
import torch.nn as nn

import cfg


class TrainingLoss :
    """
    the loss function applied during training
    """

    def __init__(self) :
    #{{{
        self.mse = nn.MSELoss(reduction='mean')
    #}}}


    def __call__(self, x, y, KLD=None, w=None, epoch=None) :
        """
        x, y ... either tensors of shape [batch, <shape>]
                 or lists of length batch and shapes <shapei>
        KLD ... the negative KL divergence of shape [batch]
        w ... tensor of shape [batch] that contains some weights
        """
    #{{{
        # compute the individual losses (for each element in the batch)
        l = [self.mse(x[ii], y[ii]) * (w[ii] if w is not None else 1) \
             for ii in range(len(x))]
        
        if epoch is None or not cfg.KLD_ANNEALING :
            kld_scaling = cfg.KLD_SCALING
        else :
            if epoch < cfg.KLD_ANNEALING_START * cfg.EPOCHS :
                kld_scaling = 0
            elif epoch > cfg.KLD_ANNEALING_END * cfg.EPOCHS :
                kld_scaling = cfg.KLD_SCALING
            else :
                kld_scaling = cfg.KLD_SCALING \
                              * (epoch/cfg.EPOCHS - cfg.KLD_ANNEALING_START) \
                              / (cfg.KLD_ANNEALING_END - cfg.KLD_ANNEALING_START) \
        
        # TODO we may want to introduce some relative scaling here,
        #      perhaps with the loss w.r.t. B12
        return (sum(l) + kld_scaling * KLD.sum() if KLD is not None else 0) / len(l), \
               [_l.item() for _l in l], \
               [_kld.item() for _kld in KLD] if KLD is not None else None
    #}}}
