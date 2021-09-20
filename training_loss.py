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


    def __call__(self, x, y, KLD=None, w=None, epoch=None, guess_loss=None) :
        """
        x, y ... either tensors of shape [batch, <shape>]
                 or lists of length batch and shapes <shapei>
        KLD ... the negative KL divergence of shape [batch]
        w ... tensor of shape [batch] that contains some weights
        guess_loss ... list/tensor of shape [batch] with the losses of the GNFW benchmark
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

        if KLD is not None and cfg.KLD_NORM :
            assert guess_loss is not None
            normed_kld = torch.tensor([k * guess_loss[ii] / 3e-2 for ii, k in enumerate(KLD)],
                                      requires_grad=True, dtype=torch.float32, device=KLD.device)
        else :
            normed_kld = None
        
        return (sum(l) + kld_scaling * (normed_kld if normed_kld is not None else KLD).sum() \
                if KLD is not None else 0) / len(l), \
               [_l.item() for _l in l], \
               [_kld.item() for _kld in KLD] if KLD is not None else None
    #}}}
