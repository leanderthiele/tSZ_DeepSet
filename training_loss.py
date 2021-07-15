import torch
import torch.nn as nn


class TrainingLoss :
    """
    the loss function applied during training
    """

    def __init__(self) :
    #{{{
        self.mse = nn.MSELoss(reduction='mean')
    #}}}


    def __call__(self, x, y, KLD=None, w=None) :
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
        
        return (sum(l) + KLD.sum() if KLD is not None else 0) / len(l), \
               [_l.item() for _l in l], \
               [_kld.item() for _kld in KLD] if KLD is not None else None
    #}}}
