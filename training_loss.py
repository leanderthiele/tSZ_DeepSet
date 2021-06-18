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


    def __call__(self, x, y, w=None) :
        """
        x, y ... either tensors of shape [batch, <shape>]
                 or lists of length batch and shapes <shapei>
        w ... tensor of shape [batch] that contains some weights
        """
    #{{{
        # compute the individual losses (for each element in the batch)
        l = [self.mse(x[ii], y[ii]) * (w[ii] if w is not None else 1) \
             for ii in range(len(x))]
        
        return sum(l) / len(l), l
    #}}}
