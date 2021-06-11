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


    def __call__(self, x, y) :
        """
        x, y ... either tensors of shape [batch, <shape>]
                 or lists of length batch and shapes <shapei>
        """
    #{{{
        if isinstance(x, list) :
            assert isinstance(y, list) and len(x) == len(y)
            return torch.linalg.norm(torch.tensor([self(x[ii], y[ii]) for ii in range(len(x))],
                                                   requires_grad=True)) / len(x)

        return self.mse(x, y)
    #}}}
