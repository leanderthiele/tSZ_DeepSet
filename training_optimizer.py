import torch

import cfg


class TrainingOptimizer(torch.optim.Adam) :
    """
    Wrapper around Adam optimizer which
        -- takes care of the appropriate places for weight decay
        -- provides a learning rate scheduling interface
    """

    def __init__(self, model, steps_per_epoch) :
    #{{{
        # we probably need to make these attributes so they
        # are not collected
        self._wd_params = list()
        self._no_wd_params = list()
        
        # solution adopted from
        # https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348/4
        # We need to do this carefully in order to avoid regularizing the layernorm weights
        for n, p in model.named_parameters() :
            
            if 'linear' in n and 'weight' in n and p.requires_grad :
                self._wd_params.append(p)
            else :
                self._no_wd_params.append(p)

        Npars = sum(p.numel() for p in self._wd_params)

        super().__init__([{'params': self._no_wd_params,
                           'weight_decay': 0},
                          {'params': self._wd_params,
                           'weight_decay': 568192 / Npars * cfg.WEIGHT_DECAY}, ])

        self._lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self,
                                                                 steps_per_epoch=steps_per_epoch,
                                                                 epochs=cfg.EPOCHS,
                                                                 **cfg.ONE_CYCLE_LR_KWARGS)
    #}}}


    def lr_step(self) :
    #{{{ 
        self._lr_scheduler.step()
    #}}}
