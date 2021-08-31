import torch

from merge_dicts import MergeDicts
import cfg


class _NoParamError(ValueError) :
    pass


class _ModuleOptimizer(torch.optim.Adam) :
    """
    Wrapper around Adam optimizer which
        -- takes care of the appropriate places for weight decay
        -- provides a learning rate scheduling interface

    Operates only on the parameters of the model in a specific module,
    which is defined by the `name' argument.
    """

    def __init__(self, model, name, steps_per_epoch) :
        """
        Construct optimizer for all parameters containing name in the model.
        Exception: if name is a list, it is reverse, i.e. construct optimizer
                   for all parameters not in the list
                   (this is our way to deal with "un-moduled" parameters
        """
    #{{{
        # we probably need to make these attributes so they
        # are not collected
        self._wd_params = list()
        self._no_wd_params = list()

        self.weight_decay = cfg.WEIGHT_DECAY[name] \
                            if not isinstance(name, list) and name in cfg.WEIGHT_DECAY \
                            else cfg.WEIGHT_DECAY['default']

        self.one_cycle_lr_kwargs = MergeDicts(cfg.ONE_CYCLE_LR_KWARGS[name], \
                                              cfg.ONE_CYCLE_LR_KWARGS['default']) \
                                   if not isinstance(name, list) and name in cfg.ONE_CYCLE_LR_KWARGS \
                                   else cfg.ONE_CYCLE_LR_KWARGS['default']

        # solution adopted from
        # https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348/4
        # We need to do this carefully in order to avoid regularizing the layernorm weights
        for n, p in model.named_parameters() :

            if isinstance(name, list) and any(n in s for s in name) :
                continue

            if not isinstance(name, list) and name not in n :
                # we only collect parameters for a specific module
                continue
            
            # we only apply weight decay to the weight matrices in linear layers.
            # Furthermore, we want to exclude the local network from weight decay
            # since we have more than enough training samples there.
            if 'linear' in n and 'weight' in n and p.requires_grad :
                self._wd_params.append(p)
            else :
                self._no_wd_params.append(p)

        self._param_groups = list()

        # note : boolean is whether list has elements
        if self._no_wd_params :
            self._param_groups.append({'params': self._no_wd_params,
                                       'weight_decay': 0})
        if self._wd_params :
            Npars = sum(p.numel() for p in self._wd_params)
            self._param_groups.append({'params': self._wd_params,
                                       'weight_decay': 568192 / Npars * self.weight_decay})

        if not self._param_groups :
            # no parameters found, raise Error (this is the only way to get out of the constructor)
            raise _NoParamError

        super().__init__(self._param_groups)

        self._lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(self,
                                                                 steps_per_epoch=steps_per_epoch,
                                                                 epochs=cfg.EPOCHS,
                                                                 **self.one_cycle_lr_kwargs)
    #}}}

    
    def lr_step(self) :
    #{{{ 
        self._lr_scheduler.step()
    #}}}



class TrainingOptimizer(list) :
    """
    Contains a bunch of _ModuleOptimizer's operating of different parts of the architecture.
    """

    def __init__(self, model, steps_per_epoch) :
    #{{{
        super().__init__()
        
        # loop over the named modules
        for k, v in cfg.NET_ARCH.items() :

            if not v :
                continue
            
            self.append(_ModuleOptimizer(model, k, steps_per_epoch))


        # now do the un-moduled parameters
        # it is possible that we don't have any such parameters, so we should be careful
        try :
            self.append(_ModuleOptimizer(model, list(cfg.NET_ARCH.keys()), steps_per_epoch))
        except _NoParamError :
            pass
    #}}}

    def __getattr__(self, name) :
        """
        a little trick to map Adam methods to our list of optimizers
        """
    #{{{ 
        assert name in ['step', 'lr_step', 'zero_grad', ]

        def out(*args, _s=self, **kwargs) :
            for x in _s :
                getattr(x, name)(*args, **kwargs)

        return out
    #}}}
