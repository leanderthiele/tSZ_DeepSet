import os.path

import torch

import cfg


def _load_pretrained(model, name) :
#{{{
    # TODO I believe for distributed training this whole thing doesn't really work because
    #      PyTorch appends a 'module.' string in front of everything
    # TODO there could also be issues with the device. In the simple case of one-process training,
    #      model is already on device and the checkpoint is also on the same device index
    if isinstance(name, tuple) :
        assert len(name) > 1
        checkpoint = torch.load(os.path.join(cfg.RESULTS_PATH, 'model_%s.pt'%name[0]))
        for module in name[1:] :
            assert hasattr(model, module)
            checkpoint_part = {k.split('.', maxsplit=1)[-1] : v \
                               for k, v in checkpoint.items() \
                               if k.startswith(module)}
            getattr(model, module).load_state_dict(checkpoint_part, strict=True)
    else :
        checkpoint = torch.load(os.path.join(cfg.RESULTS_PATH, 'model_%s.pt'%name))
        model.load_state_dict(checkpoint, strict=False)
#}}}

def InitModel(model) :
    """
    model passed is already on device
    """
#{{{
    # load pretrained model(s) if requested
    if cfg.NET_ID is not None :

        if isinstance(cfg.NET_ID, list) :
            for name in cfg.NET_ID :
                _load_pretrained(model, name)
        else :
            _load_pretrained(model, cfg.NET_ID)


    # freeze parameters if requested
    for n, p in model.named_parameters() :
        
        if cfg.NET_FREEZE is not None and any(n.startswith(s) for s in cfg.NET_FREEZE) :
            p.requires_grad = False
        else :
            p.requires_grad = True
#}}}
