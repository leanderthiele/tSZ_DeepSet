import os.path

import torch

import cfg


def _load_pretrained(model, name) :
#{{{
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
