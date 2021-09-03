
def generate_cl(trial) :

    out = []
    out.append('NET_ARCH=dict(origin=True,batt12=True,deformer=True,encoder=False,'\
                             'scalarencoder=False,decoder=False,vae=False,local=False)')

    # from our best run with the origin network
    out.append('PRT_FRACTION["DM"]=512')
    out.append('ORIGIN_DEFAULT_NLAYERS=0')
    out.append('ORIGIN_MLP_NLAYERS=3')
    out.append('NET_ID="optuna_origin_nr459"')

    lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["deformer"]["max_lr"]=%.8e'%lr)

    wd = trial.suggest_float('wd', 1e-6, 1e-2, log=True)
    out.append('WEIGHT_DECAY["deformer"]=%.8e'%wd)

    N_layers = trial.suggest_int('N_layers', 2, 8)
    out.append('DEFORMER_NLAYERS=%d'%N_layers)

    N_hidden = trial.suggest_int('N_hidden', 32, 256)
    out.append('DEFORMER_NHIDDEN=%d'%N_hidden)

    globals_passed = trial.suggest_categorical('globals_passed', ('True', 'False'))
    out.append('DEFORMER_GLOBALS_PASSED=%s'%globals_passed)

    globals_noise = trial.suggest_float('globals_noise', 1e-2, 1e1, log=True)
    out.append('GLOBALS_NOISE=%.8e'%globals_noise)

    fix_origin = trial.suggest_categorical('fix_origin', (True, False))
    if fix_origin :
        out.append('NET_FREEZE=["origin"]')
    else :
        out.append('ONE_CYCLE_LR_KWARGS["origin"]["max_lr"]=0.00019447042667833673')
        out.append('WEIGHT_DECAY["origin"]=0.004798177481146052')

    return out
