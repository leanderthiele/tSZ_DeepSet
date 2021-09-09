
def generate_cl(trial) :
    
    out = []
    out.append('NET_ARCH=dict(origin=True,batt12=True,deformer=False,encoder=False,'\
                             'scalarencoder=False,decoder=False,vae=False,local=False)')
    
    out.append('PRT_FRACTION["DM"]["validation"]=8192')
    out.append('PRT_FRACTION["TNG"]["validation"]=16384')
    out.append('VALIDATION_EPOCHS=10')

    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["origin"]["max_lr"]=%.8e'%lr)

    wd = trial.suggest_float('wd', 1e-5, 1e-1, log=True)
    out.append('WEIGHT_DECAY["origin"]=%.8e'%wd)

    N_DM = trial.suggest_int('N_DM', 1e2, 1e4, log=True)
    out.append('PRT_FRACTION["DM"]["training"]=%d'%N_DM)

    N_layers = trial.suggest_int('N_layers', 0, 2)
    out.append('ORIGIN_DEFAULT_NLAYERS=%d'%N_layers)

    MLP_N_layers = trial.suggest_int('MLP_N_layers', 2, 5)
    out.append('ORIGIN_MLP_NLAYERS=%d'%MLP_N_layers)

    N_hidden = trial.suggest_int('N_hidden', 32, 256)
    out.append('ORIGIN_MLP_NHIDDEN=%d'%N_hidden)
    out.append('ORIGIN_DEFAULT_NHIDDEN=%d'%N_hidden)

    globals_noise = trial.suggest_float('globals_noise', 2.0, 30.0)
    out.append('GLOBALS_NOISE=%.8e'%globals_noise)

    basis_noise = trial.suggest_float('basis_noise', 0.0, 10.0)
    out.append('BASIS_NOISE=%.8e'%basis_noise)

    return out
