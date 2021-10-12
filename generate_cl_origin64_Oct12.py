
def generate_cl(trial) :
    
    out = []
    out.append('NET_ARCH=dict(origin=True,batt12=True,deformer=False,encoder=False,'\
                             'scalarencoder=False,decoder=False,vae=False,local=False)')
    
    out.append('PRT_FRACTION["DM"]["training"]=4096')
    out.append('PRT_FRACTION["DM"]["validation"]=8192')
    out.append('PRT_FRACTION["TNG"]["validation"]=16384')
    out.append('VALIDATION_EPOCHS=6')

    # learned previously
    out.append('ORIGIN_MLP_NHIDDEN=230')
    out.append('ORIGIN_DEFAULT_NHIDDEN=230')

    lr = trial.suggest_float('lr', 1e-5, 5e-4, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["origin"]["max_lr"]=%.8e'%lr)

    wd = trial.suggest_float('wd', 1e-3, 1e0, log=True)
    out.append('WEIGHT_DECAY["origin"]=%.8e'%wd)

    N_layers = trial.suggest_int('N_layers', 0, 2)
    out.append('ORIGIN_DEFAULT_NLAYERS=%d'%N_layers)

    MLP_N_layers = trial.suggest_int('MLP_N_layers', 2, 5)
    out.append('ORIGIN_MLP_NLAYERS=%d'%MLP_N_layers)

    globals_noise = trial.suggest_float('globals_noise', 0.0, 30.0)
    out.append('GLOBALS_NOISE=%.8e'%globals_noise)

    mass_noise = trial.suggest_float('mass_noise', 0.0, 30.0)
    out.append('MASS_NOISE=%.8e'%mass_noise)

    basis_noise = trial.suggest_float('basis_noise', 0.0, 3.0)
    out.append('BASIS_NOISE=%.8e'%basis_noise)

    return out
