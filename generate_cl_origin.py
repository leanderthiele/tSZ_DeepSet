
def generate_cl(trial) :

    out = []
    out.append('NET_ARCH=dict(origin=True,batt12=True,deformer=False,encoder=False,'\
                             'scalarencoder=False,decoder=False,vae=False,local=False)')

    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["origin"]["max_lr"]=%.8e'%lr)

    wd = trial.suggest_float('wd', 1e-6, 1e-2, log=True)
    out.append('WEIGHT_DECAY["origin"]=%.8e'%wd)

    N_DM = trial.suggest_int('N_DM', int(1e2), int(1e5), log=True)
    out.append('PRT_FRACTION["DM"]=%d'%N_DM)

    N_layers = trial.suggest_int('N_layers', 0, 5)
    out.append('ORIGIN_DEFAULT_NLAYERS=%d'%N_layers)

    MLP_N_layers = trial.suggest_int('MLP_N_layers', 2, 8)
    out.append('ORIGIN_MLP_NLAYERS=%d'%MLP_N_layers)

    return out
