
def generate_cl(trial) :

    out = []
    out.append('NET_ARCH=dict(origin=True,batt12=True,deformer=False,encoder=False,'\
                             'scalarencoder=False,decoder=True,vae=False,local=False)')

    # the best-fit architecture from the origin run
    out.append('ORIGIN_DEFAULT_NLAYERS=0')
    out.append('ORIGIN_MLP_NLAYERS=4')
    out.append('PRT_FRACTION["DM"]["training"]=113')
    out.append('PRT_FRACTION["DM"]["validation"]=113')
    out.append('NET_ID="optuna_origin_nr862"')

    # spherically symmetric profile as a function of scalar halo quantities
    out.append('DECODER_DEFAULT_R_PASSED=True')
    out.append('DECODER_DEFAULT_GLOBALS_PASSED=True')
    out.append('DECODER_DEFAULT_BASIS_PASSED=False')

    fix_origin = trial.suggest_categorical('fix_origin', (True, False))
    if fix_origin :
        out.append('NET_FREEZE=["origin"]')
    else :
        out.append('ONE_CYCLE_LR_KWARGS["origin"]["max_lr"]=3.06e-4')
        out.append('WEIGHT_DECAY["origin"]=2.31e-4')

    lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["decoder"]["max_lr"]=%.8e'%lr)

    wd = trial.suggest_float('wd', 1e-6, 1e-2, log=True)
    out.append('WEIGHT_DECAY["decoder"]=%.8e'%wd)

    output_N_features = trial.suggest_categorical('output_N_features', (1, 2))
    out.append('OUTPUT_NFEATURES=%d'%output_N_features)

    N_layers = trial.suggest_int('N_layers', 2, 8)
    out.append('DECODER_DEFAULT_NLAYERS=%d'%N_layers)

    N_hidden = trial.suggest_int('N_hidden', 32, 256)
    out.append('DECODER_DEFAULT_NHIDDEN=%d'%N_hidden)

    noise = trial.suggest_float('noise', 1e-1, 2e1, log=True)
    out.append('GLOBALS_NOISE=%.8e'%noise)

    return out
