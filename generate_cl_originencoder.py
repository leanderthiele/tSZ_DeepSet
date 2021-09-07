
def generate_cl(trial) :

    out = []
    out.append('NET_ARCH=dict(origin=True,batt12=True,deformer=False,encoder=True,'\
                             'scalarencoder=False,decoder=True,vae=False,local=False)')

    out.append('ORIGIN_DEFAULT_NLAYERS=0')
    out.append('ORIGIN_MLP_NLAYERS=4')
    out.append('ENCODER_DEFAULT_NLAYERS=0')
    out.append('ENCODER_MLP_NLAYERS=3')
    out.append('ENCODER_DEFAULT_GLOBALS_MAXLAYER=0')
    out.append('NET_ID=[("optuna_encoder_nr1056","encoder"),("optuna_origin_nr862","origin")]')
    out.append('PRT_FRACTION["DM"]["training"]=1024')
    out.append('PRT_FRACTION["DM"]["validation"]=1024')

    out.append('DECODER_DEFAULT_BASIS_PASSED=True')
    out.append('DECODER_DEFAULT_GLOBALS_PASSED=True')
    out.append('DECODER_DEFAULT_R_PASSED=True')

    encoder_fixed = trial.suggest_categorical('encoder_fixed', (True, False))
    if encoder_fixed :
        out.append('NET_FREEZE=["origin","encoder"]')
    else :
        out.append('NET_FREEZE=["origin",]')
        out.append('ONE_CYCLE_LR_KWARGS["encoder"]["max_lr"]=5.66e-2')
        out.append('WEIGHT_DECAY["encoder"]=3.49e-5')

    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["decoder"]["max_lr"]=%.8e'%lr)

    wd = trial.suggest_float('wd', 1e-7, 1e-4, log=True)
    out.append('WEIGHT_DECAY["decoder"]=%.8e'%wd)

    N_layers = trial.suggest_categorical('N_layers', (5,6,7))
    out.append('DECODER_DEFAULT_NLAYERS=%d'%N_layers)

    N_hidden = trial.suggest_int('N_hidden', 32, 256)
    out.append('DECODER_DEFAULT_NHIDDEN=%d'%N_hidden)

    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    out.append('DECODER_DROPOUT=%.8f'%dropout)

    visible_dropout = trial.suggest_float('visible_dropout', 0.0, 0.5)
    out.append('DECODER_VISIBLE_DROPOUT=%.8f'%visible_dropout)

    globals_noise = trial.suggest_float('globals_noise', 1e0, 3e1, log=True)
    out.append('GLOBALS_NOISE=%.8f'%globals_noise)

    basis_noise = trial.suggest_float('basis_noise', 1e-2, 3e1, log=True)
    out.append('BASIS_NOISE=%.8f'%basis_noise)

    output_N_features = trial.suggest_categorical('output_N_features', (1, 2))
    out.append('OUTPUT_N_FEATURES=%d'%output_N_features)


    return out
