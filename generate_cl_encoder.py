
def generate_cl(trial) :

    out = []
    out.append('NET_ARCH=dict(origin=False,batt12=True,deformer=False,encoder=True,'\
                             'scalarencoder=False,decoder=True,vae=False,local=False)')

    lr_encoder = trial.suggest_float('lr_encoder', 1e-6, 1e-1, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["encoder"]["max_lr"]=%.8e'%lr_encoder)

    wd_encoder = trial.suggest_float('wd_encoder', 1e-7, 1e-2, log=True)
    out.append('WEIGHT_DECAY["encoder"]=%.8e'%wd_encoder)

    lr_decoder = trial.suggest_float('lr_decoder', 1e-6, 1e-1, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["decoder"]["max_lr"]=%.8e'%lr_decoder)

    wd_decoder = trial.suggest_float('wd_decoder', 1e-7, 1e-2, log=True)
    out.append('WEIGHT_DECAY["decoder"]=%.8e'%wd_decoder)

    N_DM = trial.suggest_int('N_DM', int(1e2), int(1e4), log=True)
    out.append('PRT_FRACTION["DM"]=%d'%N_DM)

    N_layers_encoder = trial.suggest_int('N_layers_encoder', 0, 5)
    out.append('ENCODER_DEFAULT_NLAYERS=%d'%N_layers_encoder)

    MLP_N_layers_encoder = trial.suggest_int('MLP_N_layers_encoder', 2, 4)
    out.append('ENCODER_MLP_NLAYERS=%d'%MLP_N_layers_encoder)

    N_layers_decoder = trial.suggest_int('N_layers_decoder', 2, 8)
    out.append('DECODER_DEFAULT_NLAYERS=%d'%N_layers_decoder)

    globals_maxlayer = trial.suggest_categorical('globals_maxlayer', (-1, 0))
    out.append('ENCODER_DEFAULT_GLOBALS_MAXLAYER=%d'%globals_maxlayer)

    decoder_basis = trial.suggest_categorical('decoder_basis', ('True', 'False'))
    out.append('DECODER_DEFAULT_BASIS_PASSED=%s'%decoder_basis)

    return out
