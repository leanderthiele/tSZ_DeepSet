
def generate_cl(trial) :
    
    out = []
    out.append('NET_ARCH=dict(origin=False,batt12=True,deformer=False,encoder=True,'\
                             'scalarencoder=True,decoder=True,vae=False,local=False)')

    out.append('ENCODER_MLP_NLAYERS=3')
    out.append('ENCODER_DEFAULT_NLAYERS=0')
    out.append('ENCODER_DEFAULT_GLOBALS_MAXLAYER=0')
    out.append('SCALAR_ENCODER_NLAYERS=1')
    out.append('SCALAR_ENCODER_MLP_NLAYERS=2')
    out.append('SCALAR_ENCODER_GLOBALS_PASSED=False')
    out.append('NET_ID=[("optuna_encoder_nr1056","encoder"),("optuna_scalarencoder_nr1111","scalarencoder")]')
    out.append('NET_FREEZE=["encoder", "scalarencoder"]')

    out.append('PRT_FRACTION["DM"]["training"]=1024')
    out.append('PRT_FRACTION["DM"]["validation"]=1024')
    
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["decoder"]["max_lr"]=%.8e'%lr)

    wd = trial.suggest_float('wd', 1e-6, 1e-2, log=True)
    out.append('WEIGHT_DECAY["decoder"]=%.8e'%wd)

    N_layers = trial.suggest_int('N_layers', 2, 8)
    out.append('DECODER_DEFAULT_NLAYERS=%d'%N_layers)

    N_hidden = trial.suggest_int('N_hidden', 32, 256)
    out.append('DECODER_DEFAULT_NHIDDEN=%d'%N_hidden)

    basis_passed= trial.suggest_categorical('basis_passed', ('True', 'False'))
    out.append('DECODER_DEFAULT_BASIS_PASSED=%s'%basis_passed)

    globals_passed = trial.suggest_categorical('globals_passed', ('True', 'False'))
    out.append('DECODER_DEFAULT_GLOBALS_PASSED=%s'%globals_passed)

    R_passed = trial.suggest_categorical('R_passed', ('True', 'False'))
    out.append('DECODER_DEFAULT_R_PASSED=%s'%R_passed)

    globals_noise = trial.suggest_float('globals_noise', 1e-1, 2e1, log=True)
    out.append('GLOBALS_NOISE=%.8e'%globals_noise)

    basis_noise = trial.suggest_float('basis_noise', 1e-2, 1e1, log=True)
    out.append('BASIS_NOISE=%.8e'%basis_noise)
    
    return out
