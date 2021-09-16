
def generate_cl(trial) :
    
    out = []
    out.append('NET_ARCH=dict(origin=True,batt12=True,deformer=False,encoder=False,'\
                             'scalarencoder=False,decoder=True,vae=False,local=False)')

    out.append('VALIDATION_EPOCHS=6')
    out.append('PRT_FRACTION["DM"]["training"]=512')
    out.append('PRT_FRACTION["DM"]["validation"]=2048')
    out.append('PRT_FRACTION["TNG"]["validation"]=4096')

    out.append('ORIGIN_DEFAULT_NLAYERS=0')
    out.append('ORIGIN_MLP_NLAYERS=3')
    out.append('ORIGIN_MLP_NHIDDEN=226')
    out.append('ORIGIN_DEFAULT_NHIDDEN=226')
    out.append('BASIS_NOISE=None')

    out.append('NET_ID=("optuna_origin64_nr1017","batt12","origin")')
    out.append('NET_FREEZE=["origin","batt12"]')

    out.append('OUTPUT_NFEATURES=2')
    out.append('DECODER_DEFAULT_R_PASSED=True')
    out.append('DECODER_DEFAULT_GLOBALS_PASSED=True')
    out.append('DECODER_DEFAULT_BASIS_PASSED=False')

    out.append('DECODER_DEFAULT_NLAYERS=4')
    out.append('DECODER_DEFAULT_NHIDDEN=64')

    lr = trial.suggest_float('lr', 1e-6, 1e-2, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["decoder"]["max_lr"]=%.8e'%lr)

    wd = trial.suggest_float('wd', 1e-6, 1e-1, log=True)
    out.append('WEIGHT_DECAY["decoder"]=%.8e'%wd)

    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    out.append('DECODER_DROPOUT=%.8e'%dropout)

    visible_dropout = trial.suggest_float('visible_dropout', 0.0, 0.5)
    out.append('DECODER_VISIBLE_DROPOUT=%.8e'%visible_dropout)

    globals_noise = trial.suggest_float('globals_noise', 10.0, 40.0)
    out.append('GLOBALS_NOISE=%.8e'%globals_noise)

    mass_noise = trial.suggest_float('mass_noise', 0.0, 20.0)
    out.append('MASS_NOISE=%.8e'%mass_noise)

    gradient_clip = trial.suggest_float('gradient_clip', 1e-1, 1e1, log=True)
    out.append('GRADIENT_CLIP=%.8e'%gradient_clip)

    return out
