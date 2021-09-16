
def generate_cl(trial) :

    out = []
    out.append('NET_ARCH=dict(origin=True,batt12=True,deformer=False,encoder=False,'\
                             'scalarencoder=False,decoder=True,vae=False,local=True)')

    out.append('EPOCHS=200')

    out.append('OUTPUT_NFEATURES=2')
    out.append('DECODER_DEFAULT_R_PASSED=True')
    out.append('DECODER_DEFAULT_GLOBALS_PASSED=True')
    out.append('DECODER_DEFAULT_BASIS_PASSED=False')

    out.append('VALIDATION_EPOCHS=6')
    out.append('N_LOCAL["validation"]=512')
    out.append('PRT_FRACTION["DM"]["validation"]=2048')
    out.append('PRT_FRACTION["TNG"]["validation"]=4096')

    # robust predictions we have from previous optuna runs
    out.append('LOCAL_NLAYERS=2')
    out.append('LOCAL_PASS_N=False')
    out.append('LOCAL_CONCAT_WITH_N=True')
    out.append('LOCAL_MLP_NLAYERS=2')
    out.append('LOCAL_MLP_NHIDDEN=189')
    out.append('LOCAL_NHIDDEN=189')
    out.append('LOCAL_NLATENT=189')
    out.append('ORIGIN_DEFAULT_NLAYERS=0')
    out.append('ORIGIN_MLP_NLAYERS=3')
    out.append('ORIGIN_MLP_NHIDDEN=226')
    out.append('ORIGIN_DEFAULT_NHIDDEN=226')
    out.append('BASIS_NOISE=None')
    out.append('PRT_FRACTION["DM"]["training"]=2048')
    out.append('N_LOCAL["training"]=256')
    out.append('R_LOCAL=310.0')

    out.append('NET_ID=("optuna_localorigin64_nr426","local","origin","batt12")')
    out.append('NET_FREEZE=["local","origin","batt12"]')

    # we want the decoder to be fairly expressive because we believe that there are some prior volume
    # effects at play here
    out.append('DECODER_DEFAULT_NLAYERS=4')
    out.append('DECODER_DEFAULT_NHIDDEN=196')

    lr = trial.suggest_float('lr', 1e-6, 1e-3, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["decoder"]["max_lr"]=%.8e'%lr)

    wd = trial.suggest_float('wd', 1e-7, 1e-3, log=True)
    out.append('WEIGHT_DECAY["decoder"]=%.8e'%wd)

    dropout = trial.suggest_float('dropout', 0.0, 0.2)
    out.append('DECODER_DROPOUT=%.8e'%dropout)

    visible_dropout = trial.suggest_float('visible_dropout', 0.0, 0.5)
    out.append('DECODER_VISIBLE_DROPOUT=%.8e'%visible_dropout)

    gradient_clip = trial.suggest_float('gradient_clip', 1e-1, 1e1, log=True)
    out.append('GRADIENT_CLIP=%.8e'%gradient_clip)

    globals_noise = trial.suggest_float('globals_noise', 10.0, 40.0)
    out.append('GLOBALS_NOISE=%.8e'%globals_noise)

    mass_noise = trial.suggest_float('mass_noise', 6.0, 30.0)
    out.append('MASS_NOISE=%.8e'%mass_noise)


    return out
