
def generate_cl(trial) :
    
    out = []
    out.append('NET_ARCH=dict(origin=True,batt12=True,deformer=False,encoder=False,'\
                             'scalarencoder=False,decoder=True,vae=False,local=True)')

    # constraints on the architecture
    out.append('OUTPUT_NFEATURES=2')
    out.append('DECODER_DEFAULT_R_PASSED=True')
    out.append('DECODER_DEFAULT_GLOBALS_PASSED=True')
    out.append('DECODER_DEFAULT_BASIS_PASSED=False')
    out.append('WEIGHT_DECAY["decoder"]=0')

    out.append('VALIDATION_EPOCHS=10')
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
    out.append('N_LOCAL["training"]=310')
    out.append('R_LOCAL=310.0')

    lr_local = trial.suggest_float('lr_local', 1e-7, 1e-3, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["local"]["max_lr"]=%.8e'%lr_local)

    lr_origin = trial.suggest_float('lr_origin', 1e-6, 5e-4, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["origin"]["max_lr"]=%.8e'%lr_origin)

    lr_decoder = trial.suggest_float('lr_decoder', 1e-6, 5e-3, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["decoder"]["max_lr"]=%.8e'%lr_decoder)

    wd_origin = trial.suggest_float('wd_origin', 1e-3, 1e0, log=True)
    out.append('WEIGHT_DECAY["origin"]=%.8e'%wd_origin)

    wd_decoder = trial.suggest_float('wd_decoder', 1e-7, 1e-3, log=True)
    out.append('WEIGHT_DECAY["decoder"]=%.8e'%wd_decoder)

    N_layers = trial.suggest_int('N_layers', 2, 5)
    out.append('DECODER_DEFAULT_NLAYERS=%d'%N_layers)

    N_hidden = trial.suggest_int('N_hidden', 32, 256)
    out.append('DECODER_DEFAULT_NHIDDEN=%d'%N_hidden)

    gradient_clip = trial.suggest_float('gradient_clip', 1e-2, 1e0, log=True)
    out.append('GRADIENT_CLIP=%.8e'%gradient_clip)

    globals_noise = trial.suggest_float('globals_noise', 5, 30)
    out.append('GLOBALS_NOISE=%.8e'%globals_noise)

    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    out.append('DECODER_DROPOUT=%.8e'%dropout)

    visible_dropout = trial.suggest_float('visible_dropout', 0.0, 0.3)
    out.append('DECODER_VISIBLE_DROPOUT=%.8e'%visible_dropout)

    origin_from_file = trial.suggest_categorical('origin_from_file', (True, False))
    local_from_file = trial.suggest_categorical('local_from_file', (True, False))

    if origin_from_file and local_from_file :
        out.append('NET_ID=[("optuna_origin64_nr1017","origin","batt12"),'\
                           '("optuna_local64_nr352","local"),]')
    elif origin_from_file and not local_from_file :
        out.append('NET_ID=[("optuna_origin64_nr1017","origin","batt12"),]')
    elif not origin_from_file and local_from_file :
        out.append('NET_ID=[("optuna_local64_nr352","local"),]')


    return out
