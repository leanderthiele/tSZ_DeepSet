
def generate_cl(trial) :

    out = []
    out.append('NET_ARCH=dict(origin=True,batt12=True,deformer=True,encoder=False,'\
                             'scalarencoder=False,decoder=False,vae=False,local=False)')

    out.append('PRT_FRACTION["DM"]["validation"]=2048')
    out.append('PRT_FRACTION["TNG"]["validation"]=8192')
    out.append('VALIDATION_EPOCHS=10')

    # from previous optuna runs
    out.append('ORIGIN_DEFAULT_NLAYERS=0')
    out.append('ORIGIN_MLP_NLAYERS=3')
    out.append('ORIGIN_MLP_NHIDDEN=226')
    out.append('ORIGIN_DEFAULT_NHIDDEN=226')
    out.append('PRT_FRACTION["DM"]["training"]=2048')

    lr_origin = trial.suggest_float('lr_origin', 1e-6, 5e-4, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["origin"]["max_lr"]=%.8e'%lr_origin)

    lr_deformer = trial.suggest_float('lr_deformer', 1e-6, 1e-3, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["deformer"]["max_lr"]=%.8e'%lr_deformer)

    globals_passed = trial.suggest_categorical('globals_passed', ('True', 'False'))
    out.append('DEFORMER_GLOBALS_PASSED=%s'%globals_passed)

    globals_noise = trial.suggest_float('globals_noise', 1, 30)
    out.append('GLOBALS_NOISE=%.8e'%globals_noise)

    basis_noise = trial.suggest_float('basis_noise', 0, 10)
    out.append('BASIS_NOISE=%.8e'%basis_noise)

    N_layers = trial.suggest_int('N_layers', 2, 5)
    out.append('DEFORMER_NLAYERS=%d'%N_layers)

    N_hidden = trial.suggest_int('N_hidden', 64, 256)
    out.append('DEFORMER_NHIDDEN=%d'%N_hidden)

    dropout = trial.suggest_categorical('dropout', 0.0, 0.8)
    out.append('DEFORMER_DROPOUT=%.8e'%dropout)

    visible_dropout = trial.suggest_categorical('visible_dropout', 0.0, 0.8)
    out.append('DEFORMER_VISIBLE_DROPOUT=%.8e'%visible_dropout)

    origin_from_file = trial.suggest_categorical('origin_from_file', (True, False))
    if origin_from_file :
        out.append('NET_ID=[("optuna_origin64_nr1017","origin","batt12"),]')

    return out
