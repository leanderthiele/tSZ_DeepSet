
def generate_cl(trial) :

    out = []
    out.append('NET_ARCH=dict(origin=True,batt12=True,deformer=True,encoder=False,'\
                             'scalarencoder=False,decoder=False,vae=False,local=False)')

    out.append('EPOCHS=70')

    out.append('PRT_FRACTION["DM"]["validation"]=2048')
    out.append('PRT_FRACTION["TNG"]["validation"]=8192')
    out.append('VALIDATION_EPOCHS=6')

    # from previous optuna runs
    out.append('ORIGIN_DEFAULT_NLAYERS=0')
    out.append('ORIGIN_MLP_NLAYERS=3')
    out.append('ORIGIN_MLP_NHIDDEN=226')
    out.append('ORIGIN_DEFAULT_NHIDDEN=226')
    out.append('PRT_FRACTION["DM"]["training"]=1024')

    # restrict parameter space a bit
    out.append('NET_ID=[("optuna_origin64_nr1017","origin","batt12"),]')
    out.append('NET_FREEZE=["origin"]')
    out.append('DEFORMER_NHIDDEN=196')
    out.append('DEFORMER_NLAYERS=4')
    out.append('DEFORMER_GLOBALS_PASSED=True')

    lr_deformer = trial.suggest_float('lr_deformer', 1e-6, 3e-5, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["deformer"]["max_lr"]=%.8e'%lr_deformer)

    globals_noise = trial.suggest_float('globals_noise', 5, 40)
    out.append('GLOBALS_NOISE=%.8e'%globals_noise)

    mass_noise = trial.suggest_float('mass_noise', 5, 30)
    out.append('MASS_NOISE=%.8e'%mass_noise)

    basis_noise = trial.suggest_float('basis_noise', 0, 20)
    out.append('BASIS_NOISE=%.8e'%basis_noise)

    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    out.append('DEFORMER_DROPOUT=%.8e'%dropout)

    visible_dropout = trial.suggest_float('visible_dropout', 0.3, 0.8)
    out.append('DEFORMER_VISIBLE_DROPOUT=%.8e'%visible_dropout)

    return out
