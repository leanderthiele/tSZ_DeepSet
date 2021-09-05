
def generate_cl(trial) :

    out = []
    out.append('NET_ARCH=dict(origin=True,batt12=True,deformer=False,encoder=False,'\
                             'scalarencoder=False,decoder=True,vae=False,local=False)')

    out.append('NET_ID="optuna_origin_nr862"')
    out.append('DECODER_DEFAULT_R_PASSED=True')
    out.append('DECODER_DEFAULT_BASIS_PASSED=False')

    fix_origin = trial.suggest_categorical('fix_origin', (True, False))
    if fix_origin :
        out.append('NET_FREEZE=["origin"]')
    else :
        out.append('ONE_CYCLE_LR_KWARGS["origin"]["max_lr"]=3.06e-4')
        out.append('WEIGHT_DECAY["origin"]=2.31e-4')

    output_N_features = trial.suggest_categorical('output_N_features', (1, 2))
    out.append('OUTPUT_NFEATURES=%d'%output_N_features)

    N_layers = trial.suggest_int('N_layers', 2, 8)
    out.append('DECODER_DEFAULT_NLAYERS=%d'%N_layers)
