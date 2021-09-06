
def generate_cl(trial) :
    
    out = []
    out.append('NET_ARCH=dict(origin=True,batt12=True,deformer=False,encoder=False,'\
                             'scalarencoder=False,decoder=True,vae=False,local=True)')

    # the already trained architectures
    out.append('PRT_FRACTION["DM"]["training"]=113')
    out.append('PRT_FRACTION["DM"]["validation"]=113')
    out.append('ORIGIN_DEFAULT_NLAYERS=0')
    out.append('ORIGIN_MLP_NLAYERS=4')

    out.append('N_LOCAL=446')
    out.append('R_LOCAL=192.8')
    out.append('LOCAL_NLAYERS=2')
    out.append('LOCAL_MLP_NLAYERS=3')
    out.append('LOCAL_MLP_NHIDDEN=190')
    out.append('LOCAL_MLP_NLATENT=190')
    out.append('LOCAL_PASS_N=False')
    out.append('LOCAL_CONCAT_WITH_N=True')
    out.append('LOCAL_NHIDDEN=190')

    out.append('NET_ID=[("optuna_origin_nr862", "origin"), ("optuna_localdecoder_nr280", "local")]')
    out.append('NET_FREEZE=["origin", "local"]')

    out.append('DECODER_DEFAULT_R_PASSED=True')
    out.append('DECODER_DEFAULT_GLOBALS_PASSED=True')
    out.append('DECODER_DEFAULT_BASIS_PASSED=False')

    lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["decoder"]["max_lr"]=%.8e'%lr)

    wd = trial.suggest_float('wd', 1e-7, 1e-1, log=True)
    out.append('WEIGHT_DECAY["decoder"]=%.8e'%wd)

    output_N_features = trial.suggest_categorical('output_N_features', (1, 2))
    out.append('OUTPUT_NFEATURES=%d'%output_N_features)

    N_layers = trial.suggest_int('N_layers', 2, 8)
    out.append('DECODER_DEFAULT_NLAYERS=%d'%N_layers)

    N_hidden = trial.suggest_int('N_hidden', 32, 256)
    out.append('DECODER_DEFAULT_NHIDDEN=%d'%N_hidden)

    noise = trial.suggest_float('noise', 1e-1, 2e1, log=True)
    out.append('GLOBALS_NOISE=%.8e'%noise)

    dropout = trial.suggest_float('dropout', 0.0, 0.9)
    out.append('DECODER_DROPOUT=%.8e'%dropout)

    visible_dropout = trial.suggest_float('visible_dropout', 0.0, 0.9)
    out.append('DECODER_VISIBLE_DROPOUT=%.8e'%visible_dropout)

    return out
