
def generate_cl(trial) :

    out = []
    out.append('NET_ARCH=dict(origin=True,batt12=True,deformer=True,encoder=False,'\
                             'scalarencoder=False,decoder=True,vae=False,local=True)')

    out.append('VALIDATION_EPOCHS=10')

    out.append('PRT_FRACTION["DM"]["validation"]=2048')
    out.append('PRT_FRACTION["TNG"]["validation"]=2048')
    out.append('PRT_FRACTION["DM"]["training"]=256')
    out.append('N_LOCAL["validation"]=1024')

    out.append('ORIGIN_DEFAULT_NLAYERS=0')
    out.append('ORIGIN_MLP_NLAYERS=4')
    out.append('DEFORMER_NLAYERS=2')
    out.append('DEFORMER_NHIDDEN=181')
    out.append('DEFORMER_GLOBALS_PASSED=False')
    out.append('LOCAL_NLAYERS=2')
    out.append('LOCAL_MLP_NLAYERS=3')
    out.append('LOCAL_MLP_NHIDDEN=190')
    out.append('LOCAL_NLATENT=190')
    out.append('LOCAL_NHIDDEN=190')
    out.append('LOCAL_PASS_N=False')
    out.append('LOCAL_CONCAT_WITH_N=True')
    out.append('DECODER_DEFAULT_R_PASSED=True')
    out.append('DECODER_DEFAULT_GLOBALS_PASSED=True')
    out.append('DECODER_DEFAULT_BASIS_PASSED=False')
    out.append('DECODER_DEFAULT_NHIDDEN=170')
    out.append('OUTPUT_NFEATURES=2')
    out.append('BASIS_NOISE=None')

    out.append('NET_ID=[("optuna_origin_nr862","origin","batt12"),'\
                       '("optuna_localdecoderorigin_nr1004","local"),'\
                       '("optuna_origindeformer_nr656","deformer")]')
    out.append('NET_FREEZE=["origin","deformer"]')

    lr_local = trial.suggest_float('lr_local', 1e-6, 1e-3, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["local"]["max_lr"]=%.8e'%lr_local)

    lr_decoder = trial.suggest_float('lr_decoder', 1e-5, 1e-1, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["decoder"]["max_lr"]=%.8e'%lr_decoder)

    wd = trial.suggest_float('wd', 1e-7, 1e-3, log=True)
    out.append('WEIGHT_DECAY["decoder"]=%.8e'%wd)

    N_layers = trial.suggest_categorical('N_layers', (2, 3))
    out.append('DECODER_DEFAULT_NLAYERS=%d'%N_layers)

    R_local = trial.suggest_float('R_local', 150.0, 400.0)
    out.append('R_LOCAL=%.8e'%R_local)

    noise = trial.suggest_float('noise', 5, 30)
    out.append('GLOBALS_NOISE=%.8e'%noise)

    dropout = trial.suggest_float('dropout', 0.0, 0.3)
    out.append('DECODER_DROPOUT=%.8e'%dropout)

    visible_dropout = trial.suggest_float('visible_dropout', 0.0, 0.3)
    out.append('DECODER_VISIBLE_DROPOUT=%.8e'%visible_dropout)

    N_local = trial.suggest_int('N_local', 128, 512, log=True)
    out.append('N_LOCAL["training"]=%d'%N_local)

    return out
