
def generate_cl(trial) :
    
    out = []
    out.append('NET_ARCH=dict(origin=False,batt12=False,deformer=False,encoder=False,'\
                             'scalarencoder=False,decoder=True,vae=False,local=True)')

    # constraints on the architecture
    out.append('OUTPUT_NFEATURES=1')
    out.append('DECODER_DEFAULT_NLAYERS=1')
    out.append('DECODER_DEFAULT_R_PASSED=False')
    out.append('DECODER_DEFAULT_GLOBALS_PASSED=False')
    out.append('DECODER_DEFAULT_BASIS_PASSED=False')
    out.append('WEIGHT_DECAY["decoder"]=0')

    out.append('VALIDATION_EPOCHS=10')
    out.append('N_LOCAL["validation"]=512')
    out.append('PRT_FRACTION["TNG"]["validation"]=2048')

    # robust predictions we have from previous optuna run
    out.append('LOCAL_NLAYERS=2')
    out.append('LOCAL_PASS_N=False')
    out.append('LOCAL_CONCAT_WITH_N=True')

    lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["local"]["max_lr"]=%.8e'%lr)
    out.append('ONE_CYCLE_LR_KWARGS["decoder"]["max_lr"]=%.8e'%lr)

    N_local = trial.suggest_int('N_local', 100, 500)
    out.append('N_LOCAL["training"]=%d'%N_local)

    R_local = trial.suggest_float('R_local', 100.0, 400.0)
    out.append('R_LOCAL=%.8e'%R_local)

    MLP_N_layers = trial.suggest_int('MLP_N_layers', 2, 5)
    out.append('LOCAL_MLP_NLAYERS=%d'%MLP_N_layers)

    N_hidden = trial.suggest_int('N_hidden', 100, 230)
    out.append('LOCAL_MLP_NHIDDEN=%d'%N_hidden)
    out.append('LOCAL_NHIDDEN=%d'%N_hidden)
    out.append('LOCAL_NLATENT=%d'%N_hidden)
    out.append('DECODER_DEFAULT_NHIDDEN=%d'%N_hidden)

    # let's also try this hyperparameter
    gradient_clip = trial.suggest_float('gradient_clip', 1e-2, 1e2, log=True)
    out.append('GRADIENT_CLIP=%.8e'%gradient_clip)

    return out
