
def generate_cl(trial) :

    out = []
    out.append('NET_ARCH=dict(origin=False,batt12=False,deformer=False,encoder=False,'\
                             'scalarencoder=False,decoder=True,vae=False,local=True)')

    out.append('EPOCHS=200')

    # constraints on the architecture
    out.append('OUTPUT_NFEATURES=1')
    out.append('DECODER_DEFAULT_NLAYERS=1')
    out.append('DECODER_DEFAULT_R_PASSED=False')
    out.append('DECODER_DEFAULT_GLOBALS_PASSED=False')
    out.append('DECODER_DEFAULT_BASIS_PASSED=False')
    out.append('WEIGHT_DECAY["decoder"]=0')

    out.append('VALIDATION_EPOCHS=10')
    out.append('N_LOCAL["validation"]=512')
    out.append('PRT_FRACTION["TNG"]["validation"]=4096')

    # robust predictions we have from previous optuna run
    out.append('LOCAL_NLAYERS=2')
    out.append('LOCAL_PASS_N=False')
    out.append('LOCAL_CONCAT_WITH_N=True')
    out.append('N_LOCAL["training"]=310')

    start_nr = trial.suggest_categorical('start_nr', (352, 265, 845, 413, 520))
    out.append('NET_ID="optuna_local64_nr%d"'%start_nr)

    R_local = {352: 309.0541588966063,
               265: 340.9711356248012,
               845: 315.002388263229,
               413: 348.95538065659673,
               520: 316.2715842258252}

    MLP_N_layers = {352: 2, 265: 2, 845: 2, 413: 2, 520: 2}

    N_hidden = {352: 189, 265: 161, 845: 196, 413: 205, 520: 222}

    out.append('R_LOCAL=%.8e'%R_local[start_nr])

    out.append('LOCAL_MLP_NLAYERS=%d'%MLP_N_layers[start_nr])

    out.append('LOCAL_MLP_NHIDDEN=%d'%N_hidden[start_nr])
    out.append('LOCAL_NHIDDEN=%d'%N_hidden[start_nr])
    out.append('LOCAL_NLATENT=%d'%N_hidden[start_nr])
    out.append('DECODER_DEFAULT_NHIDDEN=%d'%N_hidden[start_nr])

    lr = trial.suggest_float('lr', 1e-7, 1e-3, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["local"]["max_lr"]=%.8e'%lr)
    out.append('ONE_CYCLE_LR_KWARGS["decoder"]["max_lr"]=%.8e'%lr)

    gradient_clip = trial.suggest_float('gradient_clip', 1e-2, 1e1, log=True)
    out.append('GRADIENT_CLIP=%.8e'%gradient_clip)

    return out
