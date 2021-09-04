
def generate_cl(trial) :

    out = []
    out.append('NET_ARCH=dict(origin=False,batt12=False,deformer=False,encoder=False,'\
                             'scalarencoder=False,decoder=True,vae=False,local=True)')
    out.append('OUTPUT_NFEATURES=1')
    out.append('DECODER_DEFAULT_NLAYERS=1')
    out.append('DECODER_DEFAULT_R_PASSED=False')
    out.append('DECODER_DEFAULT_GLOBALS_PASSED=False')
    out.append('DECODER_DEFAULT_BASIS_PASSED=False')
    out.append('WEIGHT_DECAY["decoder"]=0')

    lr = trial.suggest_float('lr', 1e-6, 1e-4, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["local"]["max_lr"]=%.8e'%lr)
    out.append('ONE_CYCLE_LR_KWARGS["decoder"]["max_lr"]=%.8e'%lr)

    N_local = trial.suggest_int('N_local', 16, 512, log=True)
    out.append('N_LOCAL=%d'%N_local)

    R_local = trial.suggest_float('R_local', 50.0, 200.0)
    out.append('R_LOCAL=%.8e'%R_local)

    N_layers = trial.suggest_categorical('N_layers', (1,2,3))
    out.append('LOCAL_NLAYERS=%d'%N_layers)

    MLP_N_layers = trial.suggest_int('MLP_N_layers', 2, 8)
    out.append('LOCAL_MLP_NLAYERS=%d'%MLP_N_layers)

    N_hidden = trial.suggest_int('N_hidden', 32, 256)
    out.append('LOCAL_MLP_NHIDDEN=%d'%N_hidden)
    out.append('LOCAL_NHIDDEN=%d'%N_hidden)
    out.append('LOCAL_NLATENT=%d'%N_hidden)
    out.append('DECODER_DEFAULT_NHIDDEN=%d'%N_hidden)

    how_N = trial.suggest_categorical('how_N', ('both', 'pass', 'concat'))
    if how_N == 'both' :
        out.append('LOCAL_PASS_N=True')
        out.append('LOCAL_CONCAT_WITH_N=True')
    elif how_N == 'pass' :
        out.append('LOCAL_PASS_N=True')
        out.append('LOCAL_CONCAT_WITH_N=False')
    elif how_N == 'concat' :
        out.append('LOCAL_PASS_N=False')
        out.append('LOCAL_CONCAT_WITH_N=True')

    layernorm = trial.suggest_categorical('layernorm', ('True', 'False'))
    out.append('LAYERNORM=%s'%layernorm)

    return out
