
def generate_cl(trial) :

    out = []
    out.append('NET_ARCH=dict(origin=False,batt12=False,deformer=False,encoder=False,'\
                             'scalarencoder=False,decoder=False,vae=False,local=True)')
    out.append('LOCAL_NLATENT=1')
    out.append('OUTPUT_NFEATURES=1')
    out.append('LOCAL_NLAYERS=1') # required because there's no decoder
    # FIXME this should not matter!
    out.append('SCALE_PTH=False')

    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["local"]["max_lr"]=%.8e'%lr)

    # TODO increase maximum value later, this is only for debugging
    N_local = trial.suggest_int('N_local', 32, 256, log=True)
    out.append('N_LOCAL=%d'%N_local)

    R_local = trial.suggest_float('R_local', 50.0, 200.0)
    out.append('R_LOCAL=%.8e'%R_local)

    MLP_N_layers = trial.suggest_int('MLP_N_layers', 2, 8)
    out.append('LOCAL_MLP_NLAYERS=%d'%MLP_N_layers)

    MLP_N_hidden = trial.suggest_int('MLP_N_hidden', 32, 256)
    out.append('LOCAL_MLP_NHIDDEN=%d'%MLP_N_hidden)

    pass_N = trial.suggest_categorical('pass_N', ('True', 'False'))
    out.append('LOCAL_PASS_N=%s'%pass_N)

    layernorm = trial.suggest_categorical('layernorm', ('True', 'False'))
    out.append('LAYERNORM=%s'%layernorm)

    return out
