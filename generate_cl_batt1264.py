
def generate_cl(trial) :

    out = []
    out.append('NET_ARCH=dict(origin=False,batt12=True,deformer=False,encoder=False,'\
                             'scalarencoder=False,decoder=False,vae=False,local=False)')

    out.append('PRT_FRACTION["TNG"]["validation"]=16384')
    out.append('VALIDATION_EPOCHS=10')

    lr = trial.suggest_float('lr', 1e-6, 1e-1, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["batt12"]["max_lr"]=%.8e'%lr)

    N_TNG = trial.suggest_int('N_TNG', 200, 2000, log=True)
    out.append('PRT_FRACTION["TNG"]["training"]=%d'%N_TNG)


    return out
