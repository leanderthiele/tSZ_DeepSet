
def generate_cl(trial) :

    out = []
    out.append('NET_ARCH=dict(origin=True,batt12=True,deformer=False,encoder=False,'\
                             'scalarencoder=False,decoder=True,vae=True,local=True)')

    # constraints on the architecture
    out.append('OUTPUT_NFEATURES=2')
    out.append('DECODER_DEFAULT_R_PASSED=True')
    out.append('DECODER_DEFAULT_GLOBALS_PASSED=True')
    out.append('DECODER_DEFAULT_BASIS_PASSED=False')

    out.append('VALIDATION_EPOCHS=10')
    out.append('N_LOCAL["validation"]=512')
    out.append('PRT_FRACTION["DM"]["validation"]=2048')
    out.append('PRT_FRACTION["TNG"]["validation"]=4096')

    out.append('N_GAUSS=5')

    # robust predictions we have from previous optuna runs
    out.append('LOCAL_NLAYERS=2')
    out.append('LOCAL_PASS_N=False')
    out.append('LOCAL_CONCAT_WITH_N=True')
    out.append('LOCAL_MLP_NLAYERS=2')
    out.append('LOCAL_MLP_NHIDDEN=189')
    out.append('LOCAL_NHIDDEN=189')
    out.append('LOCAL_NLATENT=189')
    out.append('ORIGIN_DEFAULT_NLAYERS=0')
    out.append('ORIGIN_MLP_NLAYERS=3')
    out.append('ORIGIN_MLP_NHIDDEN=226')
    out.append('ORIGIN_DEFAULT_NHIDDEN=226')
    out.append('BASIS_NOISE=None')
    out.append('PRT_FRACTION["DM"]["training"]=2048')
    out.append('N_LOCAL["training"]=256')
    out.append('R_LOCAL=310.0')

    # things we can play with later but want to fix at the moment to focus on the VAE stuff
    out.append('DECODER_DEFAULT_NLAYERS=3')
    out.append('DECODER_DEFAULT_NHIDDEN=196')
    out.append('DECODER_DROPOUT=0.1')
    out.append('DECODER_VISIBLE_DROPOUT=0.1')
    out.append('GLOBALS_NOISE=30.0')
    out.append('WEIGHT_DECAY["decoder"]=1e-5')

    out.append('NET_ID=("optuna_localorigin64_nr561","local","origin","batt12")')
    out.append('NET_FREEZE=["local","origin","batt12"]')

    lr_decoder = trial.suggest_float('lr_decoder', 1e-6, 1e-1, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["decoder"]["max_lr"]=%.8e'%lr_decoder)

    lr_vae = trial.suggest_float('lr_vae', 1e-7, 1e-1, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["vae"]["max_lr"]=%.8e'%lr_vae)

    wd_vae = trial.suggest_float('wd_vae', 1e-8, 1e-1, log=True)
    out.append('WEIGHT_DECAY["vae"]=%.8e'%wd_vae)

    N_layers_vae = trial.suggest_int('N_layers_vae', 2, 5)
    out.append('VAE_NLAYERS=%d'%N_layers_vae)

    N_hidden_vae = trial.suggest_int('N_hidden_vae', 16, 256)
    out.append('VAE_NHIDDEN=%d'%N_hidden_vae)

    residuals_noise = trial.suggest_float('residuals_noise', 1e-2, 1e1, log=True)
    out.append('RESIDUALS_NOISE=%.8e'%residuals_noise)

    kld_scaling = trial.suggest_float('kld_scaling', 1e-6, 1e0, log=True)
    out.append('KLD_SCALING=%.8e'%kld_scaling)

    kld_annealing_start = trial.suggest_float('kld_annealing_start', 0.0, 0.3)
    out.append('KLD_ANNEALING_START=%.8e'%kld_annealing_start)
    
    kld_annealing_end = trial.suggest_float('kld_annealing_end', 0.3, 1.0)
    out.append('KLD_ANNEALING_END=%.8e'%kld_annealing_end)

    gradient_clip = trial.suggest_float('gradient_clip', 1e-1, 1e1)
    out.append('GRADIENT_CLIP=%.8e'%gradient_clip)

    N_latent_vae = trial.suggest_categorical('N_latent_vae', (1, 2))
    out.append('VAE_NLATENT=%d'%N_latent_vae)


    return out
