
def generate_cl(trial) :
    
    out = []
    out.append('NET_ARCH=dict(origin=True,batt12=True,deformer=True,encoder=True,'\
                             'scalarencoder=False,decoder=True,vae=False,local=True)')

    out.append('PRT_FRACTION["DM"]["validation"]=2048')
    out.append('PRT_FRACTION["TNG"]["validation"]=2048')

    # completed networks
    out.append('ORIGIN_DEFAULT_NLAYERS=0')
    out.append('ORIGIN_MLP_NLAYERS=4')
    out.append('DEFORMER_NLAYERS=2')
    out.append('DEFORMER_NHIDDEN=181')
    out.append('DEFORMER_GLOBALS_PASSED=False')
    out.append('N_LOCAL=446')
    out.append('R_LOCAL=192.8')
    out.append('LOCAL_NLAYERS=2')
    out.append('LOCAL_MLP_NLAYERS=3')
    out.append('LOCAL_MLP_NHIDDEN=190')
    out.append('LOCAL_NLATENT=190')
    out.append('LOCAL_NHIDDEN=190')
    out.append('LOCAL_PASS_N=False')
    out.append('LOCAL_CONCAT_WITH_N=True')

    out.append('NET_ID=[("optuna_origin_nr862","origin","batt12"),'\
                       '("optuna_localdecoder_nr280","local"),'\
                       '("optuna_origindeformer_nr656","deformer")]')
    out.append('NET_FREEZE=["origin","deformer","local"]')

    out.append('DECODER_DEFAULT_R_PASSED=True')
    out.append('DECODER_DEFAULT_BASIS_PASSED=True')

    lr_encoder = trial.suggest_float('lr_encoder', 1e-4, 1e0, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["encoder"]["max_lr"]=%.8e'%lr_encoder)

    lr_decoder = trial.suggest_float('lr_decoder', 1e-5, 1e-1, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["decoder"]["max_lr"]=%.8e'%lr_decoder)

    wd_encoder = trial.suggest_float('wd_encoder', 1e-8, 1e-3, log=True)
    out.append('WEIGHT_DECAY["encoder"]=%.8e'%wd_encoder)

    wd_decoder = trial.suggest_float('wd_decoder', 1e-8, 1e-2, log=True)
    out.append('WEIGHT_DECAY["decoder"]=%.8e'%wd_decoder)

    encoder_globals_passed = trial.suggest_categorical('encoder_globals_passed', (True, False))
    out.append('ENCODER_DEFAULT_GLOBALS_MAXLAYER=%d'%(0 if encoder_globals_passed else -1))

    decoder_globals_passed = trial.suggest_categorical('decoder_globals_passed', ('True', 'False'))
    out.append('DECODER_DEFAULT_GLOBALS_PASSED=%s'%decoder_globals_passed)

    encoder_N_layers = trial.suggest_int('encoder_N_layers', 0, 2)
    out.append('ENCODER_DEFAULT_NLAYERS=%d'%encoder_N_layers)

    encoder_MLP_N_layers = trial.suggest_int('encoder_MLP_N_layers', 2, 6)
    out.append('ENCODER_MLP_NLAYERS=%d'%encoder_MLP_N_layers)

    encoder_N_hidden = trial.suggest_int('encoder_N_hidden', 32, 256)
    out.append('ENCODER_MLP_NHIDDEN=%d'%encoder_N_hidden)
    out.append('ENCODER_DEFAULT_NHIDDEN=%d'%encoder_N_hidden)
    out.append('NETWORK_DEFAULT_NLATENT=%d'%encoder_N_hidden)

    decoder_N_layers = trial.suggest_int('decoder_N_layers', 2, 9)
    out.append('DECODER_DEFAULT_NLAYERS=%d'%decoder_N_layers)

    decoder_N_hidden = trial.suggest_int('decoder_N_hidden', 32, 256)
    out.append('DECODER_DEFAULT_NHIDDEN=%d'%decoder_N_hidden)

    globals_noise = trial.suggest_float('globals_noise', 1e0, 3e1, log=True)
    out.append('GLOBALS_NOISE=%.8e'%globals_noise)

    basis_noise = trial.suggest_float('basis_noise', 1e-1, 1e1, log=True)
    out.append('BASIS_NOISE=%.8e'%basis_noise)

    dropout = trial.suggest_float('dropout', 0.0, 0.7)
    out.append('DECODER_DROPOUT=%.8e'%dropout)

    visible_dropout = trial.suggest_float('visible_dropout', 0.0, 0.5)
    out.append('DECODER_VISIBLE_DROPOUT=%.8e'%visible_dropout)

    N_DM = trial.suggest_int('N_DM', 1e2, 1e4, log=True)
    out.append('PRT_FRACTION["DM"]["training"]=%d'%N_DM)

    output_N_features = trial.suggest_categorical('output_N_features', (1, 2))
    out.append('OUTPUT_NFEATURES=%d'%output_N_features)

    return out
