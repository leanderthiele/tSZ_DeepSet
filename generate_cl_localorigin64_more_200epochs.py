
def generate_cl(trial) :
    
    out = []
    out.append('NET_ARCH=dict(origin=True,batt12=True,deformer=False,encoder=False,'\
                             'scalarencoder=False,decoder=True,vae=False,local=True)')

    out.append('EPOCHS=200')

    # constraints on the architecture
    out.append('OUTPUT_NFEATURES=2')
    out.append('DECODER_DEFAULT_R_PASSED=True')
    out.append('DECODER_DEFAULT_GLOBALS_PASSED=True')
    out.append('DECODER_DEFAULT_BASIS_PASSED=False')
    out.append('WEIGHT_DECAY["decoder"]=0')

    out.append('VALIDATION_EPOCHS=10')
    out.append('N_LOCAL["validation"]=512')
    out.append('PRT_FRACTION["DM"]["validation"]=2048')
    out.append('PRT_FRACTION["TNG"]["validation"]=4096')

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
    out.append('N_LOCAL["training"]=310')
    out.append('R_LOCAL=310.0')

    start_nr = trial.suggest_categorical('start_nr', (561, 1204, 426, 972, 228))
    out.append('NET_ID="optuna_localorigin64_nr%d"'%start_nr)

    N_layers = {561: 2, 1204: 2, 426: 2, 972: 2, 228: 2} 

    N_hidden = {561: 194, 1204: 151, 426: 238, 972: 180, 228: 145}

    dropout = {561: 0.06454442754625904,
               1204: 0.05665936906716564,
               426: 0.06480344546146014,
               972: 0.031902725133133136,
               228: 0.06997430266228007}

    visible_dropout = {561: 0.19055405345663895,
                       1204: 0.1678429304162785,
                       426: 0.1872595097319062,
                       972: 0.1520146908712116,
                       228: 0.20284369092258078}

    wd_origin = {561: 0.013593866618041945,
                 1204: 0.48634854411878076,
                 426: 0.006861012630829063,
                 972: 0.0913812530052948,
                 228: 0.0020201630867293092}

    wd_decoder = {561: 2.2917634801642205e-06,
                  1204: 1.1365413101716966e-05,
                  426: 2.8117191266124517e-06,
                  972: 1.9825974224953784e-06,
                  228: 2.4974159198580535e-06}

    out.append('DECODER_DEFAULT_NLAYERS=%d'%N_layers[start_nr])
    out.append('DECODER_DEFAULT_NHIDDEN=%d'%N_hidden[start_nr])
    out.append('DECODER_DROPOUT=%.8e'%dropout[start_nr])
    out.append('DECODER_VISIBLE_DROPOUT=%.8e'%visible_dropout[start_nr])

    lr_local = trial.suggest_float('lr_local', 1e-7, 1e-3, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["local"]["max_lr"]=%.8e'%lr_local)

    lr_origin = trial.suggest_float('lr_origin', 1e-7, 5e-4, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["origin"]["max_lr"]=%.8e'%lr_origin)

    lr_decoder = trial.suggest_float('lr_decoder', 1e-7, 1e-2, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["decoder"]["max_lr"]=%.8e'%lr_decoder)

    wd_origin_factor = trial.suggest_float('wd_origin_factor', 1e-1, 1e1, log=True)
    out.append('WEIGHT_DECAY["origin"]=%.8e'%(wd_origin_factor*wd_origin[start_nr]))

    wd_decoder_factor = trial.suggest_float('wd_decoder_factor', 1e-1, 1e1, log=True)
    out.append('WEIGHT_DECAY["decoder"]=%.8e'%(wd_decoder_factor*wd_decoder[start_nr]))

    gradient_clip = trial.suggest_float('gradient_clip', 1e-1, 1e0, log=True)
    out.append('GRADIENT_CLIP=%.8e'%gradient_clip)

    globals_noise = trial.suggest_float('globals_noise', 5, 40)
    out.append('GLOBALS_NOISE=%.8e'%globals_noise)



    return out
