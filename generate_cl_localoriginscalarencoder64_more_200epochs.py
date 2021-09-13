
def generate_cl(trial) :

    out = []
    out.append('NET_ARCH=dict(origin=True,batt12=True,deformer=False,encoder=False,'\
                             'scalarencoder=True,decoder=True,vae=False,local=True)')

    out.append('EPOCHS=200')

    # constraints on the architecture
    out.append('OUTPUT_NFEATURES=2')
    out.append('DECODER_DEFAULT_R_PASSED=True')
    out.append('DECODER_DEFAULT_GLOBALS_PASSED=True')
    out.append('DECODER_DEFAULT_BASIS_PASSED=True')
    out.append('WEIGHT_DECAY["decoder"]=0')
    out.append('SCALAR_ENCODER_BASIS_PASSED=True')

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
    out.append('PRT_FRACTION["DM"]["training"]=2048')
    out.append('N_LOCAL["training"]=310')
    out.append('R_LOCAL=310.0')

    start_nr = trial.suggest_categorical('start_nr', (870, 313, 701, 629, 837))
    out.append('NET_ID="optuna_localoriginscalarencoder64_nr%d"'%start_nr)

    N_layers_decoder = {870: 2, 313: 2, 701: 2, 629: 2, 837: 2}

    N_hidden_decoder = {870: 217, 313: 233, 701: 217, 629: 240, 837: 205} 

    N_layers_encoder = {870: 0, 313: 1, 701: 1, 629: 1, 837: 0}

    MLP_N_layers_encoder = {870 : 2, 313: 4, 701: 2, 629: 3, 837: 2}

    N_hidden_encoder = {870: 153, 313: 144, 701: 122, 629: 147, 837: 147}

    globals_passed = {870: True, 313: False, 701: False,  629: True, 837: True}

    dropout = {870: 0.16302377755127598,
               313: 0.29642904337404097,
               701: 0.16849520885577712,
               629: 0.20746638541838283,
               837: 0.15591309241419085}

    visible_dropout = {870: 0.08156551736126039,
                       313: 0.1277624465632306,
                       701: 0.23823079211310227,
                       629: 0.26814005710523375,
                       837: 0.09639495159955885}

    origin_shift_dm = {870: False, 313: True, 701: True, 629: False, 837: False}

    wd_origin = {870: 0.017226295470795815,
                 313: 0.00956989536523893,
                 701: 0.008957320750744714,
                 629: 0.0027397327041378487,
                 837: 0.020584779881406074}

    wd_decoder = {870: 0.0009246060465075162,
                  313: 0.0017575640198731518,
                  701: 0.0006428643900380651,
                  629: 7.217171587439046e-07,
                  837: 0.001467993466968475}

    wd_encoder = {870: 0.4033372154795013,
                  313: 0.0011135968497832788,
                  701: 0.0004274451379758852,
                  629: 0.0005811984352614534,
                  837: 0.4338227922995636}

    out.append('DECODER_DEFAULT_NLAYERS=%d'%N_layers_decoder[start_nr])
    out.append('DECODER_DEFAULT_NHIDDEN=%d'%N_hidden_decoder[start_nr])
    out.append('SCALAR_ENCODER_NLAYERS=%d'%N_layers_encoder[start_nr])
    out.append('SCALAR_ENCODER_MLP_NLAYERS=%d'%MLP_N_layers_encoder[start_nr])
    out.append('SCALAR_ENCODER_NHIDDEN=%d'%N_hidden_encoder[start_nr])
    out.append('SCALAR_ENCODER_NLATENT=%d'%N_hidden_encoder[start_nr])
    out.append('SCALAR_ENCODER_MLP_NHIDDEN=%d'%N_hidden_encoder[start_nr])
    out.append('SCALAR_ENCODER_GLOBALS_PASSED=%s'%globals_passed[start_nr])
    out.append('DECODER_DROPOUT=%.8e'%dropout[start_nr])
    out.append('DECODER_VISIBLE_DROPOUT=%.8e'%visible_dropout[start_nr])
    out.append('ORIGIN_SHIFT_DM=%s'%origin_shift_dm[start_nr])

    wd_origin_factor = trial.suggest_float('wd_origin_factor', 1e-2, 1e1, log=True)
    out.append('WEIGHT_DECAY["origin"]=%.8e'%(wd_origin_factor*wd_origin[start_nr]))

    wd_decoder_factor = trial.suggest_float('wd_decoder_factor', 1e-2, 1e1, log=True)
    out.append('WEIGHT_DECAY["decoder"]=%.8e'%(wd_decoder_factor*wd_decoder[start_nr]))

    wd_encoder_factor = trial.suggest_float('wd_encoder_factor', 1e-2, 1e1, log=True)
    out.append('WEIGHT_DECAY["scalarencoder"]=%.8e'%(wd_encoder_factor*wd_encoder[start_nr]))

    lr_local = trial.suggest_float('lr_local', 1e-7, 1e-4, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["local"]["max_lr"]=%.8e'%lr_local)

    lr_origin = trial.suggest_float('lr_origin', 3e-5, 1e-3, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["origin"]["max_lr"]=%.8e'%lr_origin)

    lr_decoder = trial.suggest_float('lr_decoder', 1e-7, 2e-4, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["decoder"]["max_lr"]=%.8e'%lr_decoder)

    lr_encoder = trial.suggest_float('lr_encoder', 1e-6, 5e-4, log=True)
    out.append('ONE_CYCLE_LR_KWARGS["scalarencoder"]["max_lr"]=%.8e'%lr_encoder)

    gradient_clip = trial.suggest_float('gradient_clip', 1e-2, 1e0, log=True)
    out.append('GRADIENT_CLIP=%.8e'%gradient_clip)

    globals_noise = trial.suggest_float('globals_noise', 30, 50)
    out.append('GLOBALS_NOISE=%.8e'%globals_noise)



    return out
