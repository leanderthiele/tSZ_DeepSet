"""
Syntax in this file:
    > things that are normal to be touched from command line are in UPPER_CASE
    > things that should not be touched from command line have a _ prefixed, _UPPER_CASE
    > things that we use for communication (e.g. the multiprocessing environment) at runtime
      are in lower case
"""

# an identifier that we use to tag outputs
ID = 'default'

# for how many epochs to train
EPOCHS = 100

# whether to scale pressure by P200 internally
SCALE_PTH = True

# weight decay (applied only to weights)
WEIGHT_DECAY = dict(default=1e-4,
                    batt12=0,
                    deformer=1e-4,
                    origin=1e-4,
                    encoder=1e-4,
                    scalarencoder=1e-4,
                    decoder=1e-4,
                    vae=0,
                    local=0)

# how to construct the learning rate scheduler
ONE_CYCLE_LR_KWARGS = dict(default={'max_lr': 1e-3, 'div_factor': 1000},
                           batt12={'max_lr': 1e-2},
                           deformer={'max_lr': 1e-3},
                           origin={'max_lr': 3e-3},
                           encoder={'max_lr': 1e-2},
                           scalarencoder={'max_lr': 1e-2},
                           decoder={'max_lr': 1e-3},
                           vae={'max_lr': 1e-3},
                           local={'max_lr': 3e-4})

# where we have the halo catalog stored
HALO_CATALOG = 'halo_catalog.npz'

# the resolution of the TNG grids
TNG_RESOLUTION = 128

# how many particles of each type to use per halo
# set to None if all particles are to be used
# if an integer or float larger than one, not the fraction but this number is used
# NOTE that if the absolute number is given, it is not a bug to choose it larger than
# the number of particles in some low-mass halos. This will result in duplicate entries
# in the particle list for those halos, which should be ok.
PRT_FRACTION = dict(DM=1e4, TNG=256)

# whether to use memmap for the DM data (speeds up initial data loading)
MEMMAP_DM = False

# how far (in units of R200c) to go out when choosing TNG particles
# (set to None for no cutoff)
RMAX = 2.0

# how many local DM particles to use around each TNG position
N_LOCAL = 128

# how large the local spheres are (in kpc)
R_LOCAL = 100.0

# which components to include in the network architecture
# if batt12 is False, the deformer setting has no consequence
# one of batt12, decoder must be True and decoder must be True if encoder is True
# if decoder is False, encoder has no consequence
NET_ARCH = dict(origin=True,
                batt12=True,
                deformer=True,
                encoder=True,
                scalarencoder=True,
                decoder=True,
                vae=True,
                local=True)

# can list parts of the network architecture that should be frozen
# set to None for no freezing
NET_FREEZE = None

# can provide a pre-trained network here (can be a single ID or a list),
# set to None for no loading
NET_ID = None

# fraction of samples to use for each mode -- training is inferred
# NOTE with Mmin=5e13, we have 439 samples in total, of which 10% are ~44
#      which is very nicely divisibly by batch size 4
SAMPLE_FRACTIONS = dict(validation=0.2,
                        testing=0.1)

# binning of the pressure residuals with respect to a simple network
RESIDUALS_RMIN = 0.0
RESIDUALS_RMAX = 2.5
RESIDUALS_NBINS = 32

# noise on the pressure residuals (absolute value, data has stddev=1)
# Set to None for no noise
RESIDUALS_NOISE = None

# seed that we use whenever we need something to be consistent between runs
CONSISTENT_SEED = 137

# seed that we use for the random components of the network,
# i.e. the VAE part (should be chosen to have good balance between 0 and 1 bits!)
NETWORK_SEED = 1234567890123456789

# which globals to use -- set `none' to true if no globals are to be used
GLOBALS_USE = dict(none=False,
                   #-------#
                   logM=True,
                   Xoff=True,
                   Voff=True,
                   CM=True,
                   ang_mom=True,
                   inertia=True,
                   inertia_dot_ang_mom=False,
                   vel_dispersion=True,
                   vel_dispersion_dot_ang_mom=False,
                   vel_dispersion_dot_inertia=False)

# amount of noise added to the globals
# 1 means that confusion with other halos is almost impossible
# set to None if no noise desired
GLOBALS_NOISE = 5

# which basis vectors to use -- set `none' to true if no basis vectors are to be used
BASIS_USE = dict(none=False,
                 #--------#
                 ang_mom=True,
                 CM=True,
                 inertia=True,
                 vel_dispersion=True)

# amount of noise added to the vectors
# specifies standard deviation of the Gaussians from which rotation angles will be drawn (in degrees)
# [currently, we keep the basis vectors normed]
# set to None if no noise is desired
BASIS_NOISE = 2.0

# whether the DM particle velocities should be used
USE_VELOCITIES = True

# arguments passed to the torch DataLoader constructor
DATALOADER_ARGS = dict(batch_size=4,
                       num_workers=8,
                       pin_memory=True, # TODO for this to have an effect, I think the Batch class must have a pin_memory() method
                       prefetch_factor=1,
                       drop_last=False)

# default number of hidden layers for the MLPs
MLP_DEFAULT_NLAYERS = 4

# default number of neurons in the hidden layers for the MLPs
MLP_DEFAULT_NHIDDEN = 128

# default way to initialize the bias
# a string containing one %s which takes the bias and can be concatenated with nn.init. to form a valid statement
MLP_DEFAULT_BIAS_INIT = 'normal_(%s, mean=0.0, std=0.1)'

# should be such that torch.nn.<string>() is wellformed,
# or torch.nn.<string> is wellformed (can add arguments in parentheses here)
MLP_DEFAULT_ACTIVATION = 'LeakyReLU'

# whether to apply layer normalization in the hidden layers
LAYERNORM = True

# dropout rate (for the hidden neurons) -- set to None if not wanted
DROPOUT = None

# dropout rate (for the visible neurons) -- set to None if not wanted
VISIBLE_DROPOUT = None

# whether the number of particles is used in each initial MLP in the local network
# (should not be necessary)
LOCAL_PASS_N = False

# whether after the pooling the scalars are concatenated with N
LOCAL_CONCAT_WITH_N = True

# default number of hidden layers in the local network
# (I believe anything apart from 0, 1 doesn't make much sense)
LOCAL_NLAYERS = 1

# default number of hidden neurons in the local network (between the MLPs)
LOCAL_NHIDDEN = 192

# how many local features to compress to
LOCAL_NLATENT = 64

# structure of the MLPs in the local net: # of hidden layers
LOCAL_MLP_NLAYERS = 4

# structure of the MLPs in the local net: # of hidden neurons
LOCAL_MLP_NHIDDEN = 192

# whether to include any information about the relative position and velocity of the local environment
LOCAL_PASS_RELATIVE_TO_HALO = False

# default number of features output by scalar encoder
SCALAR_ENCODER_NLATENT = 128

# default number of hidden layers in scalar encoder [only 0 or 1 really make sense]
SCALAR_ENCODER_NLAYERS = 1

# default number of hidden neurons in scalar encoder
SCALAR_ENCODER_NHIDDEN = 128

# inside the scalar encoder MLPs
SCALAR_ENCODER_MLP_NLAYERS = 4

# inside the scalar encoder MLPs
SCALAR_ENCODER_MLP_NHIDDEN = 128

# whether basis is passed to scalar encoder
SCALAR_ENCODER_BASIS_PASSED = True

# whether globals are passed to scalar encoder
SCALAR_ENCODER_GLOBALS_PASSED = False

# default number of hidden layers in the encoder
ENCODER_DEFAULT_NLAYERS = 1

# default number of neurons in the outer layers of the MLPs
# (which correspond to hidden layers at the encoder level)
ENCODER_DEFAULT_NHIDDEN = 128

# until which layer the encoder should pass the basis
ENCODER_DEFAULT_BASIS_MAXLAYER = 0

# until which layer the encoder should pass the globals
ENCODER_DEFAULT_GLOBALS_MAXLAYER = 0

# default for each MLP in encoder
ENCODER_MLP_NLAYERS = 4

# default for each MLP in encoder
ENCODER_MLP_NHIDDEN = 128

# whether we pass the TNG radii to the decoder
DECODER_DEFAULT_R_PASSED = True

# whether we pass the globals to the decoder
DECODER_DEFAULT_GLOBALS_PASSED = True

# whether we pass the basis to the decoder
DECODER_DEFAULT_BASIS_PASSED = False

# how many layers to have in the decoder
DECODER_DEFAULT_NLAYERS = 4

# how many hidden features to have in the decoder
DECODER_DEFAULT_NHIDDEN = 128

# number of latent variables in the VAE (x2 for variance & mean)
VAE_NLATENT = 1

# whether the latent variables in the VAE are to be computed or randomly drawn
VAE_RAND_LATENT = False

# number of layers in VAE MLP
VAE_NLAYERS = 4

# number of hidden features in VAE MLP
VAE_NHIDDEN = 128

# scaling of the KL divergence loss
KLD_SCALING = 1e-5

# KLD annealing (if True, we anneal from an AE to a VAE with the given KLD scaling)
KLD_ANNEALING = True

# when to start KLD annealing (in units of # epochs)
KLD_ANNEALING_START = 0.2

# when to finish KLD annealing (in units of # epochs)
KLD_ANNEALING_END = 0.8

# how many layers we have in the origin encoder (other settings equal to the usual encoder
#    at least for now)
ORIGIN_DEFAULT_NLAYERS = 3

# default number of neurons in the outer layers of the MLPs
# (which correspond to hidden layers at the encoder level)
ORIGIN_DEFAULT_NHIDDEN = 128

# until which layer the origin net should pass the basis
ORIGIN_DEFAULT_BASIS_MAXLAYER = 0

# until which layer the origin net should pass the globals
ORIGIN_DEFAULT_GLOBALS_MAXLAYER = 0

# the individual MLPs in the origin network
ORIGIN_MLP_NLAYERS = 4

# the individual MLPs in the origin network
ORIGIN_MLP_NHIDDEN = 128

# default number of latent features in the space separating encoder and decoder
NETWORK_DEFAULT_NLATENT = 128

# number of features per point in the final output
# so far, 1 and 2 are implemented
OUTPUT_NFEATURES = 2

# where our results (like model states and loss curves) will go
RESULTS_PATH = '/scratch/gpfs/lthiele/tSZ_DeepSet_results'

# gradient clipping value (set to None for no clipping)
GRADIENT_CLIP = 1.0

# multiprocessing environment -- these variables are potentially changed by init_proc
# and set to head-node process defaults
# NOTE these things should never be touched from the command line,
#      which is indicated by them being in lower case

# number of MPI-level processes
mpi_world_size = 1

# rank within the MPI team
mpi_rank = 0

# number of MPI processes running on this node
mpi_local_world_size = 1

# rank within the MPI processes running on this node
mpi_local_rank = 0

# our name
mpi_nodename = 'localhost'

# the name of the root machine
master_addr = None

# number of GPUs visible from this MPI-level process
visible_gpus = 0

# what type of MPI environment we are in
mpi_env_type = None

# total number of processes, including spawned training processes
# but not workers
world_size = 1

# our rank in the entire world, including spawned training processes
# but not workers
rank = 0

# whether the RANK variable has been set
set_rank = False

# our rank within this MPI process (usually zero, except for spawned training processes)
local_rank = 0

# the GPU we want to use -- this needs to be modified from somewhere
device_idx = None

# whether the init_proc function has been called on this module
init_proc_called = False

# NOTE the following are 'constants', in that we expect that it should never be necessary
#      to access them from the command line
#      This is indicated by the prefixed underscore

# which redshift snapshot to work with
_SNAP_IDX = 99

# the particle types in Illustris
_PART_TYPES = dict(DM=1, TNG=0)

# the (virtual) hdf5 files where we find the simulations
_SIM_FILES = dict(DM='/tigress/lthiele/Illustris_300-1_Dark/simulation.hdf5',
                  TNG='/tigress/lthiele/Illustris_300-1_TNG/simulation.hdf5')

# the simulation box size -- avoid repeated and perhaps concurrent hdf5 reads
# by hardcoding this here
_BOX_SIZE = 205000.0

# the DM simulation particle mass
_UNIT_MASS = 0.00472716

# some constants in simulation units
_RHO_CRIT = 2.775e-8 # critical density at z=0 in Illustris code units
_G_NEWTON = 4.30091e4 # Newton's constant in Illustris code units
_OMEGA_B = 0.0486
_OMEGA_M = 0.3089

# the binary files where we store particle information and some global
# properties of the halo
# '%d' is placeholder for the halo index
# '%s' is placeholder for some string, which can be
#   globals    [DM & TNG]
#   coords     [DM & TNG]
#   Pth        [TNG only]
#   residuals  [TNG only]
#   velocities [DM only]
#   offsets    [DM only]
_STORAGE_FILES = dict(DM='/scratch/gpfs/lthiele/tSZ_DeepSet_halos/rockstar/DM_%d_%s.bin',
                      TNG='/scratch/gpfs/lthiele/tSZ_DeepSet_halos/rockstar/TNG_%d_%s.bin')

