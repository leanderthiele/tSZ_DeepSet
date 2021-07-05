# whether the init_proc function has been called on this module
INIT_PROC_CALLED = False

# which redshift snapshot to work with
SNAP_IDX = 99

# the particle types in Illustris
PART_TYPES = dict(DM=1, TNG=0)

# where we have the halo catalog stored
HALO_CATALOG = 'halo_catalog.npz'

# the (virtual) hdf5 files where we find the simulations
SIM_FILES = dict(DM='/tigress/lthiele/Illustris_300-1_Dark/simulation.hdf5',
                 TNG='/tigress/lthiele/Illustris_300-1_TNG/simulation.hdf5')

# the simulation box size -- avoid repeated and perhaps concurrent hdf5 reads
# by hardcoding this here
BOX_SIZE = 205000.0

# the DM simulation particle mass
UNIT_MASS = 0.00472716

# some constants in simulation units
RHO_CRIT = 2.775e-8 # critical density at z=0 in Illustris code units
G_NEWTON = 4.30091e4 # Newton's constant in Illustris code units
OMEGA_B = 0.0486
OMEGA_M = 0.3089

# the binary files where we store particle information and some global
# properties of the halo
# '%d' is placeholder for the halo index
# '%s' is placeholder for some string, which can be
#   globals    [DM & TNG]
#   coords     [DM & TNG]
#   masses     [TNG only]
#   Pth        [TNG only]
#   velocities [DM only]
STORAGE_FILES = dict(DM='/scratch/gpfs/lthiele/tSZ_DeepSet_halos/rockstar/DM_%d_%s.bin',
                     TNG='/scratch/gpfs/lthiele/tSZ_DeepSet_halos/rockstar/TNG_%d_%s.bin')

# how many particles of each type to use per halo
# set to None if all particles are to be used
# if an integer or float larger than one, not the fraction but this number is used
# NOTE that if the absolute number is given, it is not a bug to choose it larger than
# the number of particles in some low-mass halos. This will result in duplicate entries
# in the particle list for those halos, which should be ok.
PRT_FRACTION = dict(DM=1e4, TNG=1e5)

# whether to divide the coordinates by R200c
NORMALIZE_COORDS = True

# which components to include in the network architecture
# if batt12 is False, the deformer setting has no consequence
# one of batt12, decoder must be True and decoder must be True if encoder is True
# if decoder is False, encoder has no consequence
NET_ARCH = dict(origin=True,
                batt12=True,
                deformer=False,
                encoder=True,
                decoder=True)

# fraction of samples to use for each mode -- training is inferred
# NOTE with Mmin=5e13, we have 439 samples in total, of which 10% are ~44
#      which is very nicely divisibly by batch size 4
SAMPLE_FRACTIONS = dict(validation=0.2,
                        testing=0.1)

# seed that we use whenever we need something to be consistent between runs
CONSISTENT_SEED = 137

# which globals to use -- set `none' to true if no globals are to be used
GLOBALS_USE = dict(none=False,
                   #-------#
                   logM=True,
                   Xoff=False,
                   Voff=False,
                   CM=False,
                   ang_mom=False,
                   inertia=True,
                   inertia_dot_ang_mom=False,
                   vel_dispersion=True,
                   vel_dispersion_dot_ang_mom=False,
                   vel_dispersion_dot_inertia=False)

# amount of noise added to the globals
# 1 means that confusion with other halos is almost impossible
# set to None if no noise desired
GLOBALS_NOISE = 5.0

# which basis vectors to use -- set `none' to true if no basis vectors are to be used
BASIS_USE = dict(none=False,
                 #--------#
                 ang_mom=False,
                 CM=False,
                 inertia=True,
                 vel_dispersion=False)

# amount of noise added to the vectors
# specifies standard deviation of the Gaussians from which rotation angles will be drawn (in degrees)
# [currently, we keep the basis vectors normed]
# set to None if no noise is desired
BASIS_NOISE = 20.0

# whether the DM particle velocities should be used
USE_VELOCITIES = True

# arguments passed to the torch DataLoader constructor
DATALOADER_ARGS = dict(batch_size=8,
                       num_workers=4,
                       pin_memory=True, # TODO for this to have an effect, I think the Batch class must have a pin_memory() method
                       prefetch_factor=1,
                       drop_last=False)

# default number of hidden layers for the MLPs
MLP_DEFAULT_NLAYERS = 4

# default number of neurons in the hidden layers for the MLPs
MLP_DEFAULT_NHIDDEN = 128

# whether to apply layer normalization in the hidden layers
LAYERNORM = True

# dropout rate (for the hidden neurons) -- set to None if not wanted
DROPOUT = 0.5

# dropout rate (for the visible neurons) -- set to None if not wanted
VISIBLE_DROPOUT = None

# default number of hidden layers in the encoder
ENCODER_DEFAULT_NLAYERS = 2

# default number of neurons in the outer layers of the MLPs
# (which correspond to hidden layers at the encoder level)
ENCODER_DEFAULT_NHIDDEN = 128

# until which layer the encoder should pass the basis
ENCODER_DEFAULT_BASIS_MAXLAYER = 0

# until which layer the encoder should pass the globals
ENCODER_DEFAULT_GLOBALS_MAXLAYER = 0

# whether we pass the TNG radii to the decoder
DECODER_DEFAULT_R_PASSED = False

# whether we pass the globals to the decoder
DECODER_DEFAULT_GLOBALS_PASSED = False

# whether we pass the basis to the decoder
DECODER_DEFAULT_BASIS_PASSED = False

# default number of latent features in the space separating encoder and decoder
NETWORK_DEFAULT_NLATENT = 128

# number of features per point in the final output
# so far, 1 and 2 are implemented
OUTPUT_NFEATURES = 1

# where our results (like model states and loss curves) will go
RESULTS_PATH = 'results'

# gradient clipping value (set to None for no clipping)
GRADIENT_CLIP = 1.0

# multiprocessing environment -- these variables are potentially changed by init_proc
# and set to head-node process defaults

# number of MPI-level processes
MPI_WORLD_SIZE = 1

# rank within the MPI team
MPI_RANK = 0

# number of MPI processes running on this node
MPI_LOCAL_WORLD_SIZE = 1

# rank within the MPI processes running on this node
MPI_LOCAL_RANK = 0

# our name
MPI_NODENAME = 'localhost'

# the name of the root machine
MASTER_ADDR = None

# number of GPUs visible from this MPI-level process
VISIBLE_GPUS = 0

# what type of MPI environment we are in
MPI_ENV_TYPE = None

# total number of processes, including spawned training processes
# but not workers
WORLD_SIZE = 1

# our rank in the entire world, including spawned training processes
# but not workers
RANK = 0

# whether the RANK variable has been set
SET_RANK = False

# our rank within this MPI process (usually zero, except for spawned training processes)
LOCAL_RANK = 0

# the GPU we want to use -- this needs to be modified from somewhere
DEVICE_IDX = None
