# whether the init_proc function has been called on this module
INIT_PROC_CALLED = False

# which redshift snapshot to work with
SNAP_IDX = 99

# the particle types in Illustris
PART_TYPES = dict(DM = 1, TNG = 0)

# where we have the halo catalog stored
HALO_CATALOG = 'halo_catalog.npz'

# the (virtual) hdf5 files where we find the simulations
SIM_FILES = dict(DM = '/tigress/lthiele/Illustris_300-1_Dark/simulation.hdf5',
                 TNG = '/tigress/lthiele/Illustris_300-1_TNG/simulation.hdf5')

# how many particles of each type to use per halo
# set to None if all particles are to be used
# if an integer larger than one, not the fraction but this number is used
PRT_FRACTION = dict(DM = None, TNG = 0.1)

# whether to divide the coordinates by R200c
NORMALIZE_COORDS = True

# fraction of samples to use for each mode -- training is inferred
SAMPLE_FRACTIONS = dict(validation = 0.2,
                        testing = 0.1)

# seed that we use whenever we need something to be consistent between runs
CONSISTENT_SEED = 137

# arguments passed to the torch DataLoader constructor
DATALOADER_ARGS = dict(batch_size=2,
                       shuffle=True,
                       num_workers=1,
                       pin_memory=True,
                       prefetch_factor=1)

# default number of hidden layers for the MLPs
MLP_DEFAULT_NLAYERS = 4

# default number of neurons in the hidden layers for the MLPs
MLP_DEFAULT_NHIDDEN = 128

# number of global features
NGLOBALS = 1

# number of global basis vectors
NBASIS = 0

# default number of hidden layers in the encoder
ENCODER_DEFAULT_NLAYERS = 4

# default number of neurons in the outer layers of the MLPs
# (which correspond to hidden layers at the encoder level)
ENCODER_DEFAULT_NHIDDEN = 128

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

