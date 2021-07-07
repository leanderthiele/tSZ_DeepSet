import os
from sys import argv

import torch

from mpi_env_types import MPIEnvTypes
import cfg


def _parse_cmd_line() :
    """
    replaces settings in cfg.py with those given on the command line
    The command line should look something like
       --VARIABLE_IN_CFG=value ...
    such that
        exec(cfg.VARIABLE_IN_CFG=value)
    is a valid python statement.
    In particular, be careful that strings are properly quoted and the quotation
    marks are escaped.
    NOTE it is allowed to have spaces in the command line items, because the double dash
         is used to separate components
    NOTE we assume that VARIABLE_IN_CFG is either the complete variable or is a complete
         variable followed by a [. In particular, no composite datatypes with attribute
         access are allowed
    NOTE we assume that VARIABLE_IN_CFG does not contain an = sign
    """
#{{{
    # cut out the program name
    _argv = argv[1:]

    if len(argv) == 0 :
        return

    # concatenate into one long string
    _argv = ''.join(_argv)

    # now split on the double dash
    _argv = _argv.split('--')

    for a in _argv :

        # implicitly checks for existence of equality sign
        cfg_key, _ = a.split('=', maxsplit=1)

        # account for the fact that there may be indexing/key access involved
        cfg_key = cfg_key.split('[', maxsplit=1)[0]

        # check that this is a valid key
        assert hasattr(cfg, cfg_key)

        # TODO for the [] case, make sure we are not adding a new item to the dict
        #      (for lists this would throw)
        
        # now hopefully everything is alright, let's do this super unsafe thing
        exec('cfg.%s'%a)
#}}}


def _set_mp_env() :
    """
    sets the multiprocessing variables that can be determined without knowledge
    of the intra-process rank
    """
#{{{
    if 'SLURM_SRUN_COMM_HOST' in os.environ :
        # we were launched using srun

        cfg.MPI_WORLD_SIZE = int(os.environ['SLURM_NTASKS'])
        cfg.MPI_RANK = int(os.environ['SLURM_PROCID'])
        cfg.MPI_LOCAL_WORLD_SIZE = len(os.environ['SLURM_GTIDS'].split(','))
        cfg.MPI_LOCAL_RANK = int(os.environ['SLURM_LOCALID'])
        cfg.MPI_NODENAME = os.environ['SLURMD_NODENAME']
        cfg.MASTER_ADDR = os.environ['SLURM_SRUN_COMM_HOST']

    elif 'OMPI_COMM_WORLD_SIZE' in os.environ :
        # we were launched using mpirun

        cfg.MPI_WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        cfg.MPI_RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
        cfg.MPI_LOCAL_WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
        cfg.MPI_LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_NODE_RANK'])
        cfg.MPI_NODENAME = os.environ['HOSTNAME']

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        assert comm.Get_rank() == cfg.MPI_RANK
        if cfg.MPI_RANK == 0 :
            root_name = cfg.MPI_NODENAME
        else :
            root_name = None
        root_name = comm.bcast(root_name, root=0)
        cfg.MASTER_ADDR = root_name

    else :
        # this is a single node job, maybe on the head node
        # we can leave the variables at their default values
        pass

    cfg.VISIBLE_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cfg.MPI_ENV_TYPE = MPIEnvTypes.get(cfg.VISIBLE_GPUS, cfg.MPI_LOCAL_WORLD_SIZE)
    cfg.WORLD_SIZE = cfg.MPI_ENV_TYPE.world_size(cfg.VISIBLE_GPUS, cfg.MPI_WORLD_SIZE)
#}}}


def _set_mp_env_for_rank(local_rank) :
    """
    sets the multiprocessing variables that depend on knowledge of the intra-process rank
    """
#{{{
    assert not cfg.SET_RANK
    cfg.LOCAL_RANK = local_rank
    cfg.RANK = cfg.MPI_ENV_TYPE.rank(cfg.MPI_RANK, cfg.LOCAL_RANK, cfg.VISIBLE_GPUS)
    cfg.DEVICE_IDX = cfg.MPI_ENV_TYPE.device_idx(cfg.MPI_LOCAL_RANK, cfg.LOCAL_RANK)
    cfg.SET_RANK = True
#}}}


def InitProc(local_rank=None) :
    """
    initializes the process -- this mainly means that the cfg file is populated
    with certain things we only know at runtime.

    local_rank ... process-local rank (i.e. within this MPI process)
    """
#{{{
    if cfg.INIT_PROC_CALLED :
        
        if local_rank is not None and not cfg.SET_RANK :
            assert local_rank == 0
            _set_mp_env_for_rank(local_rank)

        return

    _set_mp_env()

    if local_rank is not None :
        _set_mp_env_for_rank(local_rank)

    cfg.INIT_PROC_CALLED = True
#}}}
