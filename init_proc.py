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
    _argv = _argv.strip().split('--')

    # since our string starts with '--', the first entry will be an empty string
    assert _argv[0] == ''
    _argv = _argv[1:]

    for a in _argv :
        
        # implicitly checks for existence of equality sign
        cfg_key, _ = a.split('=', maxsplit=1)

        # account for the fact that there may be indexing/key access involved
        cfg_key = cfg_key.split('[', maxsplit=1)[0]

        # check that this is a valid key
        assert hasattr(cfg, cfg_key), cfg_key

        # check that this is a key that we allow to be changed
        assert cfg_key.isupper(), cfg_key
        assert not cfg_key.startswith('_'), cfg_key

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

        cfg.mpi_world_size = int(os.environ['SLURM_NTASKS'])
        cfg.mpi_rank = int(os.environ['SLURM_PROCID'])
        cfg.mpi_local_world_size = len(os.environ['SLURM_GTIDS'].split(','))
        cfg.mpi_local_rank = int(os.environ['SLURM_LOCALID'])
        cfg.mpi_nodename = os.environ['SLURMD_NODENAME']
        cfg.master_addr = os.environ['SLURM_SRUN_COMM_HOST']

    elif 'OMPI_COMM_WORLD_SIZE' in os.environ :
        # we were launched using mpirun

        cfg.mpi_world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        cfg.mpi_rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        cfg.mpi_local_world_size = int(os.environ['OMPI_COMM_WORLD_LOCAL_SIZE'])
        cfg.mpi_local_rank = int(os.environ['OMPI_COMM_WORLD_NODE_RANK'])
        cfg.mpi_nodename = os.environ['HOSTNAME']

        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        assert comm.Get_rank() == cfg.mpi_rank
        if cfg.mpi_rank == 0 :
            root_name = cfg.mpi_nodename
        else :
            root_name = None
        root_name = comm.bcast(root_name, root=0)
        cfg.master_addr = root_name

    else :
        # this is a single node job, maybe on the head node
        # we can leave the variables at their default values
        pass

    cfg.visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cfg.mpi_env_type = MPIEnvTypes.get(cfg.visible_gpus, cfg.mpi_local_world_size)
    cfg.world_size = cfg.mpi_env_type.world_size(cfg.visible_gpus, cfg.mpi_world_size)
#}}}


def _set_mp_env_for_rank(local_rank) :
    """
    sets the multiprocessing variables that depend on knowledge of the intra-process rank
    """
#{{{
    assert not cfg.set_rank
    cfg.local_rank = local_rank
    cfg.rank = cfg.mpi_env_type.rank(cfg.mpi_rank, cfg.local_rank, cfg.visible_gpus)
    cfg.device_idx = cfg.mpi_env_type.device_idx(cfg.mpi_local_rank, cfg.local_rank)
    cfg.set_rank = True
#}}}


def InitProc(local_rank=None) :
    """
    initializes the process -- this mainly means that the cfg file is populated
    with certain things we only know at runtime.

    local_rank ... process-local rank (i.e. within this MPI process)
    """
#{{{
    if cfg.init_proc_called :
        
        if local_rank is not None and not cfg.set_rank :
            assert local_rank == 0
            _set_mp_env_for_rank(local_rank)

        return

    _set_mp_env()

    if local_rank is not None :
        _set_mp_env_for_rank(local_rank)

    _parse_cmd_line()

    cfg.init_proc_called = True
#}}}
