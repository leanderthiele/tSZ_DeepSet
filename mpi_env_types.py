from enum import Enum, auto


class MPIEnvTypes(Enum) :

    # we have no GPU available
    # in this case, we assume it is a single-rank job
    NOGPU = auto()

    # we see a single GPU
    SINGLEGPU = auto()

    # we see multiple GPUs that we have to share between ranks
    MULTIGPU_MULTIRANK = auto()

    # we see multiple GPUs which are all accessible from our rank
    MULTIGPU_SINGLERANK = auto()

    @classmethod
    def get(cls, visible_gpus, mpi_local_world_size) :
        """
        this is the `constructor'
        """
    #{{{
        if visible_gpus == 0 :
            return cls.NOGPU
        elif visible_gpus == 1 :
            return cls.SINGLEGPU
        elif visible_gpus > 1 and mpi_local_world_size > 1 :
            return cls.MULTIGPU_MULTIRANK
        elif visible_gpus > 1 and mpi_local_world_size == 1 :
            return cls.MULTIGPU_SINGLERANK
        else :
            raise RuntimeError('Invalid MPI environment.')
    #}}}


    def world_size(self, visible_gpus, mpi_world_size) :
        """
        returns the total number of processes in the team
        (usually this equals mpi_world_size but can be different if a single rank uses multiple GPUs,
         in which case the single rank spawns several training processes)
        """
    #{{{
        if self is MPIEnvTypes.MULTIGPU_SINGLERANK :
            return visible_gpus * mpi_world_size
        else :
            return mpi_world_size
    #}}}


    def rank(self, mpi_rank, local_rank, visible_gpus) :
        """
        returns the process's rank within the entire team (including spawned training processes)
        """
    #{{{
        if self is MPIEnvTypes.MULTIGPU_SINGLERANK :
            return mpi_rank * visible_gpus + local_rank
        else :
            return mpi_rank
    #}}}


    def device_idx(self, mpi_local_rank, local_rank) :
        """
        returns the index of the device we want to use
        """
    #{{{
        if self is MPIEnvTypes.NOGPU :
            return None
        elif self is MPIEnvTypes.SINGLEGPU :
            return 0
        elif self is MPIEnvTypes.MULTIGPU_MULTIRANK :
            return mpi_local_rank
        elif self is MULTIGPU_SINGLERANK :
            return local_rank
    #}}}
