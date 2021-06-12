from copy import copy

from torch.utils.data import DataLoader as torch_DataLoader

from init_proc import InitProc
from data_modes import DataModes
from data_set import DataSet
from data_batch import DataBatch
import cfg


class _WorkerPool :
    """
    the init_worker method is called whenever a new worker is started
    """

    def __init__(self) :
    #{{{
        # store this in the instance so it gets carried over to the worker process
        self.local_rank = copy(cfg.LOCAL_RANK)
    #}}}

    
    def init_worker(self, worker_id) :
        """
        use this method as worker_init_fn
        """
    #{{{
        InitProc(self.local_rank)
    #}}}


class DataLoader(torch_DataLoader) :
    """
    a torch-compatible data loader
    """
    
    def __init__(self, mode, seed, load_DM=True, load_TNG=True) :
        """
        mode ... one of training, validation, testing
        seed ... random seed to choose particles
        load_DM  ... whether to load the DM particles
        load_TNG ... whether to load the TNG particles
        """
    #{{{
        assert isinstance(mode, DataModes)

        self.dataset = DataSet(mode, seed, load_DM=load_DM, load_TNG=load_TNG)
        self.worker_pool = _WorkerPool()

        super().__init__(self.dataset,
                         collate_fn=DataBatch,
                         worker_init_fn=self.worker_pool.init_worker,
                         **cfg.DATALOADER_ARGS)
    #}}}
