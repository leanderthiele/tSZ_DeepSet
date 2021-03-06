from copy import copy

from torch.utils.data import DataLoader as torch_DataLoader
from torch.utils.data import get_worker_info as torch_get_worker_info

from init_proc import InitProc
from data_modes import DataModes
from data_set import DataSet
from data_batch import DataBatch
from merge_dicts import MergeDicts
import cfg


class _WorkerPool :
    """
    the init_worker method is called whenever a new worker is started
    """

    def __init__(self) :
    #{{{
        # store this in the instance so it gets carried over to the worker process
        self.local_rank = copy(cfg.local_rank)
    #}}}

    
    def init_worker(self, worker_id) :
        """
        use this method as worker_init_fn
        """
    #{{{
        InitProc(self.local_rank)

        worker_info = torch_get_worker_info()

        assert isinstance(worker_info.dataset, DataSet)

        worker_info.dataset.set_worker(worker_info.seed)
    #}}}


class DataLoader(torch_DataLoader) :
    """
    a torch-compatible data loader
    """
    
    def __init__(self, mode, **data_set_kwargs) :
    #{{{
        self.dataset = DataSet(mode, **data_set_kwargs)
        self.worker_pool = _WorkerPool()

        super().__init__(self.dataset,
                         collate_fn=DataBatch,
                         worker_init_fn=self.worker_pool.init_worker,
                         **(cfg.DATALOADER_ARGS if mode is DataModes.TRAINING \
                            else MergeDicts({'batch_size': 1, 'shuffle': False}, \
                                            cfg.DATALOADER_ARGS)))
    #}}}
