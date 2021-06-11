
from torch.utils.data import DataLoader as torch_DataLoader

from data_modes import DataModes
from data_set import DataSet
from data_batch import DataBatch
import cfg

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

        super().__init__(self.dataset,
                         collate_fn=DataBatch,
                         # TODO for distributed training (or actually whenever we have a changing variable in cfg)
                         #      we need to use the worker_init_fn argument here!
                         **cfg.DATALOADER_ARGS)
    #}}}
