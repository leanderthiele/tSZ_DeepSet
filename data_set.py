import numpy as np

import torch
from torch.utils.data.dataset import Dataset as torch_DataSet

from data_modes import DataModes
from data_item import DataItem
from halo import Halo
from halo_catalog import HaloCatalog
import cfg

class DataSet(torch_DataSet) :
    """
    torch-compatible representation of the simulation data
    """

    def __init__(self, mode, **data_item_kwargs) :
        """
        mode ... one of training, validation, testing
        """
    #{{{    
        assert isinstance(mode, DataModes)

        self.mode = mode

        self.halo_catalog = HaloCatalog(mode)

        self.data_items = []

        # FIXME for distributed training, we only want to load a subset
        #       of the data from disk as we know we will not access any others!
        #       This could save a lot of memory

        for h in self.halo_catalog :
            self.data_items.append(DataItem(h, mode, **data_item_kwargs))
    #}}}


    def set_worker(self, seed) :
        """
        to be called from worker_init_fn to initialize the random number generator for this worker
        """
    #{{{
        self.rng = np.random.default_rng(seed % 2**32)
    #}}}


    def __getitem__(self, idx) :
        """
        idx ... local index within the samples this rank operates on
        """
    #{{{
        return self.__getitem_all((idx * cfg.WORLD_SIZE + cfg.RANK) % self.__len_all())
    #}}}


    def __len__(self) :
        """
        returns number of samples this rank operates on
        (if the world size does not divide the available number of samples,
         we round up and some camples will be used twice)
        """
    #{{{
        return int(np.ceil(self.__len_all() / cfg.WORLD_SIZE))
    #}}}


    def __getitem_all(self, idx) :
        """
        idx ... global index over the entire halo catalog (used for this mode)
        """
    #{{{ 
        indices = dict(DM=self.__get_indices(self.data_items[idx].halo, 'DM'),
                       TNG=self.__get_indices(self.data_items[idx].halo, 'TNG'))

        return self.data_items[idx].sample_particles(indices)
    #}}}


    def __len_all(self) :
        """
        operates on the entire halo catalog (used for this mode)
        """
    #{{{
        return len(self.data_items)
    #}}}


    def __get_indices(self, halo, ptype) :
        """
        generates the randomly chosen index arrays that we use to choose particles
        halo  ... the halo instance on which to work
        ptype ... one of 'DM', 'TNG'
        """
    #{{{ 
        assert isinstance(halo, Halo)
        assert ptype in ['DM', 'TNG']

        if cfg.PRT_FRACTION is None \
           or ptype not in cfg.PRT_FRACTION \
           or cfg.PRT_FRACTION[ptype] is None :
            return None

        Nprt = halo.prt_len_DM if ptype == 'DM' else halo.prt_len_TNG
        Nindices = int(cfg.PRT_FRACTION[ptype] * Nprt) \
                   if isinstance(cfg.PRT_FRACTION[ptype], float) \
                   else cfg.PRT_FRACTION[ptype]

        # here we allow the possibility for duplicate entries
        # in practice, this should not be an issue and will make the code faster
        return self.rng.integers(Nprt, size=Nindices)
    #}}}
