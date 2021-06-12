import numpy as np

from data_modes import DataModes
from data_item import DataItem
from halo import Halo
import cfg

import torch
from torch.utils.data.dataset import Dataset as torch_DataSet

class DataSet(torch_DataSet) :
    """
    torch-compatible representation of the simulation data
    """

    def __init__(self, mode, seed, load_DM=True, load_TNG=True) :
        """
        mode ... one of training, validation, testing
        seed ... to choose the particle indices
        load_DM  ... whether to load the DM particles
        load_TNG ... whether to load the TNG particles
        """
    #{{{    
        assert isinstance(mode, DataModes)

        self.mode = mode
        self.sample_indices = mode.sample_indices()

        self.load_DM = load_DM
        self.load_TNG = load_TNG

        self.halo_catalog = dict(np.load(cfg.HALO_CATALOG))
        self.rng = np.random.default_rng(seed)
    #}}}


    def __getitem__(self, idx) :
        """
        idx ... local index within the samples this rank operates on
        """
    #{{{
        return self.__getitem_all(idx * cfg.WORLD_SIZE + cfg.RANK)
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
        idx ... global index over the entire halo catalog
                (can be somewhat larger than the largest index in the catalog,
                 see doc to __len__)
        """
    #{{{ 
        # bring back into the allowed range
        idx %= self.__len_all()

        h = Halo(self.halo_catalog, self.sample_indices[idx])

        indices = dict(DM = self.__get_indices(h, 'DM'),
                       TNG = self.__get_indices(h, 'TNG'))

        return DataItem(h, indices, load_DM=self.load_DM, load_TNG=self.load_TNG)
    #}}}


    def __len_all(self) :
        """
        operates on the entire halo catalog
        """
    #{{{
        return len(self.sample_indices)
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
