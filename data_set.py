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
    In training mode, in order to always be able to sync between distributed
    processes and so that batches are completely filled,
    the data set potentially returns duplicate samples.
    On the other hand, in validation and testing mode this does not happen
    and different processes will in general see different lengths of the training set
    """

    def __init__(self, mode, **data_item_kwargs) :
        """
        mode ... one of training, validation, testing
        """
    #{{{    
        assert isinstance(mode, DataModes)

        self.mode = mode

        self.halo_catalog = HaloCatalog(mode)

        # total number of objects across all processes
        if mode is DataModes.TRAINING :
            self.len_all = len(self.halo_catalog)

        # now slice the object we use for this process
        self.halo_catalog = self.halo_catalog[cfg.rank::cfg.world_size]

        self.data_items = [DataItem(h, mode, **data_item_kwargs) for h in self.halo_catalog]

        if self.mode is DataModes.TRAINING :
            # check that our hack works
            assert len(self) % cfg.DATALOADER_ARGS['batch_size'] == 0

        if self.mode is not DataModes.TRAINING and cfg.TEST_ON_ALL :
            # construct the mapping from indices to halos and TNG chunks

            assert cfg.PRT_FRACTION['TNG'][str(self.mode)] > 1
            self.chunks = [] # each entry is (data_item_idx, chunk_idx)

            for data_item_idx, d in enumerate(self.data_items) :
                N_chunks = int(np.ceil(len(d.TNG_Pth) / cfg.PRT_FRACTION['TNG'][str(self.mode)]))
                for ii in range(N_chunks) :
                    self.chunks.append((data_item_idx, ii))
    #}}}


    def set_worker(self, seed) :
        """
        to be called from worker_init_fn to initialize the random number generator for this worker
        Will not initialize the RNG if we need reproducible results
        """
    #{{{
        if self.mode is DataModes.TRAINING or not cfg.TEST_ON_ALL :
            self.rng = np.random.default_rng(seed % 2**32)
        else :
            self.rng = None
    #}}}


    def __getitem__(self, idx) :
        """
        idx ... local index within the samples this rank operates on
                (in training mode, can be larger than the maximum index that can be applied to our halo
                 catalog since we sometimes use samples twice if their total number is not
                 divisible by the WORLD_SIZE)
        """
    #{{{
        if self.mode is DataModes.TRAINING or not cfg.TEST_ON_ALL :
            # the usual procedure we use during training: randomly sample the DM and TNG particles

            if idx >= len(self.data_items) :
                # we have hit some duplicate sample -- we take a random one now
                # in order to avoid always having the same duplicate samples in the training set
                # NOTE that at this point we have an rng available
                assert self.mode is DataModes.TRAINING
                idx = self.rng.integers(len(self.data_items))

            indices = dict(DM=self.__get_indices(self.data_items[idx].halo, 'DM'),
                           TNG=self.__get_indices(self.data_items[idx].halo, 'TNG'))

            return self.data_items[idx].sample_particles(indices,
                                                         TNG_residuals_noise_rng=self.rng \
                                                                                 if self.mode is DataModes.TRAINING \
                                                                                 else None,
                                                         local_rng=self.rng)

        else :
            # during testing, we need to be careful to do everything deterministically

            data_item_idx, chunk_idx = self.chunks[idx]

            idx_min = chunk_idx * cfg.PRT_FRACTION['TNG'][str(self.mode)]
            idx_max = min(((chunk_idx+1) * cfg.PRT_FRACTION['TNG'][str(self.mode)], \
                          len(self.data_items[data_item_idx].TNG_Pth)))

            indices = dict(DM=self.__get_indices(self.data_items[data_item_idx].halo, 'DM'),
                           TNG=np.arange(idx_min, idx_max, dtype=int))

            # for the local particles
            _rng = np.random.default_rng((cfg.TESTING_SEED + idx) % 2**32)

            return self.data_items[data_item_idx].sample_particles(indices,
                                                                   TNG_residuals_noise_rng=None,
                                                                   local_rng=_rng)
    #}}}


    def __len__(self) :
        """
        returns number of samples this rank operates on
        (in training mode, if the world size does not divide the available number of samples,
         we round up and some samples will be used twice)
        """
    #{{{
        if self.mode is DataModes.TRAINING :
            n = int(np.ceil(self.len_all / cfg.world_size))
            # return the next larger integer that is divisible by the batch size
            bs = cfg.DATALOADER_ARGS['batch_size']
            return n + bs - n % bs

        if cfg.TEST_ON_ALL :
            return len(self.chunks)

        return len(self.data_items)
    #}}}


    def __get_indices(self, halo, ptype) :
        """
        generates the randomly chosen index arrays that we use to choose particles
        halo  ... the halo to operate on
        ptype ... one of 'DM', 'TNG'
        """
    #{{{ 
        assert isinstance(halo, Halo)
        assert ptype in ['DM', 'TNG']

        # TODO this is the only place where we use Nprt and currently not even that,
        #      so the use of Nprt is not really tested and this field should be treated with caution
        Nprt = getattr(halo, 'Nprt_%d_%s'%(cfg.TNG_RESOLUTION, ptype))
        Nindices = int(cfg.PRT_FRACTION[ptype][str(self.mode)] * Nprt) \
                   if isinstance(cfg.PRT_FRACTION[ptype][str(self.mode)], float) \
                      and cfg.PRT_FRACTION[ptype][str(self.mode)]<=1 \
                   else int(cfg.PRT_FRACTION[ptype][str(self.mode)])

        # here we allow the possibility for duplicate entries
        # in practice, this should not be an issue and will make the sampling a lot faster
        # NOTE that we do not restrict to Nprt -- we can just as well mod this later.
        # NOTE if we replace by 2**32, it doesn't work anymore (only zeros) -- is this a numpy bug?
        #      Yes it is, should be fixed soon!
        if self.rng is not None :
            assert self.mode is DataModes.TRAINING or not cfg.TEST_ON_ALL
            _rng = self.rng
        else :
            assert ptype == 'DM'
            # always construct the same RNG for each halo so the same DM particles are sampled
            _rng = np.random.default_rng((cfg.TESTING_SEED + halo.idx) % 2**32)

        return _rng.integers(2**34, size=Nindices)
    #}}}
