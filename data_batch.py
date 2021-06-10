import numpy as np

import torch

from data_item import DataItem
import cfg


class DataBatch :
    """
    represents a batch of DataItem(s)

    We use the constructor as the collate_fn in the DataLoader

    The initialized instance mirrors the structure of the DataItem class (with the same fields),
    but it has an additional 0th dimension everywhere representing the batch
    and tensors are now in torch
    """

    def __init__(self, data_items) :
    #{{{
        if isinstance(data_items, list) :
            assert all(isinstance(d) for d in data_items)
        else :
            data_items = [data_items, ]

        astensor = lambda x : torch.tensor(x, dtype=torch.float32)

        self.DM_in = astensor(np.stack((d.DM_in for d in data_items), axis=0))
        self.TNG_coords = astensor(np.stack((d.TNG_coords for d in data_items), axis=0))
        self.TNG_pressure = astensor(np.stack((d.TNG_pressure for d in data_items), axis=0))
    #}}}


    def to_device(self) :
        """
        CAUTION this function modifies the instance in-place
                (it also returns the altered object)
        """
    #{{{
        self.DM_in = self.DM_in.to(cfg.DEVICE_IDX)
        self.TNG_coords = self.TNG_coords.to(cfg.DEVICE_IDX)
        self.TNG_pressure = self.TNG_pressure.to(cfg.DEVICE_IDX)
        return self
    #}}}
