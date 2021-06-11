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
        stack_to_tensor = lambda s : astensor(np.stack((getattr(d, s) for d in data_items), axis=0))

        self.DM_in = stack_to_tensor('DM_in')
        self.TNG_coords = stack_to_tensor('TNG_coords')
        self.TNG_radii = stack_to_tensor('TNG_radii')
        self.TNG_Pth = stack_to_tensor('TNG_Pth')

        # the globals
        self.u = astensor(np.stack((np.array([np.log(d.halo.M200c_DM), ]) for d in data_items), axis=0))

        # halo properties for the SphericalModel
        halo_to_tensor = lambda s : astensor(np.array([getattr(d.halo, s) for d in data_items]))

        self.M200c = halo_to_tensor('M200c_DM')
        self.R200c = halo_to_tensor('R200c_DM')
        self.P200c = halo_to_tensor('P200c_DM')
    #}}}


    def to_device(self) :
        """
        CAUTION this function modifies the instance in-place
                (it also returns the altered object)
        """
    #{{{
        if cfg.DEVICE_IDX is not None :
            self.DM_in = self.DM_in.to(cfg.DEVICE_IDX)
            self.TNG_coords = self.TNG_coords.to(cfg.DEVICE_IDX)
            self.TNG_Pth = self.TNG_Pth.to(cfg.DEVICE_IDX)
            self.u = self.u.to(cfg.DEVICE_IDX)
            self.M200c = self.M200c.to(cfg.DEVICE_IDX)
            self.R200c = self.R200c.to(cfg.DEVICE_IDX)
            self.P200c = self.P200c.to(cfg.DEVICE_IDX)

        return self
    #}}}
