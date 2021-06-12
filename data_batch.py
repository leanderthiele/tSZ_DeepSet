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
            assert all(isinstance(d, DataItem) for d in data_items)
        else :
            assert isinstance(data_items, DataItem)
            data_items = [data_items, ]

        self.has_DM = data_items[0].has_DM
        assert all(self.has_DM == d.has_DM for d in data_items)
        self.has_TNG = data_items[0].has_TNG
        assert all(self.has_TNG == d.has_TNG for d in data_items)

        astensor = lambda x : torch.tensor(x, dtype=torch.float32, requires_grad=False)
        lengths_equal = lambda s : all(getattr(data_items[0], s).shape[0] == getattr(d, s).shape[0] for d in data_items)
        stack_to_tensor = lambda s : astensor(np.stack([getattr(d, s) for d in data_items], axis=0))
        list_tensors = lambda s : [astensor(getattr(d, s)).unsqueeze(0) for d in data_items]
        batch = lambda s : stack_to_tensor(s) if lengths_equal(s) else list_tensors(s)

        if self.has_DM :
            self.DM_in = batch('DM_in')

        if self.has_TNG :
            self.TNG_coords = batch('TNG_coords')
            self.TNG_radii = batch('TNG_radii')
            self.TNG_Pth = batch('TNG_Pth')

        # the globals
        self.u = astensor(np.stack([np.array([np.log(d.halo.M200c_DM), ]) for d in data_items], axis=0))

        # the basis vectors if provided
        if cfg.NBASIS != 0 :
            self.basis = batch('basis')

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

            # TODO is this correct -- is torch.Tensor the base class?
            for k, v in self.__dict__.items() :
                if k.startswith('__') :
                    continue
                elif isinstance(v, torch.Tensor) :
                    setattr(self, k, v.to(cfg.DEVICE_IDX))
                elif isinstance(v, list) and isinstance(v[0], torch.Tensor) :
                    setattr(self, k, [t.to(cfg.DEVICE_IDX) for t in v])
                else :
                    raise RuntimeError(f'Attribute {k} has unexpected type.')

        return self
    #}}}
