import numpy as np

import torch

from data_modes import DataModes
from data_item import DataItem
from global_fields import GlobalFields
from basis import Basis
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
        self.has_TNG_radii = data_items[0].has_TNG_radii
        assert all(self.has_TNG_radii == d.has_TNG_radii for d in data_items)
        self.mode = data_items[0].mode
        assert all(self.mode is d.mode for d in data_items)

        astensor = lambda x : torch.tensor(x, dtype=torch.float32, requires_grad=False)
        lengths_equal = lambda s : all(getattr(data_items[0], s).shape[0] == getattr(d, s).shape[0] \
                                       for d in data_items)
        stack_to_tensor = lambda s : astensor(np.stack([getattr(d, s) for d in data_items], axis=0))
        list_tensors = lambda s : [astensor(getattr(d, s)).unsqueeze(0) for d in data_items]
        batch = lambda s : stack_to_tensor(s) if lengths_equal(s) else list_tensors(s)

        if self.has_DM :
            self.DM_coords = batch('DM_coords')

        if self.has_TNG :
            self.TNG_coords = batch('TNG_coords')
            self.TNG_Pth = batch('TNG_Pth')
            if self.has_TNG_radii :
                self.TNG_radii = batch('TNG_radii')

        # the globals if provided
        if len(GlobalFields) != 0 :
            self.u = astensor(np.stack([GlobalFields(d.halo, \
                                                     rng=np.random.default_rng(d.hash % 2**32) \
                                                         if self.mode is DataModes.TRAINING else None) \
                                        for d in data_items], axis=0))

        # the basis vectors if provided
        if len(Basis) != 0 :
            self.basis = astensor(np.stack([Basis(d.halo) for d in data_items], axis=0))

        # halo properties for the SphericalModel
        halo_to_tensor = lambda s : astensor(np.array([getattr(d.halo, s) for d in data_items]))

        self.M200c = halo_to_tensor('M200c_DM')
        self.R200c = halo_to_tensor('R200c_DM')
        self.P200c = halo_to_tensor('P200c_DM')

        self.CM_DM   = halo_to_tensor('CM_DM')
        self.pos_DM  = halo_to_tensor('pos_DM')
        self.CM_TNG  = halo_to_tensor('CM_TNG')
        self.pos_TNG = halo_to_tensor('pos_TNG')
    #}}}


    def add_origin(self, origin, compute_TNG_radii=True) :
        """
        CAUTION this function modifies the instance in-place
                (it also returns the altered object)
        in the case when the origin was not known upon construction, we can pass it here now
        origin ... the origins we want to use, of shape [batch, 1, 3]
                   if normalization by R200c is used globally, the passed origins should
                   be in those normalized units
                   (this origin is in the coordinate system that was used to center the particles
                    initially)
        compute_TNG_radii ... whether the radial positions of the TNG particles
                              (with respect to the new origin)
        """
    #{{{
        self.DM_coords -= origin
        self.TNG_coords -= origin

        # NOTE technically speaking it would be necessary to impose periodic boundary conditions
        #      again here.
        #      However, the new origin is likely very close to the old one, so this is practically
        #      speaking never necessary.
        #      The good thing about this is that everything stays differentiable!

        if compute_TNG_radii :
            self.TNG_radii = torch.linalg.norm(self.TNG_coords, axis=-1, keepdims=True)
            self.has_TNG_radii = True

        return self
    #}}}


    def to_device(self) :
        """
        CAUTION this function modifies the instance in-place
                (it also returns the altered object)
        """
    #{{{
        if cfg.DEVICE_IDX is not None :

            for k, v in self.__dict__.items() :
                if k.startswith('__') :
                    continue
                elif isinstance(v, bool) :
                    continue
                elif isinstance(v, DataModes) :
                    continue
                elif isinstance(v, torch.Tensor) :
                    setattr(self, k, v.to(cfg.DEVICE_IDX))
                elif isinstance(v, list) and isinstance(v[0], torch.Tensor) :
                    setattr(self, k, [t.to(cfg.DEVICE_IDX) for t in v])
                else :
                    raise RuntimeError(f'Attribute {k} has unexpected type.')

        return self
    #}}}
