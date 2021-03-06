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
        self.has_DM_velocities = data_items[0].DM_vels is not None
        assert all(self.has_DM_velocities == (d.DM_vels is not None) for d in data_items)
        self.has_TNG = data_items[0].has_TNG
        assert all(self.has_TNG == d.has_TNG for d in data_items)
        self.has_TNG_residuals = data_items[0].has_TNG_residuals
        assert all(self.has_TNG_residuals == d.has_TNG_residuals for d in data_items)
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
            if self.has_DM_velocities :
                self.DM_vels = batch('DM_vels')
            else :
                self.DM_vels = None

        if self.has_TNG :
            self.TNG_coords = batch('TNG_coords')
            self.TNG_Pth = batch('TNG_Pth')
            self.TNG_radii = batch('TNG_radii')
            if self.has_TNG_residuals :
                self.TNG_residuals = batch('TNG_residuals')

        if self.has_DM and self.has_TNG and cfg.NET_ARCH['local'] :
            self.DM_N_local = batch('DM_N_local')
            self.DM_coords_local = batch('DM_coords_local')
            if self.has_DM_velocities :
                self.DM_vels_local = batch('DM_vels_local')
            else :
                self.DM_vels_local = None

        # get a random number generator if noise is required
        # NOTE that this RNG is not used when testing, so no reproducibility required here
        rng = np.random.default_rng(sum(d.hash for d in data_items) % 2**32) if self.mode is DataModes.TRAINING else None

        # the globals if provided
        if len(GlobalFields) != 0 :
            self.u = astensor(np.stack([GlobalFields(d.halo, rng=rng) for d in data_items], axis=0))
        else :
            self.u = None

        # the basis vectors if provided
        if len(Basis) != 0 :
            self.basis = astensor(np.stack([Basis(d.halo, rng=rng) for d in data_items], axis=0))
        else :
            self.basis = None

        # halo properties for the SphericalModel
        halo_to_tensor = lambda s : astensor(np.array([getattr(d.halo, s) for d in data_items]))

        self.M200c = halo_to_tensor('M200c')
        self.R200c = halo_to_tensor('R200c')
        self.P200c = halo_to_tensor('P200c')

        # store the offset scale for the origin network
        self.Xoff = halo_to_tensor('Xoff')

        # store the halo position
        self.pos  = halo_to_tensor('pos')

        # store the halo indices in the global data set, useful for debugging
        self.idx = [d.halo.idx for d in data_items]

    #}}}


    def __len__(self) :
        """
        Idiomatic way to get the batch size
        """
    #{{{
        return len(self.idx)
    #}}}


    def add_origin(self, origin) :
        """
        CAUTION this function modifies the instance in-place
                (it also returns the altered object)
        in the case when the origin was not known upon construction, we can pass it here now
        origin ... the origins we want to use, of shape [batch, 1, 3]
                   if normalization by R200c is used globally, the passed origins should
                   be in those normalized units
                   (this origin is in the coordinate system that was used to center the particles
                    initially)
        """
    #{{{
        if cfg.ORIGIN_SHIFT_DM :
            self.DM_coords = self.DM_coords - origin

        self.TNG_coords = self.TNG_coords - origin

        # NOTE technically speaking it would be necessary to impose periodic boundary conditions
        #      again here.
        #      However, the new origin is likely very close to the old one, so this is practically
        #      speaking never necessary.
        #      The good thing about this is that everything stays differentiable!

        self.TNG_radii = torch.linalg.norm(self.TNG_coords, dim=-1, keepdim=True)

        return self
    #}}}


    def to_device(self) :
        """
        CAUTION this function modifies the instance in-place
                (it also returns the altered object)
        """
    #{{{
        if cfg.device_idx is not None :

            for k, v in vars(self).items() :
                if k.startswith('__') :
                    # make sure we don't do anything to any internals
                    continue

                if isinstance(v, torch.Tensor) :
                    setattr(self, k, v.to(cfg.device_idx))
                elif isinstance(v, list) and isinstance(v[0], torch.Tensor) :
                    setattr(self, k, [t.to(cfg.device_idx) for t in v])
                else :
                    # not a valid torch tensor or list thereof, don't care
                    continue

        return self
    #}}}
