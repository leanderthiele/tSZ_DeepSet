import torch
import torch.nn as nn

from network_mlp import NetworkMLP
import cfg


class NetworkLayer(nn.Module) :
    """
    one block of the network, transforms a set of vectors into another set of vectors
    """

    def __init__(self, k_in, k_out, **MLP_kwargs) :
        """
        Nk_in  ... number of feature vectors coming in
                   (set to non-positive for the initial layer that takes the positions)
        Nk_out ... number of neurons going out
        Nbasis ... number of basis vectors that will be passed
        """
    #{{{
        super().__init__()

        self.compute_dots = k_in > 0
        self.mlp = NetworkMLP(1 + cfg.NBASIS + self.compute_dots * k_in + cfg.NGLOBALS, k_out, **MLP_kwargs)
    #}}}


    def forward(self, x, u=None, basis=None) :
        """
        x     ... the input tensor, of shape [batch, Nvecs, 3]
                  or a list of length batch with shapes [1, Nvecs[ii], 3]
        u     ... the global tensor -- assumed to be a vector, i.e. of shape [batch, Nglobals]
        basis ... the basis vectors to use -- either None if no basis is provided
                  or of shape [batch, Nbasis, 3]
        """
    #{{{
        if isinstance(x, list) :
            return torch.cat((self(xi, u, basis[ii, ...].unsqueeze(0) if basis is not None else basis)
                              for ii, xi in enumerate(x)))

        # we know that the last dimension is the real space one
        scalars = torch.linalg.norm(x, dim=-1, keepdim=True)

        # concatenate with the basis projections if required
        if basis is not None :
            projections = torch.einsum('bid,bnd->bin', x, basis)
            scalars = torch.cat((scalars, projections), dim=-1)

        if self.compute_dots :
            # compute the mutual dot products
            dots = torch.sqrt( torch.einsum('bid,bjd->bij', x, x) )
            scalars = torch.cat((scalars, dots), dim=-1)
        else :
            # we are in the very first layer and need to normalize the vector
            x /= scalars

        # concatenate with the global vector if requested
        if u is not None :
            assert len(u) == cfg.NGLOBALS
            scalars = torch.cat((u.unsqueeze(1).repeat(1,scalars.shape[1],1), scalars), dim=-1)
        else :
            assert not cfg.NGLOBALS

        # pass through the MLP, transform scalars -> scalars
        fk = self.mlp(scalars)

        # evaluate in the vector space
        vecs = torch.einsum('bio,bid->biod', fk, x)

        # return the pooled version
        return self.__pool(vecs)
    #}}}


    def __pool(self, x) :
        """
        input tensor is of shape [batch, input Nvecs, output Nvecs, 3]
        """
    #{{{
        return torch.mean(x, dim=1)
    #}}}
