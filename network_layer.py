import torch
import torch.nn as nn
import torch.linalg as LA

from network_mlp import NetworkMLP
import cfg


class NetworkLayer(nn.Module) :
    """
    one block of the network, transforms a set of vectors into another set of vectors
    """

    def __init__(self, k_in, k_out, Nglobals=0, **MLP_kwargs) :
        """
        Nk_in  ... number of feature vector coming in
                   (set to non-positive for the initial layer that takes the positions)
        Nk_out ... number of neurons going out
        """
    #{{{
        super().__init__()

        self.compute_dots = k_in > 0
        self.mlp = NetworkMLP(1 + self.compute_dots * k_in + Nglobals, k_out, **MLP_kwargs)
    #}}}


    def forward(self, x, u=None) :
        """
        x ... the input tensor, of shape [batch, Nvecs, 3]
        u ... the global tensor -- assumed to be a vector, i.e. of shape [batch, Nglobals]
        """
    #{{{
        # we know that the last dimension is the real space one
        scalars = LA.norm(x, dim=-1, keepdim=True)

        if self.compute_dots :
            # compute the mutual dot products
            dots = torch.sqrt( torch.einsum('bid,bjd->bij', x, x) )
            scalars = torch.cat((scalars, dots), dim=-1)
        else :
            # we are in the very first layer and need to normalize the vector
            x /= scalars

        if u is not None :
            scalars = torch.cat((u.unsqueeze(1).repeat(1,scalars.shape[1],1), scalars), dim=-1)

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
