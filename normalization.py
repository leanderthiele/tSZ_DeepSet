"""
A collection of small functions used to normalize various inputs.

It is important that these functions always return a different tensor
[i.e. if they implement the identity, they should call tens_copy()],
since the calling code may depend on this behaviour.
"""

import torch
import numpy as np

def choose_lib(obj) :
    """
    Utility function to allow concise operation on torch and numpy arrays
    """
    if isinstance(obj, torch.Tensor) :
        return torch
    return np


def tens_copy(obj) :
    if isinstance(obj, torch.Tensor) :
        return obj.clone()
    return obj.copy()



def local_N(x) :
    """
    transform the number of particles around a TNG position.
    Calibrated for cfg.R_LOCAL=100 kpc/h.
    """
    return (choose_lib(x).log(x) - 5.469) / 1.598


def unit_contraction(x) :
    """
    transform contractions between unit vectors.
    We assume this results in an approximately uniform distribution
    on [-1, 1] and simply want to fix the variance
    """
    return x / 0.58


def encoder_x(x) :
    """
    transform the modulus of the DM positions going into the encoder
    """
    # TODO
    return tens_copy(x)


def encoder_v(x) :
    """
    transform the modulus of the DM velocities going into the encoder
    """
    # TODO
    return tens_copy(x)


def TNG_radii(x) :
    """
    transform the TNG radii
    """
    # TODO
    return tens_copy(x)


def local_v0(x) :
    """
    transform the local bulk velocity moduli
    """
    # TODO
    return tens_copy(x)


def local_x(x) :
    """
    transform the local DM radii
    """
    # TODO
    return tens_copy(x)


def local_v(x) :
    """
    transform the local DM velocity moduli
    """
    # TODO
    return tens_copy(x)
