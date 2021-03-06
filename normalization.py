"""
A collection of small functions used to normalize various inputs.

It is important that these functions always return a different tensor
[i.e. if they implement the identity, they should call tens_copy()],
since the calling code may depend on this behaviour.
"""

import torch
import numpy as np

import cfg

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
    Calibrated at cfg.R_LOCAL=100 kpc/h, assumes N is distributed
    Poissonian with mean <N> \propto R^3 >> 1
    TODO scale the standard deviation too
    """
    mu = 5.469 + 3.0 * np.log(cfg.R_LOCAL / 100.0)
    sigma = 1.598
    return (choose_lib(x).log(x) - mu) / sigma


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
    # TODO this is not ideal
    return (x - 0.97) / 0.70


def encoder_v(x) :
    """
    transform the modulus of the DM velocities going into the encoder
    """
    return (x - 1.02) / 0.48


def TNG_radii(x) :
    """
    transform the TNG radii (passed in units of R200c)
    """
    return (x**3 - 4.0) / 2.3


def local_v0(x) :
    """
    transform the local bulk velocity moduli
    """
    return (x - 0.69) / 0.47


def local_x(x) :
    """
    transform the local DM radii
    """
    return (x**3 - 0.50) / 0.29


def local_v(x) :
    """
    transform the local DM velocity moduli
    """
    return (x - 0.92) / 0.56
