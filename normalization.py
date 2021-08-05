"""
A collection of small functions used to normalize various inputs
"""

import torch
import numpy as np

def choose_lib(obj) :
    """
    Utility function to allow concise operation on torch and numpy arrays
    """
    if isinstance(obj, torch.Tensor) :
        return torch
    else :
        return np


def local_N(N) :
    """
    transform the number of particles around a TNG position.
    Calibrated for cfg.R_LOCAL=100 kpc/h.
    """
    return (choose_lib(N).log(N) - 5.469) / 1.598


def unit_contraction(x) :
    """
    transform contractions between unit vectors.
    We assume this results in an approximately uniform distribution
    on [-1, 1] and simply want to fix the variance
    """
    return x / 0.58


def encoder_xnorm(x) :
    """
    transform the modulus of the dark matter positions going into the encoder
    """
    
