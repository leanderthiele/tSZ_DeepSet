from enum import Enum, auto

import numpy as np

import cfg

class DataModes(Enum) :
    """
    the three possible modes for which we may want to load data
    """

    TRAINING = auto()
    VALIDATION = auto()
    TESTING = auto()


    def __str__(self) :
        """
        a nice human readable representation
        """
    #{{{
        if self is DataModes.TRAINING :
            return 'training'
        elif self is DataModes.VALIDATION :
            return 'validation'
        elif self is DataModes.TESTING :
            return 'testing'
    #}}}


    def sample_indices(self) :
        """
        returns the indices in the halo catalog that are used for this data mode
        """
    #{{{
        with np.load(cfg.HALO_CATALOG) as f :
            Nsamples_tot = len(f['idx_DM'])
        
        try :
            Nsamples = int(round(Nsamples_tot * cfg.SAMPLE_FRACTIONS[str(self)]))
        except KeyError : # we have hit the one that is not specified
            Nsamples = Nsamples_tot - sum(int(round(Nsamples_tot * v)) for v in cfg.SAMPLE_FRACTIONS.values())

        return np.random.default_rng(cfg.CONSISTENT_SEED).choice(np.arange(Nsamples_tot),
                                                                 size=Nsamples, replace=False)
    #}}}
