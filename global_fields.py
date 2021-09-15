import numpy as np
import numpy.linalg as LA

from halo import Halo
from fixed_len_vec import FixedLenVec
import cfg


class GlobalFields(np.ndarray, metaclass=FixedLenVec) :
    """
    a wrapper around the numpy.ndarray, constructible from a Halo instance
    bundles the global scalar features of the halo we want to use into a numpy array
    """

    # dict with all possible global scalars in correct order, values are their dimensionalities
    _FIELDS = {'logM': 1,
               'ang_mom': 1,
               'Xoff': 1,
               'Voff': 1,
               'CM': 1,
               'inertia': 3,
               'inertia_dot_ang_mom': 3,
               'vel_dispersion': 3,
               'vel_dispersion_dot_ang_mom': 3,
               'vel_dispersion_dot_inertia': 9, }


    def __new__(cls, halo, restrict=True, rng=None) :
        """
        constructing ndarray's is a bit tricky, so we do it this way
        halo ... the halo for which the global values are to be extracted
        retrict ... if True, the output is according to the values in cfg.GLOBALS_USE,
                    else it contains all the possible scalar fields
        rng  ... if passed, should be a numpy random number generator instance.
                 Using halo.dglobals as characteristic scale, this will be used
                 to add some noise to the global features and hopefully mitigate
                 overfitting.
        """
    #{{{
        assert isinstance(halo, Halo)

        # make sure no typos in the GLOBALS_USE dict
        assert set(cfg.GLOBALS_USE.keys()) == set(GlobalFields._FIELDS.keys())

        # logarithmic mass
        logM = np.log(halo.M200c)

        # magnitude of angular momentum vector
        ang_mom_norm = LA.norm(halo.ang_mom)

        # relaxation measures
        Xoff = halo.Xoff
        Voff = halo.Voff

        # center of mass (with respect to the pos field)
        CM_norm = LA.norm(halo.CM)

        # diagonalize inertia tensor
        eigval_inertia, eigvec_inertia = LA.eigh(halo.inertia)
        eigvec_inertia = eigvec_inertia.T # get into better format -- v[i, :] is the ith eigenvector

        # compute the angles between angular momentum and inertia axes
        # note that the eigenvectors returned by eigh are already normalized
        inertia_dot_ang_mom = eigvec_inertia @ halo.ang_mom / ang_mom_norm

        # diagonalize velocity dispersion tensor
        eigval_vel_dispersion, eigvec_vel_dispersion = LA.eigh(halo.vel_dispersion)
        eigvec_vel_dispersion = eigvec_vel_dispersion.T

        # compute the angles between angular momentum and velocity dispersion axes
        vel_dispersion_dot_ang_mom = eigvec_vel_dispersion @ halo.ang_mom / ang_mom_norm

        # compute angles between inertia and velocity dispersion eigenvectors
        vel_dispersion_dot_inertia = np.einsum('id,jd->ij', eigvec_inertia, eigvec_vel_dispersion).flatten()

        # fix the parity symmetry
        inertia_dot_ang_mom[inertia_dot_ang_mom < 0] *= -1
        vel_dispersion_dot_ang_mom[vel_dispersion_dot_ang_mom < 0] *= -1
        vel_dispersion_dot_inertia[vel_dispersion_dot_inertia < 0] *= -1

        # convert to self-similar units
        ang_mom_norm /= halo.R200c * halo.V200c * halo.M200c
        Xoff /= halo.R200c
        Voff /= halo.V200c
        CM_norm /= halo.R200c
        eigval_inertia *= cfg._UNIT_MASS / (halo.R200c**2 * halo.M200c)
        eigval_vel_dispersion *= cfg._UNIT_MASS / (halo.V200c**2 * halo.M200c)

        out = []

        out.append(logM)
        out.append(ang_mom_norm)
        out.append(Xoff)
        out.append(Voff)
        out.append(CM_norm)
        out.extend(eigval_inertia)
        out.extend(inertia_dot_ang_mom)
        out.extend(eigval_vel_dispersion)
        out.extend(vel_dispersion_dot_ang_mom)
        out.extend(vel_dispersion_dot_inertia)

        assert len(out) == cls.len_all()

        # NOTE this whole implementation is a bit inefficient,
        #      but we don't care

        # add noise if requested
        if rng is not None :

            assert np.all(halo.dglobals > 0)

            # we treat mass differently
            logM_idx = cls.indices_from_name('logM')
            assert len(logM_idx) == 1
            logM_idx = logM_idx[0]

            if cfg.GLOBALS_NOISE is not None :

                for ii in filter(lambda x: x != logM_idx, range(cls.len_all())) :

                    # generate Gaussian random number of unit variance and zero mean
                    r = rng.normal()

                    # rescale by the dglobals entry
                    r *= halo.dglobals[ii, 0 if r<0 else 1]

                    # add to the output with appropriate weight
                    out[ii] += r * cfg.GLOBALS_NOISE

            if cfg.MASS_NOISE is not None :
                
                r = rng.normal()

                r *= halo.dglobals[logM_idx, 0 if r<0 else 1]

                out[logM_idx] += r * cfg.MASS_NOISE

        # if halo has u_avg and u_std fields populated, normalize the output
        # (NOTE that the order is important here, we only normalize *after* adding the noise)
        if hasattr(halo, 'u_avg') and hasattr(halo, 'u_std') :
            for ii in range(cls.len_all()) :
                out[ii] = (out[ii] - halo.u_avg[ii]) / halo.u_std[ii]

        if restrict :
            # we restrict the output according to cfg.GLOBALS_USE
            out_restricted = []
            for k, _ in filter(lambda x: x[1], cfg.GLOBALS_USE.items()) :
                out_restricted.extend(out[jj] for jj in cls.indices_from_name(k))

        return np.array(out_restricted if restrict else out).view(type=cls)
    #}}}


    @classmethod
    def _length(cls) :
        """
        should not be used directly
        """
    #{{{
        return sum(cfg.GLOBALS_USE[k]*v for k, v in GlobalFields._FIELDS.items())
    #}}}


    @classmethod
    def len_all(cls) :
        """
        returns the total number of possible global scalars (irrespective of cfg.GLOBALS_USE
        """
    #{{{
        return sum(cls._FIELDS.values())
    #}}}


    @classmethod
    def indices_from_name(cls, name) :
        """
        returns list of indices corresponding to the name which is a key in _FIELDS
        """
    #{{{
        # note that dicts are ordered in reasonable python
        fields_list = list(cls._FIELDS.keys())

        start_idx = sum(cls._FIELDS[k] for k in fields_list[:fields_list.index(name)])
        end_idx = start_idx + cls._FIELDS[name]

        return list(range(start_idx, end_idx))
    #}}}
