import numpy as np
import numpy.linalg as LA

from halo import Halo
from fixed_len_vec import FixedLenVec
import cfg


class Basis(np.ndarray, metaclass=FixedLenVec) :
    """
    a wrapper around the numpy.ndarray, constructible from a Halo instance
    bundles the global vectorial features of the halo we want to use into a numpy array
    It is a principled idea to return the unit vectors only, the magnitudes can be passed
    through the global fields.
    """

    def __new__(cls, halo, rng=None) :
        """
        constructing ndarray's is a bit tricky, so we do it this way
        halo ... the halo for which the global values are to be extracted
        rng  ... if passed, should be a numpy random number generator instance.
                 Using cfg.BASIS_NOISE as scale, vectors will be randomly rotated
        """
    #{{{    
        assert isinstance(halo, Halo)

        # make sure no typos in BASIS_USE dict
        assert set(cfg.BASIS_USE.keys()) == {'ang_mom', 'CM', 'inertia', 'vel_dispersion'}

        _, eigvec_inertia = LA.eigh(halo.inertia)
        eigvec_inertia = eigvec_inertia.T # more intuitive

        _, eigvec_vel_dispersion = LA.eigh(halo.vel_dispersion)
        eigvec_vel_dispersion = eigvec_vel_dispersion.T

        # choose the octant of the coordinate system where the angular momentum points
        # this fixes the coordinate system uniquely since the eigenvectors returned by eigh
        # are already ordered according to the eigenvalue magnitudes
        eigvec_inertia[eigvec_inertia @ halo.ang_mom < 0] *= -1
        eigvec_vel_dispersion[eigvec_vel_dispersion @ halo.ang_mom < 0] *= -1


        out = []

        if cfg.BASIS_USE['ang_mom'] :
            out.append(halo.ang_mom)
        if cfg.BASIS_USE['CM'] :
            out.append(halo.CM)
        if cfg.BASIS_USE['inertia'] :
            out.extend(eigvec_inertia)
        if cfg.BASIS_USE['vel_dispersion'] :
            out.extend(eigvec_vel_dispersion)

        out = np.array(out)

        # normalize -- we want to control separately if we pass the magnitudes separately
        #              as part of the GlobalFields instances
        out /= LA.norm(out, axis=-1, keepdims=True)

        # add noise if requested
        if rng is not None and cfg.BASIS_NOISE is not None :
            
            for ii in range(len(out)) :
                
                # generate random unit vector around z axis
                theta = rng.normal() * cfg.BASIS_NOISE * np.pi/180
                phi = rng.random() * 2*np.pi

                # find the rotation matrix that takes the z axis to the vector we want to perturb
                # [this is analogous to https://stackoverflow.com/questions/45142959]
                k = np.array([[0,          0,         out[ii,0]],
                              [0,          0,         out[ii,1]],
                              [-out[ii,0], -out[ii,1],0        ]])
                R = np.identity(3) + k + (1-out[ii,2]) / (out[ii,0]**2+out[ii,1]**2) * k @ k

                assert np.allclose(R @ np.array([0,0,1]), out[ii])

                out[ii, ...] = R @ np.array([np.sin(theta) * np.sin(phi),
                                             np.sin(theta) * np.cos(phi),
                                             np.cos(theta)              ])

        return out.view(type=cls)
    #}}}

    
    @classmethod
    def _length(cls) :
        """
        should not be used directly, adapt if more features are added to the global vector
        """
    #{{{ 
        return cfg.BASIS_USE['ang_mom'] \
               + cfg.BASIS_USE['CM'] \
               + 3 * cfg.BASIS_USE['inertia'] \
               + 3 * cfg.BASIS_USE['vel_dispersion']
    #}}}
