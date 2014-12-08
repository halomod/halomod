'''
Created on 08/12/2014

@author: Steven
'''
import copy
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from hmf.tools import growth_factor
import numpy as np
import sys

def get_cm(name, **kwargs):
    """
    Returns the correct subclass of :class:`CMRelation`.
    
    Parameters
    ----------
    name : str
        The class name of the appropriate fit
        
    \*\*kwargs : 
        Any parameters for the instantiated fit (including model parameters)
    """
    try:
        return getattr(sys.modules[__name__], name)(**kwargs)
    except AttributeError:
        raise AttributeError(str(name) + "  is not a valid CMRelation class")


class CMRelation(object):
    r"""
    Base-class for Concentration-Mass relations
    
    This class should not be called directly, rather use a subclass which is 
    specific to a certain relation. 
    """
    _pdocs = \
    """
    
    Parameters
    ----------
    M   : array
        A vector of halo masses [units M_sun/h]
        
    nu2  : array
        A vector of peak-heights, :math:`\delta_c^2/\sigma^2` corresponding to ``M``
        
    z   : float, optional
        The redshift. 
        
    delta_halo : float, optional
        The overdensity of the halo w.r.t. the mean density of the universe.
        
    cosmo : :class:`cosmo.Cosmology` instance, optional
        A cosmology. Default is the default provided by the :class:`cosmo.Cosmology`
        class. Not required if ``omegam_z`` is passed.
         
    omegam_z : float, optional
        A value for the mean matter density at the given redshift ``z``. If not
        provided, will be calculated using the value of ``cosmo``. 
        
    \*\*model_parameters : unpacked-dictionary
        These parameters are model-specific. For any model, list the available
        parameters (and their defaults) using ``<model>._defaults``
        
    """
    __doc__ += _pdocs
    _defaults = {}

    use_cosmo = False
    def __init__(self, filter, delta_c, z, cdict, m_hm, **model_parameters):
        """
        filter : a filter function at z = 0
        """
        # Check that all parameters passed are valid
        for k in model_parameters:
            if k not in self._defaults:
                raise ValueError("%s is not a valid argument for the %s CMRelation" % (k, self.__class__.__name__))

        # Gather model parameters
        self.params = copy.copy(self._defaults)
        self.params.update(model_parameters)

        # Save instance variables
        self.filter = filter
        self.delta_c = delta_c
        self.z = z
        self.cdict = cdict
        self.m_hm = m_hm

class Bullock01(CMRelation):
    _defaults = {"F":0.001, "K":3.4}

    def zc(self, m):

        ms = self.params["F"] * m
        rs = self.filter.mass_to_radius(ms)
        sigma = self.filter.sigma(rs)
        g = growth_factor(100.0, self.cdict, True)
        s = spline(g[1, :], g[0, :])
        zc = s(self.delta_c / sigma)
        zc[zc < 0] = 0.0  # hack?
        return zc

    def cm(self, m):
        return self.params["K"] * (self.zc(m) + 1.0) / (self.z + 1.0)


class BullockWDM(Bullock01):
    _defaults = {"F":0.001, "K":3.4, "m_hm":1e10,
                 "g1":15, "g2":0.3}
    def cm(self, m):
        cm = super(BullockWDM, self).cm(m)
        return cm * (1 + self.params['g1'] * self.m_hm / m) ** (-self.params["g2"])
