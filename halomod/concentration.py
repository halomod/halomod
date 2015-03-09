'''
Created on 08/12/2014

@author: Steven
'''
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import numpy as np
from hmf._framework import Model

class CMRelation(Model):
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
    def __init__(self, nu, z, growth, M, **model_parameters):
        """
        filter : a filter function at z = 0
        """
        # Save instance variables
        self.nu = nu
        self.z = z
        self.growth = growth
        self.M = M

        super(CMRelation, self).__init__(**model_parameters)

class Bullock01(CMRelation):
    _defaults = {"F":0.001, "K":3.4}

    def zc(self, m):
        g = self.growth.growth_factor_fn(inverse=True)
        zc = g(np.sqrt(self.nu))
        zc[zc < 0] = 0.0  # hack?
        return zc

    def cm(self, m):
        return self.params["K"] * (self.zc(m) + 1.0) / (self.z + 1.0)

class Cooray(CMRelation):
    _defaults = {"a":9.0, "b":0.13, "c":1.0, "ms":None}
    def ms(self):
        d = self.nu[1:] - self.nu[:-1]
        try:
            # this to start below "saturation level" in sharp-k filters.
            pos = np.where(d < 0)[0][-1]
        except IndexError:
            pos = 0
        nu = self.nu[pos:]
        ms = self.M[pos:]
        s = spline(nu, ms)
        return s(1.0)

    def cm(self, m):
        ms = self.params['ms'] or self.ms()
        return self.params['a'] / (1 + self.z) ** self.params['c'] * (ms / m) ** self.params['b']

class Duffy(Cooray):
    _defaults = {"a":6.71, "b":0.091, "c":0.44, "ms":2e12}
