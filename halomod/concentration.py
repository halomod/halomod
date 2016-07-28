'''
Created on 08/12/2014

@author: Steven
'''
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import numpy as np
from hmf._framework import Component
from scipy.optimize import minimize
import warnings

class CMRelation(Component):
    r"""
    Base-class for Concentration-Mass relations
    """
    _pdocs = \
    """

    Parameters
    ----------
    filter0 : :class:`hmf.filters.Filter` instance
        An instance of a filter function, with the power specified at z=0.
        Required for ``Bullock01``.

    mean_density0 : float
        Mean density of the universe at z=0
        Required for ``Bullock01``

    growth : :class:`hmf.growth_factor.GrowthFactor` instance
        Specifies the growth function for the cosmology.
        Required for ``Bullock01``

    delta_c : float, optional
        Critical density for collapse
        Used in ``Bullock01``

    mstar : float, optional
        The nonlinear mass at the desired redshift.
        If not provided, will be calculated if required.

    \*\*model_parameters : unpacked-dictionary
        These parameters are model-specific. For any model, list the available
        parameters (and their defaults) using ``<model>._defaults``

    """
    __doc__ += _pdocs
    _defaults = {}

    use_cosmo = False
    def __init__(self, filter0=None, mean_density0=None, growth=None,delta_c=1.686,
                 **model_parameters):
        # Save instance variables
        self.filter = filter0
        self.growth = growth
        self.mean_density0 = mean_density0
        self.delta_c=delta_c
        super(CMRelation, self).__init__(**model_parameters)

    def mass_nonlinear(self,z):
        """
        Return the nonlinear mass at z.

        Parameters
        ----------
        z : float
            Redshift. Must not be an array.
        """
        model = lambda lnr : (self.filter.sigma(np.exp(lnr))*self.growth.growth_factor(z) - self.delta_c)**2

        res = minimize(model,[1.0,])

        if res.success:
            r = np.exp(res.x[0])
            return self.filter.radius_to_mass(r,self.mean_density0*(1+z)**3)
        else:
            warnings.warn("Minimization failed :(")
            return 0

#class NFW(CMRelation):
#    _defaults = {'f':,"k":}

class Bullock01(CMRelation):
    _defaults = {"F":0.01, "K":3.4}

    def zc(self,m,z=0):
        r = self.filter.mass_to_radius(self.params["F"]*m,self.mean_density0)
        nu = self.filter.nu(r,self.delta_c)
        g = self.growth.growth_factor_fn(inverse=True)
        zc = g(np.sqrt(nu))
        zc[zc < z] = z  # hack?
        return zc

    def cm(self, m,z=0):
        return self.params["K"] * (self.zc(m,z) + 1.0) / (z + 1.0)

class Bullock01_Power(CMRelation):
    _defaults = {"a":9.0, "b":-0.13, "c":1.0, "ms":None}

    def cm(self, m,z=0):
        ms = self.params['ms']  or self.mass_nonlinear(z)
        return self.params['a'] / (1 + z) ** self.params['c'] * (m / ms) ** self.params['b']

class Duffy08(Bullock01_Power):
    _defaults = {"a":6.71, "b":-0.091, "c":0.44, "ms":2e12}

class Zehavi11(Bullock01_Power):
    _defaults = {"a":11.0, "b":-0.13, "c":1.0, "ms":2.26e12}
