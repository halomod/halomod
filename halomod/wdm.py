'''
Contains WDM versions of all models and frameworks
'''
from concentration import CMRelation
from halo_model import HaloModel
from hmf._cache import cached_property
from copy import deepcopy
from hmf.filters import TopHat
import numpy as np
from scipy import integrate as intg
from hmf.wdm import MassFunctionWDM
from hmf._framework import get_model
import sys

#===============================================================================
# C-M relations
#===============================================================================
def CMRelationWDMRescaled(name,**kwargs):
    """
    Class factory for Rescaled CM relations.
    """
    x = getattr(sys.modules["halomod.concentration"], name)

    # class CMRelationWDMRescale(x):
    #     """
    #     Base class for simply rescaled concentration-mass relations (cf. Schneider
    #     2013, Bose+15)
    #     """
    def __init__(self, m_hm, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.m_hm = m_hm

    def cm(self, m):
        cm = super(self.__class__, self).cm(m)
        g1 = self.params['g1']
        g2 = self.params['g2']
        b0 = self.params['beta0']
        b1 = self.params['beta1']
        return cm * (1 + g1 * self.m_hm / m) ** (-g2)*(1+self.z)**(b0*self.z-b1)


    K = type(name + "WDM", (x,),{})
    K._defaults.update({"g1":60, "g2":0.17,"beta0":0.026,"beta1":0.04})

    K.__init__ = __init__
    K.cm = cm
    return K(**kwargs)

#===============================================================================
# Framework
#===============================================================================
class HaloModelWDM(HaloModel, MassFunctionWDM):
    """
    This class is a derivative of HaloModel which sets a few defaults that make
    more sense for a WDM model, and also implements the framework to include a
    smooth component
    """

    def __init__(self, **kw):
        kw.setdefault("cm_relation", "DuffyWDM")
        super(HaloModelWDM, self).__init__(**kw)

    @cached_property("_wdm", "dlog10m")
    def M(self):
        """ Halo masses in M_sun/h """
        return 10 ** np.arange(self._wdm.m_fs, 18, self.dlog10m) * u.MsolMass / self._hunit

    @cached_property("matter_power")
    def p_lin(self):
        """ The Schneider12 version of P_Lin = P_halofit * W(kR)"""
        f = TopHat(None, None, None, None)
        return np.exp(self.matter_power) * f.k_space(np.exp(self.lnk) * 2.0)

    @cached_property("rho_gtm", "mean_dens", "_wdm")
    def f_halos(self):
        """
        The total fraction of mass bound up in halos
        """
        return self.rho_gtm[0] / self.mean_dens

    @cached_property("f_halos", "power_ss", "power_sh", "power_hh")
    def power_matter(self):
        return (1 - self.f_halos) ** 2 * self.power_ss + \
            2 * (1 - self.f_halos) * self.f_halos * self.power_sh + \
            self.f_halos ** 2 * self.power_hh

    @cached_property("power_m_1h", "power_m_2h")
    def power_hh(self):
        return self.power_m_1h + self.power_m_2h

    @cached_property("_wdm", "dndm", "M", "lnk", "profile", "dlog10m", "f_halos", "mean_dens")
    def power_m_1h(self):
        integrand = self.dndm_rescaled * self.M ** 3
        u = self.profile.u(self.k), self.M, norm="m") ** 2
        out = np.zeros_like(self.k)
        for i, k in enumerate(self.k):
            r = (np.pi / k)  # half the radius
            mmin = self.filter_mod.radius_to_mass(r)
            if np.any(self.M > mmin):
                integ = integrand[self.M > mmin] * u[i, self.M > mmin]
                out[i] = intg.simps(integ, dx=self.dlog10m) * np.log(10)
            else:
                out[i] = 0.0
        return out / (self.f_halos * self.mean_dens) ** 2

    @cached_property("dndm", "_wdm", "bias", "lnk", "M", "profile", "f_halos", "mean_dens",
                     "matter_power")
    def power_m_2h(self):
        # Only use schneider for now
        integrand = self.M ** 2 * self.dndm_rescaled * self.bias
        out = np.zeros_like(self.k)
        u = self.profile.u(np.exp(self.k), self.M, norm="m")
        for i, k in enumerate(self.k):
            integ = integrand * u[i, :]
            out[i] = (intg.simps(integ, dx=self.dlog10m) * np.log(10))
        return self.p_lin * out ** 2 / (self.f_halos * self.mean_dens) ** 2

    @cached_property("dndm", "_wdm", "lnk", "M", "profile", "f_halos", "mean_dens",
                     "bias", "bias_smooth")
    def power_sh(self):
        integrand = self.M ** 2 * self.dndm_rescaled * self.bias
        out = np.zeros_like(self.k)
        u = self.profile.u(self.k, c.M, norm="m")
        for i, k in enumerate(self.k):
            integ = integrand * u [i, :]
            out[i] = intg.simps(integ, dx=self.dlog10m) * np.log(10)

        return self.bias_smooth * self.p_lin * out / (self.f_halos * self.mean_dens)

    @cached_property("bias_smooth", "matter_power")
    def power_ss(self):
        return self.bias_smooth ** 2 * np.exp(self.nonlinear_power)  # self.p_lin

    @cached_property("dndm", "bias", "M", "_wdm", "mean_dens", "f_halos")
    def bias_smooth(self):
        integrand = self.M ** 2 * self.dndm_rescaled * self.bias
        integral = intg.simps(integrand, dx=self.dlog10m) * np.log(10)
        return (1 / (1 - self.f_halos)) * (1 - integral / self.mean_dens)

    @cached_property("_wdm")
    def cm(self):
        kwargs = dict(nu=self.nu, z=self.z, growth=self.growth_model,
                      M=self.M, **self.cm_params)
        if np.issubclass_(self.cm_relation,CMRelation):
            if self.cm_relation.__class__.__name__.endswith("WDM"):
                cm = self.cm_relation(m_hm=self._wdm.m_hm, **kwargs)
            else:
                cm = self.cm_relation(**kwargs)
        elif self.cm_relation.endswith("WDM"):
            cm = CMRelationWDMRescaled(self.cm_relation[:-3],m_hm=self._wdm.m_hm, **kwargs)
        else:
            cm = get_model(self.cm_relation, "halomod.concentration", **kwargs)

        return cm
