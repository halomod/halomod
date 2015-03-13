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

#===============================================================================
# C-M relations
#===============================================================================
class CMRelationWDMRescale(CMRelation):
    """
    Base class for simply rescaled concentration-mass relations (cf. Schneider 
    2013)
    """
    def __init__(self, m_hm, **kwargs):
        super(CMRelationWDMRescale, self).__init__(**kwargs)
        self.m_hm = m_hm

    def cm(self, m):
        cm = super(self.__class__, self).cm(m)
        return cm * (1 + self.params['g1'] * self.m_hm / m) ** (-self.params["g2"])

def get_cm_rescaled(name):
    x = get_model(name, "halomod.concentration")
    K = type(name + "WDM", (x, CMRelationWDMRescale))
    K._defaults.update({"g1":15, "g2":0.3})
    return K

# class BullockWDM(Bullock01, CMRelationWDMRescale):
#     _defaults = {"F":0.001, "K":3.4, "m_hm":1e10,
#                  "g1":15, "g2":0.3}
#
# class CoorayWDM(Cooray, CMRelationWDMRescale):
#     _defaults = {"a":9.0, "b":0.13,
#                  "g1":15, "g2":0.3}
#
# class DuffyWDM(Duffy, CMRelationWDMRescale):
#     _defaults = {"a":6.71, "b":0.091, "c":0.44, "ms":2e12, "g1":15, "g2":0.3}

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
        kw.setdefault("cm_relation", "BullockWDM")
        super(HaloModelWDM, self).__init__(**kw)

    @cached_property("matter_power")
    def p_lin(self):
        f = TopHat(None, None, None, None)
        return np.exp(self.matter_power) * f.k_space(np.exp(self.lnk) * 2.0)

    @cached_property("rho_gtm", "mean_dens", "_wdm")
    def f_halos(self):
        """
        The total fraction of mass bound up in halos
        """
        mmin = self._wdm.m_fs
        # TODO: This is a real hack and should be modified
        c = deepcopy(self)
        c.update(hod_params={"M_min":np.log10(mmin)})
        return c.rho_gtm[0] / c.mean_dens

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
        # Again, a horrid hack.
        c = deepcopy(self)
        c.update(hod_params={"M_min":np.log10(self._wdm.m_fs)})
        integrand = c.dndm_rescaled * c.M ** 3
        u = c.profile.u(np.exp(c.lnk), c.M, norm="m") ** 2
        out = np.zeros_like(self.lnk)
        for i, k in enumerate(np.exp(self.lnk)):
            r = (np.pi / k)  # half the radius
            mmin = self.filter_mod.radius_to_mass(r)
            if np.any(c.M > mmin):
                integ = integrand[c.M > mmin] * u[i, c.M > mmin]
                out[i] = intg.simps(integ, dx=self.dlog10m) * np.log(10)
            else:
                out[i] = 0.0
        return out / (self.f_halos * self.mean_dens) ** 2

    @cached_property("dndm", "_wdm", "bias", "lnk", "M", "profile", "f_halos", "mean_dens",
                     "matter_power")
    def power_m_2h(self):
        # Only use schneider for now
        c = deepcopy(self)
        c.update(hod_params={"M_min":np.log10(self._wdm.m_fs)})
        integrand = c.M ** 2 * c.dndm_rescaled * c.bias
        out = np.zeros_like(self.lnk)
        u = c.profile.u(np.exp(c.lnk), c.M, norm="m")
        for i, k in enumerate(np.exp(self.lnk)):
            integ = integrand * u[i, :]
            out[i] = (intg.simps(integ, dx=c.dlog10m) * np.log(10))

        return self.p_lin * out ** 2 / (self.f_halos * self.mean_dens) ** 2


    @cached_property("dndm", "_wdm", "lnk", "M", "profile", "f_halos", "mean_dens",
                     "bias", "bias_smooth")
    def power_sh(self):
        c = deepcopy(self)
        c.update(hod_params={"M_min":np.log10(self._wdm.m_fs)})
        integrand = c.M ** 2 * c.dndm_rescaled * c.bias
        out = np.zeros_like(self.lnk)
        u = c.profile.u(np.exp(c.lnk), c.M, norm="m")
        for i, k in enumerate(np.exp(self.lnk)):
            integ = integrand * u [i, :]
            out[i] = intg.simps(integ, dx=c.dlog10m) * np.log(10)

        return self.bias_smooth * self.p_lin * out / (self.f_halos * self.mean_dens)

    @cached_property("bias_smooth", "matter_power")
    def power_ss(self):
        return self.bias_smooth ** 2 * np.exp(self.nonlinear_power)  # self.p_lin

    @cached_property("dndm", "bias", "M", "_wdm", "mean_dens", "f_halos")
    def bias_smooth(self):
        c = deepcopy(self)
        c.update(hod_params={"M_min":np.log10(self._wdm.m_fs)})
        integrand = c.M ** 2 * c.dndm_rescaled * c.bias
        integral = intg.simps(integrand, dx=c.dlog10m) * np.log(10)

        return (1 / (1 - self.f_halos)) * (1 - integral / self.mean_dens)

    @cached_property("_wdm")
    def cm(self):
        kwargs = dict(nu=self.nu, z=self.z, growth=self.growth_model,
                      M=self.M, **self.cm_params)
        if np.issubclass_(self.cm_relation, CMRelationWDMRescale):
            cm = self.cm_relation(m_hm=self._wdm.m_hm, **kwargs)
        elif np.issubclass_(self.cm_relation, CMRelation):
            cm = self.cm_relation(**kwargs)
        elif self.cm_relation.endswith("WDM"):
            cm = get_cm_rescaled(self.cm_relation[:-3])(m_hm=self._wdm.m_hm, **kwargs)
        else:
            cm = get_model(self.cm_relation, "halomod.concentration", **kwargs)

        return cm
