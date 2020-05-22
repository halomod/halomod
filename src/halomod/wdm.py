"""
Contains WDM versions of all models and frameworks
"""
from .concentration import CMRelation
from .halo_model import HaloModel
from hmf import cached_quantity, parameter
import numpy as np
from scipy import integrate as intg
from hmf.alternatives.wdm import MassFunctionWDM
from hmf._internals._framework import get_model
import sys
from .integrate_corr import ProjectedCF
from copy import copy


# ===============================================================================
# C-M relations
# ===============================================================================
def CMRelationWDMRescaled(name, **kwargs):
    """
    Class factory for Rescaled CM relations.
    """
    x = getattr(sys.modules["halomod.halo_concentration"], name)

    # class CMRelationWDMRescale(x):
    #     """
    #     Base class for simply rescaled halo_concentration-mass relations (cf. Schneider
    #     2013, Bose+15)
    #     """
    def __init__(self, m_hm, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.m_hm = m_hm

    def cm(self, m, z=0):
        cm = super(self.__class__, self).halo_cm(m)
        g1 = self.params["g1"]
        g2 = self.params["g2"]
        b0 = self.params["beta0"]
        b1 = self.params["beta1"]
        return cm * (1 + g1 * self.m_hm / m) ** (-g2) * (1 + z) ** (b0 * z - b1)

    K = type(name + "WDM", (x,), {})
    K._defaults.update({"g1": 60, "g2": 0.17, "beta0": 0.026, "beta1": 0.04})

    K.__init__ = __init__
    K.cm = cm
    return K(**kwargs)


# ===============================================================================
# Framework
# ===============================================================================
class HaloModelWDM(HaloModel, MassFunctionWDM):
    """
    This class is a derivative of HaloModel which sets a few defaults that make
    more sense for a WDM model, and also implements the framework to include a
    smooth component
    """

    def __init__(self, **kw):
        kw.setdefault("halo_concentration_model", "Ludlow2016")
        super(HaloModelWDM, self).__init__(**kw)

    @parameter("switch")
    def concentration_model(self, val):
        """A halo_concentration-mass relation"""
        if not isinstance(val, str) and not np.issubclass_(val, CMRelation):
            raise ValueError(
                "halo_concentration_model must be a subclass of halo_concentration.CMRelation"
            )
        return val

    # @cached_quantity
    # def M(self):
    #     """ Halo masses in M_sun/h """
    #     return 10 ** np.arange(np.log10(self._wdm.m_fs), 18, self.dlog10m)

    @cached_quantity
    def f_halos(self):
        """
        The total fraction of mass bound up in halos
        """
        return self.rho_gtm[0] / self.mean_density0

    @cached_quantity
    def power_mm(self):
        return (
            (1 - self.f_halos) ** 2 * self.power_mm_ss
            + 2 * (1 - self.f_halos) * self.f_halos * self.power_mm_sh
            + self.f_halos ** 2 * self.power_mm_hh
        )

    @cached_quantity
    def power_mm_hh(self):
        return (
            super(HaloModelWDM, self).power_mm
            * self.mean_density0 ** 2
            / self.rho_gtm[0] ** 2
        )

    @cached_quantity
    def power_mm_sh(self):
        integrand = (
            self.m ** 2
            * self.dndm
            * self.bias
            * self.profile.u(self.k, self.m, norm="m")
        )
        pch = intg.simps(integrand, dx=np.log(10) * self.dlog10m)
        return self.bias_smooth * self._power_halo_centres * pch / self.rho_gtm[0]

    @cached_quantity
    def power_mm_ss(self):
        return self.bias_smooth ** 2 * self._power_halo_centres

    @cached_quantity
    def bias_smooth(self):
        return (1 - self.f_halos * self.bias_effective_matter) / (1 - self.f_halos)

    @cached_quantity
    def cm(self):
        this_filter = copy(self.filter)
        this_filter.power = self._power0
        this_profile = self.profile_model(
            None, self.mean_density0, self.delta_halo, self.z, **self.profile_params
        )

        kwargs = dict(
            filter0=this_filter,
            mean_density0=self.mean_density0,
            growth=self.growth,
            delta_c=self.delta_c,
            profile=this_profile,
            cosmo=self.cosmo,
            delta_halo=self.delta_halo,
            **self.concentration_params
        )

        if np.issubclass_(self.concentration_model, CMRelation):
            if self.concentration_model.__class__.__name__.endswith("WDM"):
                cm = self.concentration_model(m_hm=self.wdm.m_hm, **kwargs)
            else:
                cm = self.concentration_model(**kwargs)
        elif self.concentration_model.endswith("WDM"):
            cm = CMRelationWDMRescaled(
                self.concentration_model[:-3], m_hm=self.wdm.m_hm, **kwargs
            )
        else:
            cm = get_model(
                self.concentration_model, "halomod.halo_concentration", **kwargs
            )

        return cm


class ProjectedCFWDM(ProjectedCF, HaloModelWDM):
    pass
