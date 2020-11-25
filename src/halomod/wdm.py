"""
Contains WDM versions of all models and frameworks
"""
from .halo_model import DMHaloModel
from hmf import cached_quantity, parameter
import numpy as np
from scipy import integrate as intg
from hmf.alternatives.wdm import MassFunctionWDM
import sys
from .integrate_corr import ProjectedCF
from .concentration import CMRelation
from hmf._internals._framework import get_mdl
from numpy import issubclass_

# ===============================================================================
# C-M relations
# ===============================================================================
def CMRelationWDMRescaled(name):
    """Class factory for Rescaled CM relations."""
    if name.endswith("WDM"):
        name = name[:-3]

    x = getattr(sys.modules["halomod.concentration"], name)

    def __init__(self, m_hm=1000, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.m_hm = m_hm

    def cm(self, m, z=0):
        """Rescaled Concentration-Mass relation for WDM."""
        cm = super(self.__class__, self).cm(m, z)
        g1 = self.params["g1"]
        g2 = self.params["g2"]
        b0 = self.params["beta0"]
        b1 = self.params["beta1"]
        return cm * (1 + g1 * self.m_hm / m) ** (-g2) * (1 + z) ** (b0 * z - b1)

    K = type(name + "WDM", (x,), {})
    K._defaults.update({"g1": 60, "g2": 0.17, "beta0": 0.026, "beta1": 0.04})

    K.__init__ = __init__
    K.cm = cm
    return K


# ===============================================================================
# Framework
# ===============================================================================
class HaloModelWDM(DMHaloModel, MassFunctionWDM):
    """
    This class is a derivative of HaloModel which sets a few defaults that make
    more sense for a WDM model, and also implements the framework to include a
    smooth component.

    See Schneider et al. 2012 for details on the smooth component.
    """

    def __init__(self, **kw):
        kw.setdefault("halo_concentration_model", "Ludlow2016")
        super(HaloModelWDM, self).__init__(**kw)

    @cached_quantity
    def f_halos(self):
        """The total fraction of mass bound up in halos."""
        return self.rho_gtm[0] / self.mean_density

    @cached_quantity
    def power_auto_matter(self):
        """Auto power spectrum of dark matter"""
        return (
            (1 - self.f_halos) ** 2 * self.power_auto_matter_ss
            + 2 * (1 - self.f_halos) * self.f_halos * self.power_auto_matter_sh
            + self.f_halos ** 2 * self.power_auto_matter_hh
        )

    @cached_quantity
    def power_auto_matter_hh(self) -> np.ndarray:
        """The halo-halo matter power spectrum (includes both 1-halo and 2-halo terms)."""
        return (
            (self.power_1h_auto_matter + self.power_2h_auto_matter)
            * self.mean_density ** 2
            / self.rho_gtm[0] ** 2
        )

    @cached_quantity
    def power_auto_matter_sh(self) -> np.ndarray:
        """The smooth-halo cross power spectrum."""
        integrand = (
            self.m
            * self.dndm
            * self.halo_bias
            * self.halo_profile.u(self.k, self.m, norm="m")
        )
        pch = intg.simps(integrand, self.m)
        return self.bias_smooth * self._power_halo_centres_table * pch / self.rho_gtm[0]

    @cached_quantity
    def power_auto_matter_ss(self) -> np.ndarray:
        """The smooth-smooth matter power spectrum."""
        return self.bias_smooth ** 2 * self._power_halo_centres_table

    @cached_quantity
    def bias_smooth(self):
        """Bias of smooth component of the field

        Eq. 35 from Smith and Markovic 2011.
        """
        return (1 - self.f_halos * self.bias_effective_matter) / (1 - self.f_halos)

    @cached_quantity
    def mean_density_halos(self):
        """Mean density of matter in halos"""
        return self.rho_gtm[0]

    @cached_quantity
    def mean_density_smooth(self):
        """Mean density of matter outside halos"""
        return (1 - self.f_halos) * self.mean_density

    @parameter("model")
    def halo_concentration_model(self, val):
        """A halo_concentration-mass relation"""
        if isinstance(val, str) and val.endswith("WDM"):
            return CMRelationWDMRescaled(val)
        return get_mdl(val, "CMRelation")

    @cached_quantity
    def halo_concentration(self):
        """Halo Concentration"""
        cm = super().halo_concentration

        if hasattr(cm, "m_hm"):
            cm.m_hm = self.wdm.m_hm

        return cm


class ProjectedCFWDM(ProjectedCF, HaloModelWDM):
    """Projected Correlation Function for WDM halos."""

    pass
