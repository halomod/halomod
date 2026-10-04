"""Contains WDM versions of all models and frameworks."""

import functools

import numpy as np
from hmf import cached_quantity, parameter
from hmf._internals._framework import get_mdl
from hmf.alternatives.wdm import MassFunctionWDM
from scipy import integrate as intg

from . import concentration
from .concentration import CMRelation
from .halo_model import DMHaloModel, TracerHaloModel
from .integrate_corr import ProjectedCF

# ===============================================================================
# C-M relations
# ===============================================================================
#: Default parameters of the WDM rescaling of Schneider et al. (2012), added to the
#: parameters of the CDM concentration-mass relation being rescaled.
_WDM_RESCALING_DEFAULTS = {"g1": 60, "g2": 0.17, "beta0": 0.026, "beta1": 0.04}


@functools.cache
def _make_wdm_rescaled_cm_relation(name: str) -> type[CMRelation]:
    """Build the WDM-rescaled subclass of the CDM concentration model ``name``.

    This is memoised, so that each CDM model has exactly one WDM counterpart, which
    is registered exactly once in the :class:`~halomod.concentration.CMRelation`
    plugin registry.
    """
    parent = getattr(concentration, name)

    def __init__(self, m_hm: float = 1000, **kwargs):
        super(K, self).__init__(**kwargs)
        self.m_hm = m_hm

    def cm(self, m, z=0):
        """Rescaled Concentration-Mass relation for WDM."""
        cm = super(K, self).cm(m, z)
        g1 = self.params["g1"]
        g2 = self.params["g2"]
        b0 = self.params["beta0"]
        b1 = self.params["beta1"]
        return cm * (1 + g1 * self.m_hm / m) ** (-g2) * (1 + z) ** (b0 * z - b1)

    K = type(
        name + "WDM",
        (parent,),
        {
            "__module__": __name__,
            "__qualname__": name + "WDM",
            "__doc__": (
                f"WDM-rescaled version of :class:`~halomod.concentration.{name}`.\n\n"
                "The CDM concentration is multiplied by "
                "``(1 + g1 * m_hm / m)**(-g2) * (1 + z)**(beta0 * z - beta1)``, "
                "following Schneider et al. (2012), where ``m_hm`` is the "
                "half-mode mass of the WDM model."
            ),
            # The subclass needs its own dict, otherwise the WDM parameters would be
            # written into the parent (CDM) model's defaults.
            "_defaults": {**parent._defaults, **_WDM_RESCALING_DEFAULTS},
            "__init__": __init__,
            "cm": cm,
        },
    )
    return K


def CMRelationWDMRescaled(name: str) -> type[CMRelation]:
    """Return the WDM-rescaled version of a CDM concentration-mass relation.

    The returned class multiplies the concentration of the CDM model by
    ``(1 + g1 * m_hm / m)**(-g2) * (1 + z)**(beta0 * z - beta1)`` (Schneider et al.
    2012), where ``m_hm`` is the WDM half-mode mass, passed to its constructor (and
    set by :class:`HaloModelWDM`). Its parameters are those of the CDM model plus
    ``g1``, ``g2``, ``beta0`` and ``beta1``; the CDM model itself is not modified.

    Parameters
    ----------
    name : str
        Name of a concentration-mass relation in :mod:`halomod.concentration`,
        optionally with a ``"WDM"`` suffix (e.g. ``"Duffy08"`` or ``"Duffy08WDM"``).

    Returns
    -------
    type
        A subclass of the named CDM model, called ``name + "WDM"``. Repeated calls
        with the same model return the same class.
    """
    return _make_wdm_rescaled_cm_relation(name.removesuffix("WDM"))


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
        super().__init__(**kw)

    @cached_quantity
    def f_halos(self):
        """The total fraction of mass bound up in halos."""
        return self.rho_gtm[0] / self.mean_density

    @cached_quantity
    def power_auto_matter(self):
        """Auto power spectrum of dark matter."""
        return (
            (1 - self.f_halos) ** 2 * self.power_auto_matter_ss
            + 2 * (1 - self.f_halos) * self.f_halos * self.power_auto_matter_sh
            + self.f_halos**2 * self.power_auto_matter_hh
        )

    @cached_quantity
    def power_auto_matter_hh(self) -> np.ndarray:
        """The halo-halo matter power spectrum (includes both 1-halo and 2-halo terms)."""
        return (
            (self.power_1h_auto_matter + self.power_2h_auto_matter)
            * self.mean_density**2
            / self.rho_gtm[0] ** 2
        )

    @cached_quantity
    def power_auto_matter_sh(self) -> np.ndarray:
        """The smooth-halo cross power spectrum."""
        integrand = (
            self.m * self.dndm * self.halo_bias * self.halo_profile.u(self.k_hm, self.m, norm="m")
        )
        pch = intg.simpson(integrand, x=self.m)
        return self.bias_smooth * self._power_halo_centres_fnc(self.k_hm) * pch / self.rho_gtm[0]

    @cached_quantity
    def power_auto_matter_ss(self) -> np.ndarray:
        """The smooth-smooth matter power spectrum."""
        return self.bias_smooth**2 * self._power_halo_centres_fnc(self.k_hm)

    @cached_quantity
    def bias_smooth(self):
        """Bias of smooth component of the field.

        Eq. 35 from Smith and Markovic 2011.
        """
        return (1 - self.f_halos * self.bias_effective_matter) / (1 - self.f_halos)

    @cached_quantity
    def mean_density_halos(self):
        """Mean density of matter in halos."""
        return self.rho_gtm[0]

    @cached_quantity
    def mean_density_smooth(self):
        """Mean density of matter outside halos."""
        return (1 - self.f_halos) * self.mean_density

    @parameter("model")
    def halo_concentration_model(self, val):
        """A halo_concentration-mass relation."""
        if isinstance(val, str) and val.endswith("WDM"):
            return CMRelationWDMRescaled(val)
        return get_mdl(val, "CMRelation")

    @cached_quantity
    def halo_concentration(self):
        """Halo Concentration."""
        cm = super().halo_concentration

        if hasattr(cm, "m_hm"):
            cm.m_hm = self.wdm.m_hm

        return cm


class TracerHaloModelWDM(TracerHaloModel, HaloModelWDM):
    def __init__(self, **kw):
        kw.setdefault("halo_concentration_model", "Ludlow2016")
        super().__init__(**kw)


class ProjectedCFWDM(ProjectedCF, HaloModelWDM):
    """Projected Correlation Function for WDM halos."""
