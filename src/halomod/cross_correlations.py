"""
Modules defining cross-correlated samples.

Has classes for both pure HOD cross-correlations
(i.e. number of cross-pairs) and for HaloModel-derived quantities
based on these cross-pairs.

To construct a :class:`CrossCorrelations` one need to specify the
halo models to be cross-correlated, and how they're correlated.

Examples
--------
Cross-correlating the same galaxy samples in different redshifts::

>>> from halomod import HaloModel
>>> from halomod.cross_correlations import CrossCorrelations, ConstantCorr
>>> cross = CrossCorrelations(cross_hod_model=ConstantCorr, halo_model_1_params=dict(z=1.0),
>>>                           halo_model_2_params=dict(z=0.0))
>>> pkcorr = cross.power_cross

`pkcorr` corresponds to the cross-power at `cross.halo_model_1.k_hm`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import update_wrapper
from typing import Any

import numpy as np
from hmf import Component, Framework
from hmf._internals._cache import cached_quantity, hidden_loc, parameter, subframework
from hmf._internals._framework import get_mdl, pluggable
from scipy import integrate as intg
from scipy.interpolate import InterpolatedUnivariateSpline as _IUS

from . import _references, tools
from .halo_model import TracerHaloModel


def _forward_subframework_dependencies(fmwork: Framework, name: str) -> None:
    """Make every quantity currently being computed depend on what ``name`` depends on.

    hmf records which sub-framework parameters a quantity depends on only while that
    quantity is being computed. When a quantity reads another quantity that is already
    cached, it inherits that quantity's own parameters but not its sub-framework
    parameters, so it would not be invalidated when e.g. ``halo_model_1`` is updated.
    This copies the sub-framework dependencies of ``name`` to every quantity that is
    currently being computed.
    """
    active = [q for q in getattr(fmwork, hidden_loc(fmwork, "active_q")) if q != name]
    if not active:
        return

    for sub_name in getattr(fmwork, hidden_loc(fmwork, "subframeworks"), set()):
        # Read the stored sub-framework directly: going through the property would
        # itself register indexes.
        sub = getattr(fmwork, hidden_loc(fmwork, sub_name), None)
        if sub is None:
            continue
        sub_prpa = getattr(sub, hidden_loc(sub, "recalc_prop_par"))
        deps = sub_prpa.get(":" + name)
        if not deps:
            continue
        for q in active:
            sub_prpa.setdefault(":" + q, set()).update(deps)


def _cross_cached_quantity(f: Callable[[Any], Any]) -> property:
    """Like :func:`hmf.cached_quantity`, but also tracks sub-framework dependencies.

    The quantity's sub-framework dependencies are passed on to any quantity that reads it
    (see :func:`_forward_subframework_dependencies`), so a cached cross-correlation
    quantity is recomputed whenever a parameter of either halo model changes, however
    its intermediate quantities happened to be cached.
    """
    prop = cached_quantity(f)
    name = f.__name__

    def _get_property(self):
        value = prop.fget(self)
        _forward_subframework_dependencies(self, name)
        return value

    # This also copies the marker hmf uses to list the quantity in quantities_available().
    update_wrapper(_get_property, prop.fget)
    return property(_get_property, None, prop.fdel, prop.__doc__)


@pluggable
class _HODCross(ABC, Component):
    """Provides methods necessary to compute cross-correlation pairs for HOD models."""

    _defaults = {}

    def __init__(self, hods, **model_params):
        super().__init__(**model_params)

        assert len(hods) == 2
        self.hods = hods

    @abstractmethod
    def R_ss(self, m):
        r"""The cross-correlation of numbers of pairs within a halo.

        Notes
        -----
        Defined by

        .. math:: \langle T_1 T_2 \rangle  = \langle T_1 \rangle \langle T_2 \rangle +
                  \sigma_1 \sigma_2 R_{ss},

        where :math:`T` is the total amount of tracer in the halo's profile (i.e. not
        counting the central component, if this exists).
        """

    @abstractmethod
    def R_cs(self, m):
        r"""
        The cross-correlation of central-satellite pairs within a halo.

        Central from first hod, satellite from second.

        Notes
        -----
        Defined by

        .. math:: \langle T^c_1 T^s_2 \rangle  =
                  \langle T^c_1 \rangle \langle T^s_2 \rangle +
                  \sigma^c_1 \sigma^s_2 R_{cs},

        where :math:`T^s` is the total amount of tracer in the halo's profile (i.e. not
        counting the central component,if this exists).
        """

    @abstractmethod
    def R_sc(self, m):
        r"""
        The cross-correlation of satellite-central pairs within a halo.

        Central from second hod, Satellite from first.

        Notes
        -----
        Defined by

        .. math:: \langle T^s_1 T^c_2 \rangle  =
                  \langle T^s_1 \rangle \langle T^c_2 \rangle +
                  \sigma^s_1 \sigma^c_2 R_{sc},

        where :math:`T^s` is the total amount of tracer in the halo's profile (i.e. not
        counting the central component,if this exists).
        """

    @abstractmethod
    def self_pairs(self, m):
        r"""The expected number of cross-pairs at a separation of zero."""

    def ss_cross_pairs(self, m):
        r"""The average value of cross-pairs in a halo of mass m.

        Notes
        -----
        Given by

        .. math:: `\langle T^s_1 T^s_2 \rangle - Q`
        """
        h1, h2 = self.hods

        return (
            h1.satellite_occupation(m) * h2.satellite_occupation(m)
            + h1.sigma_satellite(m) * h2.sigma_satellite(m) * self.R_ss(m)
            - self.self_pairs(m)
        )

    def cs_cross_pairs(self, m):
        r"""The average value of cross-pairs in a halo of mass m.

        Notes
        -----
        .. math:: \langle T^c_1 T^s_2 \rangle.

        """
        h1, h2 = self.hods

        return h1.central_occupation(m) * h2.satellite_occupation(m) + h1.sigma_central(
            m
        ) * h2.sigma_satellite(m) * self.R_cs(m)

    def sc_cross_pairs(self, m):
        r"""The average value of cross-pairs in a halo of mass m,.

        Notes
        -----
        .. math:: \langle T^s_1 T^c_2 \rangle
        """
        h1, h2 = self.hods

        return h2.central_occupation(m) * h1.satellite_occupation(m) + h2.sigma_central(
            m
        ) * h1.sigma_satellite(m) * self.R_sc(m)


class ConstantCorr(_HODCross):
    """Correlation relation for constant cross-correlation pairs."""

    _defaults = {"R_ss": 0.0, "R_cs": 0.0, "R_sc": 0.0}

    def R_ss(self, m):
        return self.params["R_ss"]

    def R_cs(self, m):
        return self.params["R_cs"]

    def R_sc(self, m):
        return self.params["R_sc"]

    def self_pairs(self, m):
        """The expected number of cross-pairs at a separation of zero."""
        return 0


class CrossCorrelations(Framework):
    r"""
    The Framework for cross-correlations.

    This class generates two :class:`~halomod.halo_model.TracerHaloModel`,
    and calculates their cross-correlation according to the cross-correlation
    model given.

    Parameters
    ----------
    cross_hod_model : str or :class:`_HODCross` subclass, optional
        Model for the HOD of cross correlation. Default is :class:`ConstantCorr`.
    cross_hod_params : dict
        Parameters for HOD used in cross-correlation.
    halo_model_1_params,halo_model_2_params : dict
        Parameters for the tracers used in cross-correlation.

    """

    def __init__(
        self,
        cross_hod_model: str | type[_HODCross] = ConstantCorr,
        cross_hod_params: dict | None = None,
        halo_model_1_params: dict | None = None,
        halo_model_2_params: dict | None = None,
    ):
        super().__init__()

        self.cross_hod_model = cross_hod_model
        self.cross_hod_params = cross_hod_params or {}

        self._halo_model_1_params = halo_model_1_params or {}
        self._halo_model_2_params = halo_model_2_params or {}

    def get_acknowledgments(self, flat: bool = False) -> dict[str, tuple[str, ...]] | list[str]:
        """Get the references to cite for the current setup of the cross-correlation.

        This extends hmf's ``Framework.get_acknowledgments`` with an entry ``"halomod"``
        (the paper describing halomod), placed right after the ``"hmf"`` entry. The
        other entries hold the references of the ``cross_hod_model``, and those of the
        models of each halo model, prefixed by its name (e.g.
        ``"halo_model_1.hod_model"``).

        Parameters
        ----------
        flat
            If True, return a single list of all the references with duplicates
            removed (keeping the first occurrence), e.g. for a paper's bibliography.

        Returns
        -------
        dict or list
            A dict mapping each source to a tuple of formatted citations, or a flat
            list of citations if ``flat`` is True.

        Examples
        --------
        >>> cross = CrossCorrelations(cross_hod_model=ConstantCorr)
        >>> bibliography = cross.get_acknowledgments(flat=True)
        """
        return _references.get_acknowledgments(super().get_acknowledgments, flat)

    @parameter("model")
    def cross_hod_model(self, val):
        return get_mdl(val, "_HODCross")

    @parameter("param")
    def cross_hod_params(self, val):
        return val

    @subframework
    def halo_model_1(self) -> TracerHaloModel:
        """Halo Model of the first tracer."""
        return TracerHaloModel(**self._halo_model_1_params)

    @subframework
    def halo_model_2(self) -> TracerHaloModel:
        """Halo Model of the second tracer."""
        return TracerHaloModel(**self._halo_model_2_params)

    # ===========================================================================
    # Cross-correlations
    # ===========================================================================
    @_cross_cached_quantity
    def cross_hod(self):
        """HOD model of the cross-correlation."""
        return self.cross_hod_model(
            [self.halo_model_1.hod, self.halo_model_2.hod], **self.cross_hod_params
        )

    @_cross_cached_quantity
    def power_1h_cross_fnc(self):
        """Total 1-halo cross-power."""
        hm1, hm2 = self.halo_model_1, self.halo_model_2
        mask = np.logical_and(
            np.logical_and(
                np.logical_not(np.isnan(self.cross_hod.ss_cross_pairs(hm1.m))),
                np.logical_not(np.isnan(self.cross_hod.sc_cross_pairs(hm1.m))),
            ),
            np.logical_not(np.isnan(self.cross_hod.cs_cross_pairs(hm1.m))),
        )

        m = hm1.m[mask]
        u1 = hm1.tracer_profile_ukm[:, mask]
        u2 = hm2.tracer_profile_ukm[:, mask]

        integ = hm1.dndm[mask] * (
            u1 * u2 * self.cross_hod.ss_cross_pairs(m)
            + u1 * self.cross_hod.sc_cross_pairs(m)
            + u2 * self.cross_hod.cs_cross_pairs(m)
        )

        p = intg.simpson(integ, x=m)

        p /= hm1.mean_tracer_den * hm2.mean_tracer_den
        return tools.ExtendedSpline(hm1.k, p, lower_func="power_law", upper_func="power_law")

    @_cross_cached_quantity
    def power_1h_cross(self):
        """Total 1-halo cross-power."""
        return self.power_1h_cross_fnc(self.halo_model_1.k_hm)

    @_cross_cached_quantity
    def corr_1h_cross_fnc(self):
        """The 1-halo term of the cross correlation."""
        corr = tools.hankel_transform(self.power_1h_cross_fnc, self.halo_model_1._r_table, "r")
        return tools.ExtendedSpline(
            self.halo_model_1._r_table,
            corr,
            lower_func="power_law",
            upper_func=tools._zero,
        )

    @_cross_cached_quantity
    def corr_1h_cross(self):
        """The 1-halo term of the cross correlation."""
        return self.corr_1h_cross_fnc(self.halo_model_1.r)

    @_cross_cached_quantity
    def power_2h_cross_fnc(self):
        """The 2-halo term of the cross-power spectrum.

        Uses spline integration from ``tracer_mmin`` (the same lower-bound
        convention as :attr:`~halomod.HaloModel.power_2h_auto_tracer`) so
        that the result varies smoothly as the HOD ``M_min`` parameter is
        changed and is consistent with the auto-power spectrum when both
        tracer populations are identical.
        """
        hm1, hm2 = self.halo_model_1, self.halo_model_2

        u1 = hm1.tracer_profile_ukm  # (k, m)
        u2 = hm2.tracer_profile_ukm  # (k, m)

        # Build integrands including the m-Jacobian for log-mass integration:
        #   ∫ f(m) dm  =  ∫ f(m)·m  d(ln m)
        # Each model uses its own halo_bias so grids don't need to match.
        integrand1 = hm1.dndm * hm1.halo_bias * hm1._total_occupation * u1 * hm1.m  # (k, m)
        integrand2 = hm2.dndm * hm2.halo_bias * hm2._total_occupation * u2 * hm2.m  # (k, m)

        lnm1 = np.log(hm1.m)
        lnm2 = np.log(hm2.m)

        xmin1 = hm1.tracer_mmin
        if xmin1 is not None and xmin1 > hm1.m[0]:
            lnxmin1 = np.log(xmin1)
            b1 = np.apply_along_axis(
                lambda f: _IUS(lnm1, f).integral(lnxmin1, lnm1[-1]), -1, integrand1
            )
        else:
            b1 = intg.simpson(integrand1, x=lnm1)

        xmin2 = hm2.tracer_mmin
        if xmin2 is not None and xmin2 > hm2.m[0]:
            lnxmin2 = np.log(xmin2)
            b2 = np.apply_along_axis(
                lambda f: _IUS(lnm2, f).integral(lnxmin2, lnm2[-1]), -1, integrand2
            )
        else:
            b2 = intg.simpson(integrand2, x=lnm2)

        p = (
            b1
            * b2
            * hm1._power_halo_centres_fnc(hm1.k)
            / (hm1.mean_tracer_den * hm2.mean_tracer_den)
        )

        return tools.ExtendedSpline(
            hm1.k,
            p,
            lower_func=hm1.linear_power_fnc,
            match_lower=True,
            upper_func="power_law",
        )

    @_cross_cached_quantity
    def power_2h_cross(self):
        """The 2-halo term of the cross-power spectrum."""
        return self.power_2h_cross_fnc(self.halo_model_1.k_hm)

    @_cross_cached_quantity
    def corr_2h_cross_fnc(self):
        """The 2-halo term of the cross-correlation."""
        corr = tools.hankel_transform(
            self.power_2h_cross_fnc, self.halo_model_1._r_table, "r", h=1e-4
        )
        return tools.ExtendedSpline(
            self.halo_model_1._r_table,
            corr,
            lower_func="power_law",
            upper_func=tools._zero,
        )

    @_cross_cached_quantity
    def corr_2h_cross(self):
        """The 2-halo term of the cross-correlation."""
        return self.corr_2h_cross_fnc(self.halo_model_1.r)

    def power_cross_fnc(self, k):
        """Total tracer cross power spectrum."""
        return self.power_1h_cross_fnc(k) + self.power_2h_cross_fnc(k)

    @_cross_cached_quantity
    def power_cross(self):
        """Total tracer cross power spectrum."""
        return self.power_cross_fnc(self.halo_model_1.k_hm)

    def corr_cross_fnc(self, r):
        """The tracer cross correlation function."""
        return self.corr_1h_cross_fnc(r) + self.corr_2h_cross_fnc(r) + 1

    @_cross_cached_quantity
    def corr_cross(self):
        """The tracer cross correlation function."""
        return self.corr_cross_fnc(self.halo_model_1.r)
