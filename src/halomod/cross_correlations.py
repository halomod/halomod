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
    >>> from halomod.cross_correlations import CrossCorrelations, HODCross
    >>> cross = CrossCorrelations(cross_hod_model=ConstantCorr, halo_model_1_params=dict(z=1.0),
    >>>                           halo_model_2_params=dict(z=0.0))
    >>> pkcorr = cross.power_cross
"""

from .halo_model import TracerHaloModel
from hmf import Component, Framework
from hmf._internals._framework import get_mdl, pluggable
from hmf._internals._cache import parameter, cached_quantity, subframework
from abc import ABC, abstractmethod
import numpy as np
from scipy import integrate as intg
from . import tools


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

        .. math:: \langle T_1 T_2 \rangle  = \langle T_1 \rangle \langle T_2 \rangle + \sigma_1 \sigma_2 R_{ss},

        where :math:`T` is the total amount of tracer in the halo's profile (i.e. not counting the
        central component, if this exists).
        """
        pass

    @abstractmethod
    def R_cs(self, m):
        r"""
        The cross-correlation of central-satellite pairs within a halo.

        Central from first hod, satellite from second.

        Notes
        -----
        Defined by

        .. math:: \langle T^c_1 T^s_2 \rangle  = \langle T^c_1 \rangle \langle T^s_2 \rangle + \sigma^c_1 \sigma^s_2 R_{cs},

        where :math:`T^s` is the total amount of tracer in the halo's profile (i.e. not counting the
        central component,if this exists).
        """
        pass

    @abstractmethod
    def R_sc(self, m):
        r"""
        The cross-correlation of satellite-central pairs within a halo.

        Central from second hod, Satellite from first.

        Notes
        -----
        Defined by

        .. math:: \langle T^s_1 T^c_2 \rangle  = \langle T^s_1 \rangle \langle T^c_2 \rangle + \sigma^s_1 \sigma^c_2 R_{sc},

        where :math:`T^s` is the total amount of tracer in the halo's profile (i.e. not counting
        the central component,if this exists).
        """
        pass

    @abstractmethod
    def self_pairs(self, m):
        r"""The expected number of cross-pairs at a separation of zero."""
        pass

    def ss_cross_pairs(self, m):
        r"""The average value of cross-pairs in a halo of mass m.

        Notes
        -----
        .. math:: `\langle T^s_1 T^s_2 \rangle - Q`"""
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
        r"""The average value of cross-pairs in a halo of mass m,

        Notes
        -----
        .. math:: \langle T^s_1 T^c_2 \rangle
        """
        h1, h2 = self.hods

        return h2.central_occupation(m) * h1.satellite_occupation(m) + h2.sigma_central(
            m
        ) * h1.sigma_satellite(m) * self.R_sc(m)


class ConstantCorr(_HODCross):
    """Correlation relation for constant cross-correlation pairs"""

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
    cross_hod_model : class
        Model for the HOD of cross correlation.
    cross_hod_params : dict
        Parameters for HOD used in cross-correlation.
    halo_model_1_params,halo_model_2_params : dict
        Parameters for the tracers used in cross-correlation.

    """

    def __init__(
        self,
        cross_hod_model,
        cross_hod_params={},
        halo_model_1_params={},
        halo_model_2_params={},
    ):
        super().__init__()

        self.cross_hod_model = cross_hod_model
        self.cross_hod_params = cross_hod_params

        self._halo_model_1_params = halo_model_1_params
        self._halo_model_2_params = halo_model_2_params

    @parameter("model")
    def cross_hod_model(self, val):
        return get_mdl(val, "_HODCross")

    @parameter("param")
    def cross_hod_params(self, val):
        return val

    @subframework
    def halo_model_1(self) -> TracerHaloModel:
        """Halo Model of the first tracer"""
        return TracerHaloModel(**self._halo_model_1_params)

    @subframework
    def halo_model_2(self) -> TracerHaloModel:
        """Halo Model of the second tracer"""
        return TracerHaloModel(**self._halo_model_2_params)

    # ===========================================================================
    # Cross-correlations
    # ===========================================================================
    @cached_quantity
    def cross_hod(self):
        """HOD model of the cross-correlation"""
        return self.cross_hod_model(
            [self.halo_model_1.hod, self.halo_model_2.hod], **self.cross_hod_params
        )

    @cached_quantity
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

        p = intg.simps(integ, m)

        p /= hm1.mean_tracer_den * hm2.mean_tracer_den
        return tools.ExtendedSpline(
            hm1.k, p, lower_func="power_law", upper_func="power_law"
        )

    @property
    def power_1h_cross(self):
        """Total 1-halo cross-power."""
        return self.power_1h_cross_fnc(self.halo_model_1.k_hm)

    @cached_quantity
    def corr_1h_cross_fnc(self):
        """The 1-halo term of the cross correlation"""
        corr = tools.hankel_transform(
            self.power_1h_cross_fnc, self.halo_model_1._r_table, "r"
        )
        return tools.ExtendedSpline(
            self.halo_model_1._r_table,
            corr,
            lower_func="power_law",
            upper_func=tools._zero,
        )

    @cached_quantity
    def corr_1h_cross(self):
        """The 1-halo term of the cross correlation"""
        return self.corr_1h_cross_fnc(self.halo_model_1.r)

    @cached_quantity
    def power_2h_cross_fnc(self):
        """The 2-halo term of the cross-power spectrum."""
        hm1, hm2 = self.halo_model_1, self.halo_model_2

        u1 = hm1.tracer_profile_ukm[:, hm1._tm]
        u2 = hm2.tracer_profile_ukm[:, hm2._tm]

        bias = hm1.halo_bias

        # Do this the simple way for now
        b1 = intg.simps(
            hm1.dndm[hm1._tm] * bias[hm1._tm] * hm1.total_occupation[hm1._tm] * u1,
            hm1.m[hm1._tm],
        )
        b2 = intg.simps(
            hm2.dndm[hm2._tm] * bias[hm2._tm] * hm2.total_occupation[hm2._tm] * u2,
            hm2.m[hm2._tm],
        )

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

    @property
    def power_2h_cross(self):
        """The 2-halo term of the cross-power spectrum."""
        return self.power_2h_cross_fnc(self.halo_model_1.k_hm)

    @cached_quantity
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

    @cached_quantity
    def corr_2h_cross(self):
        """The 2-halo term of the cross-correlation."""
        return self.corr_2h_cross_fnc(self.halo_model_1.r)

    def power_cross_fnc(self, k):
        """Total tracer cross power spectrum."""
        return self.power_1h_cross_fnc(k) + self.power_2h_cross_fnc(k)

    @property
    def power_cross(self):
        """Total tracer cross power spectrum."""
        return self.power_cross_fnc(self.halo_model_1.k_hm)

    def corr_cross_fnc(self, r):
        """The tracer cross correlation function."""
        return self.corr_1h_cross_fnc(r) + self.corr_2h_cross_fnc(r) + 1

    @property
    def corr_cross(self):
        """The tracer cross correlation function."""
        return self.corr_cross_fnc(self.halo_model_1.r)
