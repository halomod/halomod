"""
Main halo model module.

Contains Frameworks that combine all components necessary for halo model calculations
(eg. mass function, bias, concentration, halo profile).

Two main classes are provided: :class:`DMHaloModel` for dark-matter only halo models,
and :class:`TracerHaloModel` for halo models including a tracer population embedded in
the dark matter haloes, via a HOD.

The :class:`HaloModel` class is provided as an alias of :class:`TracerHaloModel`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from copy import copy

import numpy as np
import scipy.integrate as intg
from hmf import Cosmology, MassFunction, cached_quantity, parameter
from hmf._internals import get_mdl
from hmf.cosmology.cosmo import astropy_to_colossus
from hmf.density_field.filters import TopHat
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.optimize import minimize

# import hmf.tools as ht
from . import tools
from .concentration import CMRelation
from .halo_exclusion import Exclusion, NoExclusion


class DMHaloModel(MassFunction):
    """
    Dark-matter-only halo model class.

    This Framework is subclassed from hmf's ``MassFunction`` class, and operates in a
    similar manner.

    **kwargs: anything that can be used in the MassFunction class

    """

    def __init__(
        self,
        rmin=0.01,
        rmax=120.0,
        rnum=100,
        rlog=True,
        dr_table=0.01,
        hm_logk_min=-2,
        hm_logk_max=2,
        hm_dlog10k=0.05,
        halo_profile_model="NFW",
        halo_profile_params=None,
        halo_concentration_model="Duffy08",
        halo_concentration_params=None,
        bias_model="Tinker10",
        bias_params=None,
        sd_bias_model=None,
        sd_bias_params=None,
        exclusion_model="NoExclusion",
        exclusion_params=None,
        colossus_params=None,
        hc_spectrum="linear",
        Mmin=0,
        Mmax=18,
        force_1halo_turnover=True,
        force_unity_dm_bias: bool = True,
        **hmf_kwargs,
    ):
        """
        Initializer for the class.

        Note that all `*_model` parameters can be a string or a class of the type
        described below. If a string, it should be the name of a class that must exist
        in the relevant module within `halomod`.

        Parameters
        ----------
        rmin : float or arry-like, optional
            Minimum length scale over which to calculate correlations, in Mpc/h.
            Alternatively, if an array, this is used to specify the entire array of
            scales and `rmax`, `rnum` and `rlog` are ignored.
        rmax : float, optional
            Maximum length scale over which to calculate correlations, in Mpc/h
        rnum : int, optional
            The number of bins for correlation functions.
        rlog : bool, optional
            Whether the array of scales is regular in log-space.
        halo_profile_model: str or :class:`~profiles.Profile` subclass, optional
            The model for the density profile of the halos.
        halo_profile_params : dict, optional
            Parameters for the density profile model (see its docstring for details)
        halo_concentration_model : str or :class:`~concentration.CMRelation` subclass
            The model for the concentration-mass-redshift relation of the halos.
        halo_concentration_params : dict, optional
            Parameters for the concentration-mass relation (see its docstring for details)
        bias_model : str or :class:`~bias.Bias` subclass, optional
            The model of halo bias.
        bias_params : dict, optional
            Parameters for the bias model (see its docstring for details)
        sd_bias_model : str, None, or :class:`~bias.ScaleDepBias` subclass, optional
            A model for scale-dependent bias (as a function of `r`). Setting to None
            will use no scale-dependent bias.
        sd_bias_params : dict, optional
            Parameters for the scale-dependent bias model (see its docstring for details).
        exclusion_model : str, None or :class:`~halo_exclusion.Exclusion` subclass
            A model for how halo exclusion is calculated.
        exclusion_params : dict, optional
            Parameters for the halo exclusion model
        hc_spectrum : str, {'linear', 'nonlinear', 'filtered-nl', 'filtered-lin'}
            A choice for how the halo-centre power spectrum is defined. The "filtered"
            options arise from eg. Schneider, Smith et al. (2014).
        force_unity_dm_bias : bool
            At the largest scales, the DM should not be biased against itself, if all
            DM is in halos. That is, the effective bias of dark matter should be unity
            unless an alternative halo model is being used (eg. WDM). However, the
            integral that computes this bias is not in fact infinite, and may come short
            of unity (or, indeed, an unnormalized HMF/bias pair may be used). If this is
            set to true, the matter bias is forcibly renormalized to unity.

        Other Parameters
        ----------------
        All other parameters are passed to :class:`~MassFunction`.
        """
        self.bias_model, self.bias_params = bias_model, bias_params or {}

        try:
            hmf_model = self.bias_model.pair_hmf[0]
        except IndexError:
            hmf_model = "Tinker10"

        if "hmf_model" not in hmf_kwargs:
            hmf_kwargs["hmf_model"] = hmf_model

        super().__init__(Mmin=Mmin, Mmax=Mmax, **hmf_kwargs)

        # Initially save parameters to the class.
        self.halo_profile_model, self.halo_profile_params = (
            halo_profile_model,
            halo_profile_params or {},
        )
        self.halo_concentration_model, self.halo_concentration_params = (
            halo_concentration_model,
            halo_concentration_params or {},
        )

        self.sd_bias_model, self.sd_bias_params = sd_bias_model, sd_bias_params or {}
        self.exclusion_model, self.exclusion_params = (
            exclusion_model,
            exclusion_params or {},
        )

        # Note that these values are only chosen for accuracy, especially
        # for generating power spectra as hankel transforms.
        self._logr_table_min = -3
        self._logr_table_max = 2.5
        self.dr_table = dr_table

        self.rmin = rmin
        self.rmax = rmax
        self.rnum = rnum
        self.rlog = rlog

        self.hm_logk_min = hm_logk_min
        self.hm_logk_max = hm_logk_max
        self.hm_dlog10k = hm_dlog10k

        self.hc_spectrum = hc_spectrum
        self.force_1halo_turnover = force_1halo_turnover
        self.force_unity_dm_bias = force_unity_dm_bias
        self.colossus_params = colossus_params or {}

    # ===============================================================================
    # Parameters
    # ===============================================================================
    def validate(self):
        super().validate()
        assert self.rmin < self.rmax, f"rmin >= rmax: {self.rmin}, {self.rmax}"
        assert len(self.r) > 0, "r has length zero!"
        assert self.hm_logk_min < self.hm_logk_max, (
            f"hm_logk_min >= hm_logk_max: {self.hm_logk_min}, {self.hm_logk_max}"
        )
        assert len(self.k_hm) > 0, "k_hm has length zero!"
        assert self._logr_table_min < self._logr_table_max, (
            f"_logr_table_min >= logr_table_max: {self._logr_table_min}, {self._logr_table_max}"
        )

    @parameter("model")
    def bias_model(self, val):
        """Bias Model."""
        return get_mdl(val, "Bias")

    @parameter("param")
    def bias_params(self, val):
        """Dictionary of parameters for the Bias model."""
        return val

    @parameter("switch")
    def hc_spectrum(self, val):
        """The spectrum with which the halo-centre power spectrum is identified.

        Choices are 'linear', 'nonlinear', 'filtered-lin' or 'filtered-nl'.
        'filtered' spectra are filtered with a real-space top-hat window
        function at a scale of 2 Mpc/h, which ensures that haloes
        do not overlap on scales small than this.
        """
        if val not in ["linear", "nonlinear", "filtered-lin", "filtered-nl"]:
            raise ValueError(
                "hc_spectrum must be one of linear, nonlinear, filtered-lin and filtered-nl"
            )
        return val

    @parameter("model")
    def halo_profile_model(self, val):
        """The halo density halo_profile model."""
        return get_mdl(val, "Profile")

    @parameter("param")
    def halo_profile_params(self, val):
        """Dictionary of parameters for the Profile model."""
        return val

    @parameter("model")
    def halo_concentration_model(self, val):
        """A halo_concentration-mass relation."""
        return get_mdl(val, CMRelation)

    @parameter("param")
    def halo_concentration_params(self, val):
        """Dictionary of parameters for the concentration model."""
        return val

    @parameter("switch")
    def rmin(self, val):
        """Minimum length scale."""
        return val

    @parameter("res")
    def rmax(self, val):
        """Maximum length scale."""
        val = float(val)
        if val > 10**self._logr_table_max:
            warnings.warn(
                "rmax is larger than the interpolation table maximum "
                f"[{10**self._logr_table_max:.2e}]. Larger values will yield zero "
                "correlation.",
                stacklevel=2,
            )
        return val

    @parameter("res")
    def rnum(self, val):
        """Number of r bins."""
        return int(val)

    @parameter("option")
    def rlog(self, val):
        """If True, r bins are logarithmically distributed."""
        return bool(val)

    @parameter("res")
    def dr_table(self, val):
        """The width of r bin."""
        return float(val)

    @parameter("res")
    def hm_dlog10k(self, val):
        """The width of k bin in log10."""
        return float(val)

    @parameter("res")
    def hm_logk_min(self, val):
        """The minimum k bin in log10."""
        return float(val)

    @parameter("res")
    def hm_logk_max(self, val):
        """The maximum k bin in log10."""
        return float(val)

    @parameter("model")
    def sd_bias_model(self, val):
        """Model of Scale Dependant Bias."""
        if val is None:
            return None
        else:
            return get_mdl(val, "ScaleDepBias")

    @parameter("param")
    def sd_bias_params(self, val):
        """Dictionary of parameters for Scale Dependant Bias."""
        return val

    @parameter("switch")
    def force_1halo_turnover(self, val):
        """Suppress 1-halo power on scales larger than a few virial radii."""
        return bool(val)

    @parameter("model")
    def exclusion_model(self, val):
        """A string identifier for the type of halo exclusion used (or None)."""
        if val is None:
            val = "NoExclusion"
        return get_mdl(val, "Exclusion")

    @parameter("param")
    def colossus_params(self, val):
        """Options for colossus cosmology not set/derived in the astropy cosmology."""
        return val

    @parameter("param")
    def exclusion_params(self, val):
        """Dictionary of parameters for the Exclusion model."""
        return val

    # ===========================================================================
    # Basic Quantities
    # ===========================================================================
    @cached_quantity
    def _r_table(self):
        """A high-resolution, high-range table of r values for internal interpolation."""
        return 10 ** np.arange(self._logr_table_min, self._logr_table_max, self.dr_table)

    @cached_quantity
    def colossus_cosmo(self):
        """
        An instance of a COLOSSUS cosmology, which can be used to perform various
        COLOSSUS operations.
        """
        return astropy_to_colossus(
            self.cosmo, sigma8=self.sigma_8, ns=self.n, **self.colossus_params
        )

    @cached_quantity
    def k_hm(self):
        """The wave-numbers at which halo-model power spectra are calculated.

        Typically smaller in range than k for linear theory.
        """
        return 10 ** np.arange(self.hm_logk_min, self.hm_logk_max, self.hm_dlog10k)

    @cached_quantity
    def r(self):
        """Scales at which correlation functions are computed [Mpc/h]."""
        if hasattr(self.rmin, "__len__"):
            return np.array(self.rmin)
        elif self.rlog:
            return np.exp(np.linspace(np.log(self.rmin), np.log(self.rmax), self.rnum))
        else:
            return np.linspace(self.rmin, self.rmax, self.rnum)

    @cached_quantity
    def bias(self):
        """The halo bias as a function of halo mass."""
        return self.bias_model(
            nu=self.nu,
            delta_c=self.delta_c,
            m=self.m,
            mstar=self.mass_nonlinear,
            delta_halo=self.halo_overdensity_mean,
            n=self.n,
            cosmo=self.cosmo,
            sigma_8=self.sigma_8,
            n_eff=self.n_eff,
            **self.bias_params,
        )

    @cached_quantity
    def halo_concentration(self):
        """The concentration-mass relation."""
        this_filter = copy(self.filter)
        this_filter.power = self._power0
        this_profile = self.halo_profile_model(
            cm_relation=None,
            mdef=self.mdef,
            z=self.z,
            **self.halo_profile_params,
        )

        return self.halo_concentration_model(
            filter0=this_filter,
            growth=self.growth,
            delta_c=self.delta_c,
            profile=this_profile,
            cosmo=Cosmology(cosmo_model=self.cosmo),
            mdef=self.mdef,
            **self.halo_concentration_params,
        )

    @cached_quantity
    def halo_profile(self):
        """A class containing the elements necessary to calculate halo halo_profile quantities."""
        return self.halo_profile_model(
            cm_relation=self.halo_concentration,
            mdef=self.mdef,
            z=self.z,
            **self.halo_profile_params,
        )

    @cached_quantity
    def sd_bias(self):
        """A class containing relevant methods to calculate scale-dependent bias corrections."""
        if self.sd_bias_model is None:
            return None
        else:
            return self.sd_bias_model(
                self.corr_halofit_mm_fnc(self._r_table), **self.sd_bias_params
            )

    @cached_quantity
    def halo_bias(self):
        """Halo bias."""
        return self.bias.bias()

    @cached_quantity
    def cmz_relation(self):
        """Concentration-mass-redshift relation."""
        return self.halo_concentration.cm(self.m, self.z)

    # ===========================================================================
    # Halo/DM Statistics
    # ===========================================================================
    @cached_quantity
    def sd_bias_correction(self):
        """Return the correction for scale dependancy of bias."""
        if self.sd_bias is not None:
            return self.sd_bias.bias_scale()
        else:
            return None

    @cached_quantity
    def _power_halo_centres_fnc(self):
        """
        Power spectrum of halo centres, unbiased.

        Notes
        -----
        This defines the halo-centre power spectrum, which is a part of the 2-halo
        term calculation. Formally, we make the assumption that the halo-centre
        power spectrum is linearly biased, and this function returns

        .. math :: P^{hh}_c (k) /(b_1(m_1)b_2(m_2))

        """
        if self.hc_spectrum == "filtered-lin":
            f = TopHat(None, None)
            p = self.power * f.k_space(self.k * 2.0)
            first_zero = np.where(p <= 0)[0][0]
            p[first_zero:] = 0
            return tools.ExtendedSpline(
                self.k,
                p,
                lower_func=self.linear_power_fnc,
                upper_func=tools._zero,
                match_lower=False,
            )
        elif self.hc_spectrum == "filtered-nl":
            f = TopHat(None, None)
            p = self.nonlinear_power * f.k_space(self.k * 3.0)
            first_zero = np.where(p <= 0)[0][0]
            p[first_zero:] = 0
            return tools.ExtendedSpline(
                self.k,
                p,
                lower_func=self.nonlinear_power_fnc,
                upper_func=tools._zero,
                match_lower=False,
            )
        elif self.hc_spectrum == "linear":
            return self.linear_power_fnc
        elif self.hc_spectrum == "nonlinear":
            warnings.warn(
                "Using halofit for tracer stats is only valid up to"
                + " quasi-linear scales k<~1 (h/Mpc).",
                stacklevel=2,
            )
            return self.nonlinear_power_fnc
        else:
            raise ValueError("hc_spectrum was specified incorrectly!")

    @cached_quantity
    def linear_power_fnc(self):
        """A callable returning the linear power as a function of k (in h/Mpc)."""
        return tools.ExtendedSpline(
            self.k,
            self.power,
            lower_func=lambda k: k**self.n,
            upper_func="power_law",
            domain=(0, np.inf),
        )

    @cached_quantity
    def nonlinear_power_fnc(self):
        """A callable returning the nonlinear (halofit) power as a function of k (in h/Mpc)."""
        return tools.ExtendedSpline(
            self.k,
            self.nonlinear_power,
            lower_func=lambda k: k**self.n,
            upper_func="power_law",
            domain=(0, np.inf),
        )

    @cached_quantity
    def corr_linear_mm_fnc(self):
        """A callable returning the linear auto-correlation function of dark matter."""
        corr = tools.hankel_transform(self.linear_power_fnc, self._r_table, "r")
        return tools.ExtendedSpline(
            self._r_table,
            corr,
            lower_func="power_law",
            upper_func=lambda x: np.zeros_like(x),
        )

    @cached_quantity
    def corr_linear_mm(self):
        """The linear auto-correlation function of dark matter."""
        return self.corr_linear_mm_fnc(self.r)

    @cached_quantity
    def corr_halofit_mm_fnc(self):
        """A callable returning the nonlinear auto-correlation function of dark matter."""
        corr = tools.hankel_transform(self.nonlinear_power_fnc, self._r_table, "r")
        return tools.ExtendedSpline(
            self._r_table,
            corr,
            lower_func="power_law",
            upper_func=lambda x: np.zeros_like(x),
        )

    @cached_quantity
    def corr_halofit_mm(self):
        """The nonlinear (from halofit) auto-correlation function of dark matter."""
        return self.corr_halofit_mm_fnc(self.r)

    @cached_quantity
    def _corr_mm_base_fnc(self):
        """A callable returning the matter correlation function used throughout the calculations."""
        if self.hc_spectrum == "linear":
            return self.corr_linear_mm_fnc
        elif self.hc_spectrum in ["nonlinear", "filtered-nl"]:
            return self.corr_halofit_mm_fnc

    def power_hh(self, k, mmin=None, mmax=None, mmin2=None, mmax2=None):
        """
        The halo-centre power spectrum of haloes in a given mass range.

        The power of a given pair of halo masses is assumed to be linearly biased,
        :math:`P_hh(k) = b(m_1)b(m_2)P_{lin}(k)`

        Parameters
        ----------
        k : np.ndarray
            Array of wavenumbers. Units h/Mpc.
        mmin : real, default :attr:`.Mmin`
            The minimum halo mass of the range (for the first of the halo pairs).
            Note: masses here are log10 masses.
        mmax : real, default :attr:`.Mmax`
            The maximum halo mass of the range (for the first of the halo pairs).
            If a single halo mass is desired, set mmax==mmin.
        mmin2 : real, default `None`
            The minimum halo mass of the range (for the second of the halo pairs).
            By default, takes the same value as `mmin`.
        mmax : real, default `None`
            The maximum halo mass of the range (for the second of the halo pairs).
            By default, takes the same value as `mmax`.
        """
        if mmin is None:
            mmin = self.Mmin
        if mmax is None:
            mmax = self.Mmax
        if mmin2 is None:
            mmin2 = mmin
        if mmax2 is None:
            mmax2 = mmax

        if mmin == mmax or mmin2 == mmax2:
            spl = spline(np.log10(self.m), self.bias)

        def get_b(mn, mx):
            if mn == mx:
                return spl(mn)

            mask = np.logical_and(self.m >= 10**mn, self.m <= 10**mx)
            return intg.simpson(
                self.halo_bias[mask] * self.dndm[mask], x=self.m[mask]
            ) / intg.simpson(self.dndm[mask], x=self.m[mask])

        return get_b(mmin, mmax) * get_b(mmin2, mmax2) * self._power_halo_centres_fnc(k)

    # ===========================================================================
    # Halo Profile cached quantities
    # ===========================================================================
    @cached_quantity
    def halo_profile_ukm(self):
        """Mass-normalised fourier halo profile, with shape (len(k), len(m))."""
        return self.halo_profile.u(self.k, self.m, c=self.cmz_relation)

    @cached_quantity
    def halo_profile_rho(self):
        """Mass-normalised halo density profile, with shape (len(r), len(m))."""
        return self.halo_profile.rho(self._r_table, self.m, norm="m", c=self.cmz_relation)

    @cached_quantity
    def halo_profile_lam(self):
        """Mass-normalised halo profile self-convolution, with shape (len(r), len(m))."""
        if self.halo_profile.has_lam:
            return self.halo_profile.lam(self._r_table, self.m, c=self.cmz_relation)
        else:
            return None

    # ===========================================================================
    # 2-point DM statistics
    # ===========================================================================
    def _do_1halo_integral(self, max_mmin, integrand, mean_dens):
        """Do the 1-halo integral for some quantity, doing the turnover trick."""
        dens_min = 4 * np.pi * self.mean_density0 * self.halo_overdensity_mean / 3
        p = np.zeros_like(self.k)
        for i, (k, integ) in enumerate(zip(self.k, integrand)):
            if self.force_1halo_turnover:
                r = np.pi / k / 10  # The 10 is a complete heuristic hack.
                mmin = max(max_mmin, dens_min * r**3)
            else:
                mmin = max_mmin

            p[i] = tools.spline_integral(self.m, integ, xmin=mmin)

        return p / mean_dens**2

    @cached_quantity
    def power_1h_auto_matter_fnc(self):
        """A callable returning the halo model 1-halo DM auto-power spectrum."""
        p = self._do_1halo_integral(
            max_mmin=self.m[0],
            integrand=self.dndm * self.m**2 * self.halo_profile_ukm**2,
            mean_dens=self.mean_density0,
        )

        return tools.ExtendedSpline(
            self.k,
            p,
            lower_func=tools._zero if self.force_1halo_turnover else "boundary",
            upper_func="power_law",
        )

    @property
    def power_1h_auto_matter(self):
        """The halo model-derived nonlinear 1-halo dark matter auto-power spectrum."""
        return self.power_1h_auto_matter_fnc(self.k_hm)

    @cached_quantity
    def corr_1h_auto_matter_fnc(self):
        """A callable returning the halo model 1-halo DM auto-correlation function."""
        if self.halo_profile.has_lam:
            lam = self.halo_profile_lam
            integrand = self.dndm * self.m**3 * lam

            table = (
                intg.trapezoid(integrand, dx=np.log(10) * self.dlog10m) / self.mean_density0**2 - 1
            )
        else:
            table = tools.hankel_transform(self.power_1h_auto_matter_fnc, self._r_table, "r")

        return tools.ExtendedSpline(
            self._r_table,
            table,
            lower_func="power_law",
            upper_func=lambda x: np.zeros_like(x),
        )

    @property
    def corr_1h_auto_matter(self):
        """The halo model 1-halo dark matter auto-correlation function."""
        return self.corr_1h_auto_matter_fnc(self.r)

    @cached_quantity
    def _matter_exclusion(self):
        densityfunc = self.dndm * self.m / self.mean_density_in_halos

        if self.sd_bias_model is not None:
            bias = np.outer(self.sd_bias_correction, self.halo_bias)
        else:
            bias = self.halo_bias

        return self.exclusion_model(
            m=self.m,
            density=densityfunc,
            power_integrand=densityfunc * self.halo_profile_ukm,
            bias=bias,
            r=self._r_table,
            halo_density=self.halo_overdensity_mean * self.mean_density0,
            **self.exclusion_params,
        )

    def _get_power_2h_primitive(
        self,
        exclusion: Exclusion,
        effective_bias: float,
        debias: bool = True,
    ) -> tuple[Callable, np.ndarray]:
        """Get the 2-halo term of an auto-power spectrum.

        This is 'primitive' because it can be 2D, i.e. it can have an r-based scale
        dependence based either on scale dependent bias or halo exclusion.
        """
        # Instead of having a 2D function here we *could* just get the r-based functions
        # in k-space and convolve them. But I'm not sure that's a more efficient option.
        intg = exclusion.integrate()
        phh = self._power_halo_centres_fnc(self.k)

        # See Tinker+05 Appendix B for details on using the modified density in halo
        # exclusion models.
        intg = (intg.T * exclusion.density_mod).T

        # Now, we need to debias the results. Mostly, this is for DM correlations, and
        # the point is that even if one uses a perfect pair of HMF/bias, the numerical
        # integral of m*dndm*bias might not equal the mean density, just because we use
        # a limited range. So, we normalize this so that we definitely have a unity bias.
        # For tracer power spectra, the effective bias passed in should not be unity, but
        # instead should fix itself to the numerically-calculated effective bias.
        if debias:
            bias = exclusion.bias[-1] if exclusion.bias.ndim == 2 else exclusion.bias

            eff_bias = tools.spline_integral(exclusion.m, exclusion.density * bias, log=True)
            intg *= (effective_bias / eff_bias) ** 2

        if intg.ndim == 2:
            return [
                tools.ExtendedSpline(
                    self.k,
                    x * phh,
                    lower_func=self.linear_power_fnc,
                    match_lower=True,
                    upper_func="power_law"
                    if (self.exclusion_model == NoExclusion and "filtered" not in self.hc_spectrum)
                    else tools._zero,
                )
                for i, x in enumerate(intg)
            ]
        else:
            return tools.ExtendedSpline(
                self.k,
                intg * phh,
                lower_func=self.linear_power_fnc,
                match_lower=True,
                upper_func="power_law"
                if (self.exclusion_model == NoExclusion and "filtered" not in self.hc_spectrum)
                else tools._zero,
            )

    def _get_corr_2h_auto_fnc(
        self, exclusion: Exclusion, effective_bias, debias=True
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """Get a callable returning the 2-halo term of an auto-correlation."""
        power_primitive = self._get_power_2h_primitive(exclusion, effective_bias, debias=debias)

        if len(power_primitive) == 1:
            # In the case that there is no scale-dependence from SD bias or exclusion
            power_primitive = power_primitive[0]

        # Need to set h smaller here because this might need to be transformed back
        # to power.
        corr = tools.hankel_transform(power_primitive, self._r_table, "r", h=1e-4)

        lower_func = "power_law" if isinstance(Exclusion, NoExclusion) else tools._zero
        return tools.ExtendedSpline(
            self._r_table, corr, lower_func=lower_func, upper_func=tools._zero
        )

    @cached_quantity
    def corr_2h_auto_matter_fnc(
        self,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """A callable returning the 2-halo term of the matter auto-correlation at arbitrary k."""
        tools.norm_warn(self)
        return self._get_corr_2h_auto_fnc(
            self._matter_exclusion,
            self.bias_effective_matter,
            debias=self.force_unity_dm_bias,
        )

    @property
    def corr_2h_auto_matter(
        self,
    ) -> Callable[[float | np.ndarray], float | np.ndarray]:
        """The 2-halo term of the matter auto-correlation."""
        return self.corr_2h_auto_matter_fnc(self.r)

    def _get_power_2h_auto_fnc(self, exclusion, effective_bias, debias=True):
        """Get the halo model 2-halo matter auto-power spectrum."""
        if self.exclusion_model is NoExclusion and self.sd_bias_model is None:
            # Here we have a 1D primitive power, so we can just return it.
            return self._get_power_2h_primitive(exclusion, effective_bias, debias=debias)[0]

        # Otherwise, first calculate the correlation function.
        corr = self._get_corr_2h_auto_fnc(exclusion, effective_bias, debias=debias)
        out = tools.hankel_transform(corr, self.k, "k", h=0.001)

        # Everything below about k=1e-2 is essentially just the linear power biased,
        # and the hankel transform stops working at some small k.
        mask = self.k_hm < 1e-2
        if np.any(mask):
            warnings.warn(
                "power_2h_auto_tracer for k < 1e-2 is not computed directly, but "
                "is rather just the linear power * effective bias.",
                stacklevel=2,
            )
            out[mask] = self.power[mask] * effective_bias

        return tools.ExtendedSpline(self.k, out, lower_func="power_law", upper_func=tools._zero)

    @cached_quantity
    def power_2h_auto_matter_fnc(self) -> np.ndarray:
        """Return the halo model 2-halo matter auto-power spectrum at k."""
        tools.norm_warn(self)
        return self._get_power_2h_auto_fnc(
            self._matter_exclusion,
            self.bias_effective_matter,
            debias=self.force_unity_dm_bias,
        )

    @property
    def power_2h_auto_matter(self) -> np.ndarray:
        """The halo model 2-halo matter auto-power spectrum at :attr:`k_hm`."""
        return self.power_2h_auto_matter_fnc(self.k_hm)

    @cached_quantity
    def corr_auto_matter_fnc(self):
        """A callable returning the halo-model DM auto-correlation function."""
        return lambda r: self.corr_1h_auto_matter_fnc(r) + self.corr_2h_auto_matter_fnc(r) + 1

    @property
    def corr_auto_matter(self):
        """The halo-model-derived nonlinear dark matter auto-correlation function."""
        return self.corr_auto_matter_fnc(self.r)

    @cached_quantity
    def power_auto_matter_fnc(self):
        """A callable returning the halo-model DM auto-power spectrum."""
        return lambda k: self.power_1h_auto_matter_fnc(k) + self.power_2h_auto_matter_fnc(k)

    @property
    def power_auto_matter(self):
        """The halo-model-derived nonlinear dark power auto-power spectrum."""
        return self.power_auto_matter_fnc(self.k_hm)

    def _get_naive_bias_effective(self, density, mean_density, mmin=None):
        return tools.spline_integral(self.m, density, xmin=mmin, log=True) / mean_density

    @cached_quantity
    def bias_effective_matter(self) -> float:
        """The effective bias of matter in DM halos.

        This *should* be unity for models in which all dark matter is encapsulated in
        halos. However, sometimes the pair of HMF/bias models chosen are not pairs
        in the peak-background split sense, and will not (even in principle) yield an
        integral of unity. Also, even when in principle the value should be unity, the
        numerical integral is over a limited range of mass, and will generally not
        account for all the mass in the Universe.

        To account for this, if :attr:`force_unity_dm_bias` is True, the returned
        bias will be unity if the bias/HMF models are a pair. If they are not a pair,
        we still somewhat account for the limited mass range by returning the integral
        normalized by the mean density in halos greater than our minimum mass.

        If :attr:`force_unity_dm_bias` is False, none of these fudges are performed.
        """
        # If we have a normalized HMF and its bias pair, it's always unity.
        if self.force_unity_dm_bias:
            return 1.0

        return self._get_naive_bias_effective(
            self.m * self.dndm * self.halo_bias, self.mean_density_in_halos
        )

    @cached_quantity
    def mean_density_in_halos(self):
        return self.rho_gtm[0]


class TracerHaloModel(DMHaloModel):
    """
    Describe spatial statistics of a tracer population (eg. galaxies) using a HOD.

    All of the quantities in :class:`~DMHaloModel` are available here, with the addition
    of several more explicitly for the tracer population, including cross-correlations.

    Note that the flexibility of prescribing different models for the tracer population
    than the underlying DM halo population is afforded for such components as the
    profile and concentration-mass relation. By default,these are set to be the same as
    the DM models. This may be useful if eg. galaxies are not expected to trace
    the underlying dark matter density within a halo.

    Note that all `*_model` parameters can be a string or a class of the type
    described below. If a string, it should be the name of a class that must exist
    in the relevant module within ``halomod``.

    Parameters
    ----------
    hod_model : str or :class:`~hod.HOD` subclass, optional
        A model for the halo occupation distribution.
    hod_params : dict, optional
        Parameters for the HOD model.
    tracer_profile_model : str or :class:`~profiles.Profile` subclass, optional
        A density profile model for the abundance of the tracer within haloes of a
        given mass.
    tracer_profile_params : dict, optional
        Parameters for the tracer density profile model.
    tracer_concentration_model : str or :class:`~concentration.CMRelation` subclass, optional
        A concentration-mass relation supporting the tracer profile.
    tracer_concentration_params : dict, optional
        Parameters for the tracer CM relation.
    tracer_density: float, optional
        Total density of the tracer, in the units specified by the HOD model. This
        can be used to set the minimum halo mass of the HOD.

    Other Parameters
    ----------------
    All other parameters are passed to :class:`~DMHaloModel`.
    """

    def __init__(
        self,
        hod_model="Zehavi05",
        hod_params=None,
        tracer_profile_model=None,
        tracer_profile_params=None,
        tracer_concentration_model=None,
        tracer_concentration_params=None,
        tracer_density=None,
        **halomodel_kwargs,
    ):
        super().__init__(**halomodel_kwargs)

        # Initially save parameters to the class.
        self.hod_params = hod_params or {}
        self.hod_model = hod_model
        self.tracer_profile_model, self.tracer_profile_params = (
            tracer_profile_model,
            tracer_profile_params or {},
        )
        self.tracer_concentration_model, self.tracer_concentration_params = (
            tracer_concentration_model,
            tracer_concentration_params or {},
        )

        # A special argument, making it possible to define M_min by mean density
        self.tracer_density = tracer_density

        # Find mmin if we want to
        if tracer_density is not None:
            mmin = self._find_m_min(tracer_density)
            self.hod_params = {"M_min": mmin}

    def update(self, **kwargs):
        """Updates any parameter passed."""
        if "tracer_density" in kwargs:
            self.tracer_density = kwargs.pop("tracer_density")
        elif "hod_params" in kwargs and "M_min" in kwargs["hod_params"]:
            self.tracer_density = None

        super().update(**kwargs)

        if self.tracer_density is not None:
            mmin = self._find_m_min(self.tracer_density)
            self.hod_params = {"M_min": mmin}

    # ===============================================================================
    # Parameters
    # ===============================================================================
    def validate(self):
        super().validate()
        assert np.sum(self._tm) > 1, "the HOD model you've supplied masks out all given masses!"

    @parameter("param")
    def tracer_density(self, val):
        """Mean density of the tracer, ONLY if passed directly."""
        return val

    @parameter("param")
    def hod_params(self, val: dict):
        """Dictionary of parameters for the HOD model."""
        return val

    @parameter("model")
    def hod_model(self, val):
        """:class:`~hod.HOD` class."""
        return get_mdl(val, "HOD")

    @parameter("param")
    def tracer_profile_params(self, val: dict):
        """Dictionary of parameters for the tracer Profile model."""
        val = val or {}
        assert isinstance(val, dict)
        return val

    @parameter("model")
    def tracer_profile_model(self, val):
        """The tracer density halo_profile model."""
        if val is None:
            return val
        return get_mdl(val, "Profile")

    @parameter("model")
    def tracer_concentration_model(self, val):
        """The tracer concentration-mass relation."""
        if val is None:
            return val
        return get_mdl(val, CMRelation)

    @parameter("param")
    def tracer_concentration_params(self, val):
        """Dictionary of parameters for tracer concentration-mass relation."""
        val = val or {}
        assert isinstance(val, dict)
        return val

    @parameter("switch")
    def force_1halo_turnover(self, val):
        """Suppress 1-halo power on scales larger than a few virial radii."""
        return bool(val)

    # ===========================================================================
    # Basic Quantities
    # ===========================================================================
    # THE FOLLOWING IS LEFT IN AS A REMINDER NEVER TO DO IT
    # CHANGING THE MINIMUM MASS DYNAMICALLY DESTROYS MANY THINGS, LIKE THE ABILITY TO
    # CROSS-CORRELATE TWO CLASSES.
    # @cached_quantity
    # def mmin(self):
    #     "This is the true minimum mass for this framework"
    #     return min(self.Mmin, self.hod.mmin)
    #
    # @cached_quantity
    # def m(self):
    #     return 10 ** np.arange(self.mmin, self.Mmax, self.dlog10m)

    @cached_quantity
    def _tm(self):
        """
        A tracer mask -- i.e. a mask on mass which restricts the range to those where
        the tracer exists for the given HOD.
        """
        if self.hod.mmin is None:
            return self.m >= self.m.min()

        if self.hod.mmin < self.Mmin:
            warnings.warn(
                "The HOD is defined to lower masses than currently calculated. "
                "Please set Mmin lower.",
                stacklevel=2,
            )

        return self.m >= 10**self.hod.mmin

    @cached_quantity
    def tracer_concentration(self):
        """Concentration-mass relation model instance."""
        if self.tracer_concentration_model is None:
            return self.halo_concentration

        this_filter = copy(self.filter)
        this_filter.power = self._power0

        if self.tracer_profile_model is None:
            this_profile = self.halo_profile_model(
                cm_relation=None,
                mdef=self.mdef,
                z=self.z,
                **self.halo_profile_params,
            )
        else:
            # Need to get the tracer profile params if it wasn't given.
            # If we have the same tracer and halo profiles, use the halo profile
            # params. Otherwise, don't give any params.
            if (
                not self.tracer_profile_params
                and self.tracer_profile_model == self.halo_profile_model
            ):
                tr_params = self.halo_profile_params
            else:
                tr_params = self.tracer_profile_params

            this_profile = self.tracer_profile_model(
                cm_relation=None,
                mdef=self.mdef,
                z=self.z,
                **tr_params,
            )

        if (
            not self.tracer_concentration_params
            and self.tracer_concentration_model == self.halo_concentration_model
        ):
            tr_params = self.halo_concentration_params
        else:
            tr_params = self.tracer_concentration_params

        return self.tracer_concentration_model(
            cosmo=Cosmology(self.cosmo),
            filter0=this_filter,
            growth=self.growth,
            delta_c=self.delta_c,
            profile=this_profile,
            mdef=self.mdef,
            **tr_params,
        )

    @cached_quantity
    def tracer_cmz_relation(self):
        """The concentrations corresponding to :meth:`m`."""
        return self.tracer_concentration.cm(self.m, self.z)

    @cached_quantity
    def tracer_profile(self):
        """Object to calculate quantities of the tracer profile."""
        if self.tracer_profile_model is None:
            return self.halo_profile

        if not self.tracer_profile_params and self.tracer_profile_model == self.halo_profile_model:
            tr_params = self.halo_profile_params
        else:
            tr_params = self.tracer_profile_params

        return self.tracer_profile_model(
            cm_relation=self.tracer_concentration,
            mdef=self.mdef,
            z=self.z,
            **tr_params,
        )

    @cached_quantity
    def hod(self):
        """A class representing the HOD."""
        return self.hod_model(
            cosmo=self.cosmo,
            cm_relation=self.tracer_concentration,
            profile=self.tracer_profile,
            mdef=self.mdef,
            **self.hod_params,
        )

    # ===========================================================================
    # Basic HOD Quantities
    # ===========================================================================
    @cached_quantity
    def total_occupation(self):
        """The mean total occupation of the tracer as a function of halo mass."""
        return self.hod.total_occupation(self.m)

    @cached_quantity
    def satellite_occupation(self):
        """The mean satellite occupation of the tracer as a function of halo mass."""
        return self.hod.satellite_occupation(self.m)

    @cached_quantity
    def central_occupation(self):
        """The mean central occupation of the tracer as a function of halo mass."""
        return self.hod.central_occupation(self.m)

    @property
    def _central_occupation(self):
        """The central occupation to use when integrating over mass.

        The reason is because if a sharp cut happens, we need to make sure the spline
        carries all the way through past the mmin as unity. Setting the pixel below mmin
        to zero causes a bad spline.
        """
        return (
            np.ones_like(self.m)
            if (self.hod.sharp_cut and (self.hod._central or self.hod.central_condition_inherent))
            else self.central_occupation
        )

    @property
    def _total_occupation(self):
        """The total occupation to use when integrating over mass.

        See _central_occupation for why.
        """
        return self._central_occupation + self.satellite_occupation

    @property
    def tracer_mmin(self):
        """The minimum halo mass of integrals over the tracer population.

        This is a little tricky, because HOD's which don't enforce the central condition,
        even if they have a sharp cut at mmin, should not stop the integral at the
        central's Mmin, but should rather continue to pick up the satellites in lower
        mass haloes.
        """
        if self.hod.sharp_cut and (self.hod._central or self.hod.central_condition_inherent):
            return 10**self.hod.mmin
        else:
            return None

    # ===========================================================================
    # Derived HOD Quantities
    # ===========================================================================
    @cached_quantity
    def mean_tracer_den(self):
        """
        The mean density of the tracer.

        This is always the *integrated* density. If `tracer_density` is supplied to the
        constructor, that value can be found as :meth:`.tracer_density`. It should be
        very close to this value.
        """
        return tools.spline_integral(
            self.m, self.dndm * self._total_occupation, xmin=self.tracer_mmin
        )

    @cached_quantity
    def mean_tracer_den_unit(self):
        """The mean density of the tracer, in the units defined in the HOD."""
        return self.mean_tracer_den * self.hod.unit_conversion(self.cosmo, self.z)

    @cached_quantity
    def bias_effective_tracer(self):
        """The tracer occupation-weighted halo bias factor (Tinker 2005)."""
        # Integrand is just the density of galaxies at mass m by bias
        b = tools.spline_integral(
            self.m,
            self.dndm * self._total_occupation * self.halo_bias,
            xmin=self.tracer_mmin,
        )
        return b / self.mean_tracer_den

    @cached_quantity
    def mass_effective(self):
        """Average host-halo mass (in log10 units)."""
        # Integrand is just the density of galaxies at mass m by m
        m = tools.spline_integral(
            self.m, self.m * self.dndm * self._total_occupation, xmin=self.tracer_mmin
        )

        return np.log10(m / self.mean_tracer_den)

    @cached_quantity
    def satellite_fraction(self):
        """The total fraction of tracers that are satellites.

        Note: this may not exist for every kind of tracer.
        """
        # Integrand is just the density of satellite galaxies at mass m
        s = tools.spline_integral(
            self.m, self.dndm * self.satellite_occupation, xmin=self.tracer_mmin
        )
        return s / self.mean_tracer_den

    @cached_quantity
    def central_fraction(self):
        """The total fraction of tracers that are centrals.

        Note: This may not exist for every kind of tracer.
        """
        return 1 - self.satellite_fraction

    @cached_quantity
    def tracer_density_m(self):
        """The total tracer density in halos of mass m."""
        return self.dndm * self.total_occupation

    # ===========================================================================
    # Tracer Profile cached quantities
    # ===========================================================================
    @cached_quantity
    def tracer_profile_ukm(self):
        """The mass-normalised fourier density profile of the tracer, shape (len(k), len(m))."""
        return self.tracer_profile.u(self.k, self.m, c=self.tracer_cmz_relation)

    @cached_quantity
    def tracer_profile_rho(self):
        """The mass-normalised density profile of the tracer, with shape (len(r), len(m))."""
        return self.tracer_profile.rho(self._r_table, self.m, norm="m", c=self.tracer_cmz_relation)

    @cached_quantity
    def tracer_profile_lam(self):
        """The mass-normalised profile self-convolution of the tracer, shape (len(r), len(m))."""
        if self.tracer_profile.has_lam:
            return self.tracer_profile.lam(self._r_table, self.m, c=self.tracer_cmz_relation)
        else:
            return None

    # ===========================================================================
    # 2-point tracer-tracer (HOD) statistics
    # ===========================================================================

    @cached_quantity
    def power_1h_ss_auto_tracer_fnc(self):
        """A callable returning the satellite-satellite part of
        the 1-halo term of the tracer auto-power spectrum.

        Note: May not exist for every kind of tracer.
        """
        p = self._do_1halo_integral(
            max_mmin=self.hod.mmin,
            integrand=self.tracer_profile_ukm**2 * self.dndm * self.hod.ss_pairs(self.m),
            mean_dens=self.mean_tracer_den,
        )

        return tools.ExtendedSpline(
            self.k,
            p,
            lower_func=tools._zero if self.force_1halo_turnover else "boundary",
            upper_func="power_law",
        )

    @property
    def power_1h_ss_auto_tracer(self):
        """The satellite-satellite part of the 1-halo term of the tracer auto-power spectrum.

        Note: May not exist for every kind of tracer.
        """
        return self.power_1h_ss_auto_tracer_fnc(self.k_hm)

    @cached_quantity
    def corr_1h_ss_auto_tracer_fnc(self):
        """
        A callable returning the satellite-satellite part of
        the 1-halo term of the tracer auto-correlation function.

        Note: May not exist for every kind of tracer.
        """
        ss_pairs = self.hod.ss_pairs(self.m)
        if self.tracer_profile.has_lam:
            c = np.zeros_like(self._r_table)
            for i, lam in enumerate(self.tracer_profile_lam):
                c[i] = tools.spline_integral(
                    self.m, lam * self.dndm * ss_pairs, xmin=self.tracer_mmin
                )
            c = c / self.mean_tracer_den**2 - 1

        else:
            c = tools.hankel_transform(self.power_1h_ss_auto_tracer_fnc, self._r_table, "r")
        return tools.ExtendedSpline(
            self._r_table, c, lower_func="power_law", upper_func=tools._zero
        )

    @property
    def corr_1h_ss_auto_tracer(self):
        """
        The satellite-satellite part of the 1-halo term of the tracer auto-correlation function.

        Note: May not exist for every kind of tracer.
        """
        return self.corr_1h_ss_auto_tracer_fnc(self.r)

    @cached_quantity
    def power_1h_cs_auto_tracer_fnc(self):
        """
        A callable returning the cen-sat part of
        the 1-halo term of the tracer auto-power spectrum.

        Note: May not exist for every kind of tracer.
        """
        p = self._do_1halo_integral(
            max_mmin=self.hod.mmin,
            integrand=self.dndm * 2 * self.hod.cs_pairs(self.m) * self.tracer_profile_ukm,
            mean_dens=self.mean_tracer_den,
        )

        return tools.ExtendedSpline(
            self.k,
            p,
            lower_func=tools._zero if self.force_1halo_turnover else "boundary",
            upper_func="power_law" if np.all(p[-10:] > 0) else tools._zero,
        )

    @property
    def power_1h_cs_auto_tracer(self):
        """The cen-sat part of the 1-halo term of the tracer auto-power spectrum.

        Note: May not exist for every kind of tracer.
        """
        return self.power_1h_cs_auto_tracer_fnc(self.k_hm)

    @cached_quantity
    def corr_1h_cs_auto_tracer_fnc(self):
        """A callable returning the cen-sat part of
        the 1-halo term of the tracer auto-correlation function.

        Note: May not exist for every kind of tracer.
        """
        c = np.zeros_like(self._r_table)
        cs_pairs = self.hod.cs_pairs(self.m)
        for i, rho in enumerate(self.tracer_profile_rho):
            c[i] = tools.spline_integral(
                self.m, self.dndm * 2 * cs_pairs * rho, xmin=self.tracer_mmin
            )

        c = c / self.mean_tracer_den**2 - 1

        return tools.ExtendedSpline(
            self._r_table, c, lower_func="power_law", upper_func=tools._zero
        )

    @property
    def corr_1h_cs_auto_tracer(self):
        """The cen-sat part of the 1-halo term of the tracer auto-correlation function.

        Note: May not exist for every kind of tracer.
        """
        return self.corr_1h_cs_auto_tracer_fnc(self.r)

    @cached_quantity
    def power_1h_auto_tracer_fnc(self):
        """A callable returning the total 1-halo term of the tracer auto power spectrum."""
        return lambda k: (self.power_1h_cs_auto_tracer_fnc(k) + self.power_1h_ss_auto_tracer_fnc(k))

    @property
    def power_1h_auto_tracer(self):
        """The total 1-halo term of the tracer auto power spectrum."""
        return self.power_1h_auto_tracer_fnc(self.k_hm)

    @cached_quantity
    def corr_1h_auto_tracer_fnc(self):
        """A callable returning the 1-halo term of the tracer auto correlations."""
        if self.tracer_profile.has_lam:
            c = np.zeros_like(self._r_table)

            ss_pairs = self.hod.ss_pairs(self.m)
            cs_pairs = self.hod.cs_pairs(self.m)
            for i, (rho, lam) in enumerate(zip(self.tracer_profile_rho, self.tracer_profile_lam)):
                c[i] = tools.spline_integral(
                    self.m,
                    self.dndm
                    * (ss_pairs * lam + 2 * cs_pairs * rho)
                    * (self._central_occupation if self.hod._central else 1),
                    xmin=self.tracer_mmin,
                )

            c /= self.mean_tracer_den**2

        else:
            try:
                return (
                    lambda r: self.corr_1h_cs_auto_tracer_fnc(r)
                    + self.corr_1h_ss_auto_tracer_fnc(r)
                    + 1
                )
            except AttributeError:
                c = tools.hankel_transform(self.power_1h_auto_tracer_fnc, self.r, "r")

        return tools.ExtendedSpline(
            self._r_table, c, lower_func="power_law", upper_func=tools._zero
        )

    @property
    def corr_1h_auto_tracer(self):
        """The 1-halo term of the tracer auto correlations."""
        return self.corr_1h_auto_tracer_fnc(self.r)

    @cached_quantity
    def _tracer_exclusion(self):
        densityfunc = self.dndm[self._tm] * self.total_occupation[self._tm] / self.mean_tracer_den

        if self.sd_bias_model is not None:
            bias = np.outer(self.sd_bias_correction, self.halo_bias)[:, self._tm]
        else:
            bias = self.halo_bias[self._tm]

        return self.exclusion_model(
            m=self.m[self._tm],
            density=densityfunc,
            power_integrand=densityfunc * self.tracer_profile_ukm[:, self._tm],
            bias=bias,
            r=self._r_table,
            halo_density=self.halo_overdensity_mean * self.mean_density0,
            **self.exclusion_params,
        )

    @cached_quantity
    def power_2h_auto_tracer_fnc(self):
        """Return the 2-halo term of the tracer auto-power spectrum at k."""
        return self._get_power_2h_auto_fnc(
            self._tracer_exclusion,
            self.bias_effective_tracer,
            debias=False,
        )

    @property
    def power_2h_auto_tracer(self):
        """The 2-halo term of the tracer auto-power spectrum."""
        return self.power_2h_auto_tracer_fnc(self.k_hm)

    @cached_quantity
    def corr_2h_auto_tracer_fnc(self):
        """A callable returning the 2-halo term of the tracer auto-correlation."""
        return self._get_corr_2h_auto_fnc(
            self._tracer_exclusion, self.bias_effective_tracer, debias=False
        )

    @property
    def corr_2h_auto_tracer(self):
        """The 2-halo term of the tracer auto-correlation."""
        return self.corr_2h_auto_tracer_fnc(self.r)

    @property
    def power_auto_tracer_fnc(self):
        return lambda k: (self.power_1h_auto_tracer_fnc(k) + self.power_2h_auto_tracer_fnc(k))

    @property
    def power_auto_tracer(self):
        """Auto-power spectrum of the tracer."""
        return self.power_auto_tracer_fnc(self.k_hm)

    @property
    def corr_auto_tracer_fnc(self):
        """A callable returning the tracer auto correlation function."""
        return lambda r: self.corr_1h_auto_tracer_fnc(r) + self.corr_2h_auto_tracer_fnc(r)

    @property
    def corr_auto_tracer(self):
        """The tracer auto correlation function."""
        return self.corr_auto_tracer_fnc(self.r)

    # ===========================================================================
    # Cross-correlations
    # ===========================================================================
    @cached_quantity
    def power_1h_cross_tracer_matter_fnc(self):
        """
        A callable returning the total 1-halo cross-power spectrum
        between tracer and matter.
        """
        p = np.zeros_like(self.k)
        for i, (ut, uh) in enumerate(zip(self.tracer_profile_ukm, self.halo_profile_ukm)):
            p[i] = tools.spline_integral(
                self.m,
                self.dndm
                * (uh * ut * self._total_occupation * self.m + uh * self.satellite_occupation),
                xmin=self.tracer_mmin,
            )

        p /= self.mean_tracer_den * self.mean_density0
        return tools.ExtendedSpline(self.k, p, lower_func="power_law", upper_func="power_law")

    @property
    def power_1h_cross_tracer_matter(self):
        """The total 1-halo cross-power spectrum between tracer and matter."""
        return self.power_1h_cross_tracer_matter_fnc(self.k_hm)

    @cached_quantity
    def corr_1h_cross_tracer_matter_fnc(self):
        """A callable returning the 1-halo cross-corr between tracer and matter."""
        corr = tools.hankel_transform(self.power_1h_cross_tracer_matter_fnc, self._r_table, "r")
        return tools.ExtendedSpline(
            self._r_table, corr, lower_func="power_law", upper_func=tools._zero
        )

    @property
    def corr_1h_cross_tracer_matter(self):
        """The 1-halo term of the cross correlation between tracer and matter."""
        return self.corr_1h_cross_tracer_matter_fnc(self.r)

    @cached_quantity
    def power_2h_cross_tracer_matter_fnc(self):
        """A callable returning the 2-halo cross-power between tracer and matter."""
        # Do this the simple way for now
        bt = np.zeros_like(self.k)
        bm = np.zeros_like(self.k)
        for i, (ut, um) in enumerate(zip(self.tracer_profile_ukm, self.halo_profile_ukm)):
            bt[i] = tools.spline_integral(
                self.m,
                self.dndm * self.halo_bias * self._total_occupation * ut,
                xmin=self.tracer_mmin,
            )
            bm[i] = tools.spline_integral(self.m, self.dndm * self.halo_bias * self.m * um)

        power = (
            bt
            * bm
            * self._power_halo_centres_fnc(self.k)
            / (self.mean_tracer_den * self.mean_density0)
        )

        return tools.ExtendedSpline(
            self.k,
            power,
            lower_func="power_law",
            upper_func="power_law" if "filtered" not in self.hc_spectrum else tools._zero,
        )

    @property
    def power_2h_cross_tracer_matter(self):
        """The 2-halo term of the cross-power spectrum between tracer and matter."""
        return self.power_2h_cross_tracer_matter_fnc(self.k_hm)

    @cached_quantity
    def corr_2h_cross_tracer_matter_fnc(self):
        """A callable returning the 2-halo cross-corr between tracer and matter."""
        corr = tools.hankel_transform(self.power_2h_cross_tracer_matter_fnc, self._r_table, "r")
        return tools.ExtendedSpline(
            self._r_table, corr, lower_func="power_law", upper_func=tools._zero
        )

    @property
    def corr_2h_cross_tracer_matter(self):
        """The 2-halo term of the cross-correlation between tracer and matter."""
        return self.corr_2h_cross_tracer_matter_fnc(self.r)

    @cached_quantity
    def power_cross_tracer_matter_fnc(self):
        """A callable returning cross-power spectrum of tracer and matter."""
        return lambda k: self.power_1h_cross_tracer_matter_fnc(
            k
        ) + self.power_2h_cross_tracer_matter_fnc(k)

    @property
    def power_cross_tracer_matter(self):
        """Cross-power spectrum between tracer and matter."""
        return self.power_cross_tracer_matter_fnc(self.k_hm)

    @cached_quantity
    def corr_cross_tracer_matter_fnc(self):
        """A callable returning the cross-correlation of tracer with matter."""
        return (
            lambda r: self.corr_1h_cross_tracer_matter_fnc(r)
            + self.corr_2h_cross_tracer_matter_fnc(r)
            + 1
        )

    @property
    def corr_cross_tracer_matter(self):
        """Cross-correlation of tracer with matter."""
        return self.corr_cross_tracer_matter_fnc(self.r)

    # ===========================================================================
    # Other utilities
    # ===========================================================================
    def _find_m_min(self, ng):
        """
        Calculate the minimum mass of a halo to contain a (central) galaxy
        based on a known mean galaxy density.
        """
        _ = self.power  # This just makes sure the power is gotten and copied
        c = self.clone(hod_params={"M_min": self.Mmin}, dlog10m=0.01)

        integrand = c.m[c._tm] * c.dndm[c._tm] * c.total_occupation[c._tm]

        density_message = (
            f"Maximum mean galaxy density exceeded. User input required density of {ng}, "
            "but maximum density (with HOD M_min == DM Mmin) is {}. "
            "Consider decreasing Mmin,or checking tracer_density."
        )
        if self.hod.sharp_cut:
            integral = intg.cumtrapz(integrand[::-1], dx=np.log(c.m[1] / c.m[0]))

            if integral[-1] < ng:
                raise NGException(density_message.format(integral[-1]))

            ind = np.where(integral > ng)[0][0]

            m = c.m[c._tm][::-1][1:][max(ind - 4, 0) : min(ind + 4, len(c.m))]
            integral = integral[max(ind - 4, 0) : min(ind + 4, len(c.m))]

            spline_int = spline(np.log(integral), np.log(m), k=3)
            mmin = spline_int(np.log(ng)) / np.log(10)
        else:
            # Anything else requires us to do some optimization unfortunately.
            integral = intg.simpson(integrand, dx=np.log(c.m[1] / c.m[0]))
            if integral < ng:
                raise NGException(density_message.format(integral))

            def model(mmin):
                c.update(hod_params={"M_min": mmin})
                integrand = c.m[c._tm] * c.dndm[c._tm] * c.total_occupation[c._tm]
                integral = intg.simpson(integrand, dx=np.log(c.m[1] / c.m[0]))
                return abs(integral - ng)

            res = minimize(model, 12.0, tol=1e-3, method="Nelder-Mead", options={"maxiter": 200})
            mmin = res.x[0]

        return mmin


# For compatibility
HaloModel = TracerHaloModel


class NGException(Exception):
    """Exception raised when mean tracer density errors in computation."""
