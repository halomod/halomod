"""
Main halo model module.

Contains Frameworks that combine all components necessary for halo model calculations (eg. mass function, bias,
concentration, halo profile).

Two main classes are provided: :class:`DMHaloModel` for dark-matter only halo models, and :class:`TracerHaloModel`
for halo models including a tracer population embedded in the dark matter haloes, via a HOD.

The :class:`HaloModel` class is provided as an alias of :class:`TracerHaloModel`.
"""
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.integrate as intg
import numpy as np
from scipy.optimize import minimize

from hmf import MassFunction, cached_quantity, parameter, Cosmology

# import hmf.tools as ht
from . import tools
from . import hod
from .concentration import CMRelation
from .halo_exclusion import Exclusion, NoExclusion


from copy import copy, deepcopy
from numpy import issubclass_
from hmf._internals._framework import get_model_
from . import profiles
from . import bias
from hmf.density_field.filters import TopHat
import warnings

from hmf.cosmology.cosmo import astropy_to_colossus


class DMHaloModel(MassFunction):
    """
    Dark-matter-only halo model class.

    This Framework is subclassed from hmf's ``MassFunction`` class, and operates in a similar manner.

    **kwargs: anything that can be used in the MassFunction class

    """

    rlog = True

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
        hc_spectrum="nonlinear",
        Mmin=0,
        Mmax=18,
        force_1halo_turnover=True,
        **hmf_kwargs,
    ):
        """
        Initializer for the class.

        Note that all `*_model` parameters can be a string or a class of the type described below. If a string,
        it should be the name of a class that must exist in the relevant module within `halomod`.

        Parameters
        ----------
        rmin : float or arry-like, optional
            Minimum length scale over which to calculate correlations, in Mpc/h. Alternatively, if an array,
            this is used to specify the entire array of scales and `rmax`, `rnum` and `rlog` are ignored.
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
        halo_concentration_model : str or :class:`~concentration.CMRelation` subclass, optional
            The model for the concentration-mass-redshift relation of the halos.
        halo_concentration_params : dict, optional
            Parameters for the concentration-mass relation (see its docstring for details)
        bias_model : str or :class:`~bias.Bias` subclass, optional
            The model of halo bias.
        bias_params : dict, optional
            Parameters for the bias model (see its docstring for details)
        sd_bias_model : str, None, or :class:`~bias.ScaleDepBias` subclass, optional
            A model for scale-dependent bias (as a function of `r`). Setting to None will use no scale-dependent bias.
        sd_bias_params : dict, optional
            Parameters for the scale-dependent bias model (see its docstring for details).
        exclusion_model : str, None or :class:`~halo_exclusion.Exclusion` subclass
            A model for how halo exclusion is calculated.
        exclusion_params : dict, optional
            Parameters for the halo exclusion model
        hc_spectrum : str, {'linear', 'nonlinear', 'filtered-nl', 'filtered-lin'}
            A choice for how the halo-centre power spectrum is defined. The "filtered" options arise from eg.
            Schneider, Smith et al. (2014).

        Other Parameters
        ----------------
        All other parameters are passed to :class:`~MassFunction`.
        """
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
        self.bias_model, self.bias_params = bias_model, bias_params or {}
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
        self.colossus_params = colossus_params or {}

    # ===============================================================================
    # Parameters
    # ===============================================================================
    @parameter("model")
    def bias_model(self, val):
        if not (isinstance(val, str) or issubclass_(val, bias.Bias)):
            raise ValueError("bias_model must be a subclass of bias.Bias")
        elif isinstance(val, str):
            return get_model_(val, "halomod.bias")
        else:
            return val

    @parameter("param")
    def bias_params(self, val):
        return val

    @parameter("switch")
    def hc_spectrum(self, val):
        if val not in ["linear", "nonlinear", "filtered-lin", "filtered-nl"]:
            raise ValueError(
                "hc_spectrum must be one of linear, nonlinear, filtered-lin and filtered-nl"
            )
        return val

    @parameter("model")
    def halo_profile_model(self, val):
        """The halo density halo_profile model"""
        if not (isinstance(val, str) or issubclass_(val, profiles.Profile)):
            raise ValueError(
                "halo_profile_model must be a subclass of profiles.Profile"
            )
        elif isinstance(val, str):
            return get_model_(val, "halomod.profiles")
        else:
            return val

    @parameter("param")
    def halo_profile_params(self, val):
        """Dictionary of parameters for the Profile model"""
        return val

    @parameter("model")
    def halo_concentration_model(self, val):
        """A halo_concentration-mass relation"""
        if not (isinstance(val, str) or issubclass_(val, CMRelation)):
            raise ValueError(
                "halo_concentration_model must be a subclass of halo_concentration.CMRelation"
            )
        elif isinstance(val, str):
            return get_model_(val, "halomod.concentration")
        else:
            return val

    @parameter("param")
    def halo_concentration_params(self, val):
        return val

    @parameter("switch")
    def rmin(self, val):
        return val

    @parameter("res")
    def rmax(self, val):
        val = float(val)
        if val > 10 ** self._logr_table_max:
            warnings.warn(
                f"rmax is larger than the interpolation table maximum [{10**self._logr_table_max:.2e}]. Larger values will yield zero correlation."
            )
        return val

    @parameter("res")
    def rnum(self, val):
        return int(val)

    @parameter("option")
    def rlog(self, val):
        return bool(val)

    @parameter("res")
    def dr_table(self, val):
        return float(val)

    @parameter("res")
    def hm_dlog10k(self, val):
        return float(val)

    @parameter("res")
    def hm_logk_min(self, val):
        return float(val)

    @parameter("res")
    def hm_logk_max(self, val):
        return float(val)

    @parameter("model")
    def sd_bias_model(self, val):
        if (
            not isinstance(val, str)
            and not issubclass_(val, bias.ScaleDepBias)
            and val is not None
        ):
            raise ValueError(
                "scale_dependenent_bias must be a subclass of bias.ScaleDepBias"
            )
        elif isinstance(val, str):
            model = get_model_(val, "halomod.bias")
            if not issubclass_(model, bias.ScaleDepBias):
                raise ValueError(
                    "scale_dependenent_bias must be a subclass of bias.ScaleDepBias"
                )
            return model
        else:
            return val

    @parameter("param")
    def sd_bias_params(self, val):
        return val

    @parameter("switch")
    def force_1halo_turnover(self, val):
        return bool(val)

    @parameter("model")
    def exclusion_model(self, val):
        """A string identifier for the type of halo exclusion used (or None)"""
        if val is None:
            val = "NoExclusion"

        if issubclass_(val, Exclusion):
            return val
        else:
            return get_model_(val, "halomod.halo_exclusion")

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
        return 10 ** np.arange(
            self._logr_table_min, self._logr_table_max, self.dr_table
        )

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
        """
        Scales at which correlation functions are computed [Mpc/h]
        """
        if hasattr(self.rmin, "__len__"):
            r = np.array(self.rmin)
        else:
            if self.rlog:
                r = np.exp(np.linspace(np.log(self.rmin), np.log(self.rmax), self.rnum))
            else:
                r = np.linspace(self.rmin, self.rmax, self.rnum)

        return r

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
            cm_relation=None, mdef=self.mdef, z=self.z, **self.halo_profile_params,
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
        """A class containing the elements necessary to calculate halo halo_profile quantities"""
        return self.halo_profile_model(
            cm_relation=self.halo_concentration,
            mdef=self.mdef,
            z=self.z,
            **self.halo_profile_params,
        )

    @cached_quantity
    def sd_bias(self):
        """A class containing relevant methods to calculate scale-dependent bias corrections"""
        if self.sd_bias_model is None:
            return None
        else:
            return self.sd_bias_model(
                self._corr_mm_base_fnc(self._r_table), **self.sd_bias_params
            )

    @cached_quantity
    def halo_bias(self):
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
            return self.nonlinear_power_fnc
        else:
            raise ValueError("hc_spectrum was specified incorrectly!")

    @cached_quantity
    def linear_power_fnc(self):
        """A callable returning the linear power as a function of k (in h/Mpc)."""
        return tools.ExtendedSpline(
            self.k,
            self.power,
            lower_func=lambda k: k ** self.n,
            upper_func="power_law",
            domain=(0, np.inf),
        )

    @cached_quantity
    def nonlinear_power_fnc(self):
        """A callable returning the nonlinear (halofit) power as a function of k (in h/Mpc)."""
        return tools.ExtendedSpline(
            self.k,
            self.nonlinear_power,
            lower_func=lambda k: k ** self.n,
            upper_func="power_law",
            domain=(0, np.inf),
        )

    @cached_quantity
    def corr_linear_mm_fnc(self):
        """The linear auto-correlation function of dark matter."""
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
        """The linear auto-correlation function of dark matter."""
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
        """The matter correlation function used throughout the calculations."""
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
                b = spl(mn)
            else:
                mask = np.logical_and(self.m >= 10 ** mn, self.m <= 10 ** mx)
                b = intg.simps(
                    self.bias[mask] * self.dndm[mask], self.m[mask]
                ) / intg.simps(self.dndm[mask], self.m[mask])
            return b

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
        return self.halo_profile.rho(
            self._r_table, self.m, norm="m", c=self.cmz_relation
        )

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
    @cached_quantity
    def power_1h_auto_matter_fnc(self):
        """The halo model-derived nonlinear 1-halo dark matter auto-power spectrum."""
        u = self.halo_profile_ukm
        integrand = self.dndm * self.m ** 3 * u ** 2

        p = (
            intg.trapz(integrand, dx=np.log(10) * self.dlog10m)
            / self.mean_density0 ** 2
        )

        return tools.ExtendedSpline(
            self.k, p, lower_func="power_law", upper_func="power_law"
        )

    @property
    def power_1h_auto_matter(self):
        """The halo model-derived nonlinear 1-halo dark matter auto-power spectrum."""
        return self.power_1h_auto_matter_fnc(self.k_hm)

    @cached_quantity
    def corr_1h_auto_matter_fnc(self):
        """The halo model-derived nonlinear 1-halo dark matter auto-correlation function."""
        if self.halo_profile.has_lam:
            lam = self.halo_profile_lam
            integrand = self.dndm * self.m ** 3 * lam

            table = (
                intg.trapz(integrand, dx=np.log(10) * self.dlog10m)
                / self.mean_density0 ** 2
                - 1
            )
        else:
            table = tools.hankel_transform(
                self.power_1h_auto_matter_fnc, self._r_table, "r"
            )

        return tools.ExtendedSpline(
            self._r_table,
            table,
            lower_func="power_law",
            upper_func=lambda x: np.zeros_like(x),
        )

    @property
    def corr_1h_auto_matter(self):
        """The halo model-derived nonlinear 1-halo dark matter auto-correlation function."""
        return self.corr_1h_auto_matter_fnc(self.r)

    @cached_quantity
    def power_2h_auto_matter_fnc(self):
        """The halo model-derived nonlinear 2-halo dark matter auto-power spectrum."""
        # TODO: check what to do here.
        # Basically, HMcode assumes that the large-scale power is equivalent
        # to the linear power, with no biasing. I think this *has* to be true
        # since the matter power is for *all* mass. But other codes (eg. chomp)
        # do the normal integral which includes biasing...
        return self._power_halo_centres_fnc

    @property
    def power_2h_auto_matter(self):
        """The halo model-derived nonlinear 2-halo dark matter auto-power spectrum."""
        # TODO: check what to do here.
        # Basically, HMcode assumes that the large-scale power is equivalent
        # to the linear power, with no biasing. I think this *has* to be true
        # since the matter power is for *all* mass. But other codes (eg. chomp)
        # do the normal integral which includes biasing...
        return self._power_halo_centres_fnc(self.k_hm)

    @cached_quantity
    def corr_2h_auto_matter_fnc(self):
        """The halo-model-derived nonlinear 2-halo dark matter auto-correlation function."""
        corr = tools.hankel_transform(self.power_2h_auto_matter_fnc, self._r_table, "r")
        return tools.ExtendedSpline(
            self._r_table,
            corr,
            lower_func="power_law"
            if (
                self.exclusion_model == NoExclusion
                and "filtered" not in self.hc_spectrum
            )
            else tools._zero,
            upper_func=lambda x: np.zeros_like(x),
        )

    @property
    def corr_2h_auto_matter(self):
        """The halo-model-derived nonlinear 2-halo dark matter auto-correlation function."""
        return self.corr_2h_auto_matter_fnc(self.r)

    @cached_quantity
    def corr_auto_matter_fnc(self):
        """The halo-model-derived nonlinear dark matter auto-correlation function."""
        return (
            lambda r: self.corr_1h_auto_matter_fnc(r)
            + self.corr_2h_auto_matter_fnc(r)
            + 1
        )

    @property
    def corr_auto_matter(self):
        """The halo-model-derived nonlinear dark matter auto-correlation function."""
        return self.corr_auto_matter_fnc(self.r)

    @cached_quantity
    def power_auto_matter_fnc(self):
        """The halo-model-derived nonlinear dark power auto-power spectrum."""
        return lambda k: self.power_1h_auto_matter_fnc(
            k
        ) + self.power_2h_auto_matter_fnc(k)

    @property
    def power_auto_matter(self):
        """The halo-model-derived nonlinear dark power auto-power spectrum."""
        return self.power_auto_matter_fnc(self.k_hm)

    @cached_quantity
    def bias_effective_matter(self):
        """The effective bias of matter in DM halos.

        This *should* be unity for models in which all dark matter is encapsulated in
        halos (though in practice it won't be for some halo-bias models). For models
        in which some dark matter exists in a "smooth" component outside halos (eg.
        the WDM model of Schneider et al. 2012) it also will not be unity.
        """
        # Integrand is just the density of halos at mass m by bias
        integrand = self.m * self.dndm * self.halo_bias
        b = intg.simps(integrand, self.m)
        return b / self.rho_gtm[0]


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
    force_1halo_turnover : bool, optional
        Whether to force the 1-halo term to turnover on large scales. THis induces a
        heuristic modification which ensures that the 1-halo term does not grow
        larger than the two-halo term on very large scales.

    Other Parameters
    ----------------
    All other parameters are passed to :class:`~DMHaloModel`.
    """

    def __init__(
        self,
        hod_model="Zehavi05",
        hod_params={},
        tracer_profile_model=None,
        tracer_profile_params=None,
        tracer_concentration_model=None,
        tracer_concentration_params=None,
        tracer_density=None,
        force_1halo_turnover=True,
        **halomodel_kwargs,
    ):
        super().__init__(**halomodel_kwargs)

        # Initially save parameters to the class.
        self.hod_params = hod_params
        self.hod_model = hod_model
        self.tracer_profile_model, self.tracer_profile_params = (
            tracer_profile_model,
            tracer_profile_params,
        )
        self.tracer_concentration_model, self.tracer_concentration_params = (
            tracer_concentration_model,
            tracer_concentration_params,
        )

        self.force_1halo_turnover = force_1halo_turnover
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
        elif "hod_params" in kwargs:
            if "M_min" in kwargs["hod_params"]:
                self.tracer_density = None

        super().update(**kwargs)

        if self.tracer_density is not None:
            mmin = self._find_m_min(self.tracer_density)
            self.hod_params = {"M_min": mmin}

    # ===============================================================================
    # Parameters
    # ===============================================================================
    @parameter("param")
    def tracer_density(self, val):
        """Mean density of the tracer, ONLY if passed directly."""
        return val

    @parameter("param")
    def hod_params(self, val):
        """Dictionary of parameters for the HOD model."""
        return val

    @parameter("model")
    def hod_model(self, val):
        """:class:`~hod.HOD` class"""
        if not (isinstance(val, str) or issubclass_(val, hod.HOD)):
            raise ValueError("hod_model must be a subclass of hod.HOD")
        elif isinstance(val, str):
            return get_model_(val, "halomod.hod")
        else:
            return val

    @parameter("param")
    def tracer_profile_params(self, val):
        """Dictionary of parameters for the Profile model"""
        if val is None:
            return self.halo_profile_params
        else:
            return val

    @parameter("model")
    def tracer_profile_model(self, val):
        """The halo density halo_profile model"""
        if val is None:
            return val
        if not (isinstance(val, str) or issubclass_(val, profiles.Profile)):
            raise ValueError(
                "halo_profile_model must be a subclass of profiles.Profile"
            )
        if isinstance(val, str):
            return get_model_(val, "halomod.profiles")
        else:
            return val

    @parameter("model")
    def tracer_concentration_model(self, val):
        """A halo_concentration-mass relation"""
        if val is None:
            return val
        if not (isinstance(val, str) or issubclass_(val, CMRelation)):
            raise ValueError(
                "halo_concentration_model must be a subclass of concentration.CMRelation"
            )
        elif isinstance(val, str):
            return get_model_(val, "halomod.concentration")
        else:
            return val

    @parameter("param")
    def tracer_concentration_params(self, val):
        if val is None:
            return self.halo_concentration_params
        else:
            return val

    @parameter("switch")
    def force_1halo_turnover(self, val):
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
                "Please set Mmin lower."
            )

        return self.m >= 10 ** self.hod.mmin

    @cached_quantity
    def tracer_concentration(self):
        """Concentration-mass relation model instance."""
        if self.tracer_concentration_model is None:
            return self.halo_concentration

        this_filter = copy(self.filter)
        this_filter.power = self._power0

        if self.tracer_profile_model is None:
            this_profile = self.halo_profile_model(
                cm_relation=None, mdef=self.mdef, z=self.z, **self.halo_profile_params,
            )
        else:
            this_profile = self.tracer_profile_model(
                cm_relation=None,
                mdef=self.mdef,
                z=self.z,
                **self.tracer_profile_params,
            )

        return self.tracer_concentration_model(
            cosmo=Cosmology(self.cosmo),
            filter0=this_filter,
            growth=self.growth,
            delta_c=self.delta_c,
            profile=this_profile,
            mdef=self.mdef,
            **self.tracer_concentration_params,
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

        return self.tracer_profile_model(
            cm_relation=self.tracer_concentration,
            mdef=self.mdef,
            z=self.z,
            **self.tracer_profile_params,
        )

    @cached_quantity
    def hod(self):
        """A class representing the HOD"""
        return self.hod_model(**self.hod_params)

    # ===========================================================================
    # Basic HOD Quantities
    # ===========================================================================
    @cached_quantity
    def total_occupation(self):
        """The mean occupation of the tracer as a function of halo mass"""
        return self.hod.total_occupation(self.m)

    @cached_quantity
    def satellite_occupation(self):
        return self.hod.satellite_occupation(self.m)

    @cached_quantity
    def central_occupation(self):
        return self.hod.central_occupation(self.m)

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
        integrand = self.dndm[self._tm] * self.total_occupation[self._tm]
        return intg.simps(integrand, self.m[self._tm])

    @cached_quantity
    def mean_tracer_den_unit(self):
        """
        The mean density of the tracer, in the units defined in the HOD.
        """
        return self.mean_tracer_den * self.hod.unit_conversion(self.cosmo, self.z)

    @cached_quantity
    def bias_effective_tracer(self):
        """
        The tracer occupation-weighted halo bias factor (Tinker 2005)
        """
        # Integrand is just the density of galaxies at mass m by bias
        integrand = (
            self.m[self._tm]
            * self.dndm[self._tm]
            * self.total_occupation[self._tm]
            * self.halo_bias[self._tm]
        )
        b = intg.trapz(integrand, dx=np.log(self.m[1] / self.m[0]))
        return b / self.mean_tracer_den

    @cached_quantity
    def mass_effective(self):
        """
        Average host-halo mass (in log10 units)
        """
        # Integrand is just the density of galaxies at mass m by m
        integrand = (
            self.m[self._tm] ** 2
            * self.dndm[self._tm]
            * self.total_occupation[self._tm]
        )

        m = intg.trapz(integrand, dx=np.log(self.m[1] / self.m[0]))
        return np.log10((m / self.mean_tracer_den))

    @cached_quantity
    def satellite_fraction(self):
        """The total fraction of tracers that are satellites.

        Note: this may not exist for every kind of tracer.
        """
        if hasattr(self.hod, "satellite_occupation"):
            # Integrand is just the density of satellite galaxies at mass m
            integrand = (
                self.m[self._tm]
                * self.dndm[self._tm]
                * self.hod.satellite_occupation(self.m[self._tm])
            )
            s = intg.trapz(integrand, dx=np.log(self.m[1] / self.m[0]))
            return s / self.mean_tracer_den
        else:
            raise AttributeError("This HOD has no concept of a satellite")

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
        return self.tracer_profile.rho(
            self._r_table, self.m, norm="m", c=self.tracer_cmz_relation
        )

    @cached_quantity
    def tracer_profile_lam(self):
        """The mass-normalised profile self-convolution of the tracer, shape (len(r), len(m))."""
        if self.tracer_profile.has_lam:
            return self.tracer_profile.lam(
                self._r_table, self.m, c=self.tracer_cmz_relation
            )
        else:
            return None

    # ===========================================================================
    # 2-point tracer-tracer (HOD) statistics
    # ===========================================================================
    @cached_quantity
    def power_1h_ss_auto_tracer_fnc(self):
        """The satellite-satellite part of the 1-halo term of the tracer auto-power spectrum.

        Note: May not exist for every kind of tracer.
        """
        if not hasattr(self.hod, "ss_pairs"):
            raise AttributeError("The HOD being used has no satellite occupation")

        u = self.tracer_profile_ukm[:, self._tm]
        integ = (
            u ** 2
            * self.dndm[self._tm]
            * self.m[self._tm]
            * self.hod.ss_pairs(self.m[self._tm])
        )

        if self.force_1halo_turnover:
            r = np.pi / self.k / 10  # The 10 is a complete heuristic hack.
            mmin = (
                4 * np.pi * r ** 3 * self.mean_density0 * self.halo_overdensity_mean / 3
            )
            mask = np.outer(self.m[self._tm], np.ones_like(self.k)) < mmin
            integ[mask.T] = 0

        p = intg.trapz(integ, dx=self.dlog10m * np.log(10))

        p /= self.mean_tracer_den ** 2

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
        The satellite-satellite part of the 1-halo term of the tracer auto-correlation function.

        Note: May not exist for every kind of tracer.
        """
        if not hasattr(self.hod, "ss_pairs"):
            raise AttributeError("The HOD being used has no satellite occupation")

        if self.tracer_profile.has_lam:
            lam = self.tracer_profile_lam
            integ = (
                self.m[self._tm]
                * self.dndm[self._tm]
                * self.hod.ss_pairs(self.m[self._tm])
                * lam[:, self._tm]
            )

            c = intg.trapz(integ, dx=self.dlog10m * np.log(10))
            c = c / self.mean_tracer_den ** 2 - 1

        else:
            c = tools.hankel_transform(
                self.power_1h_ss_auto_tracer_fnc, self._r_table, "r"
            )
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
        """The cen-sat part of the 1-halo term of the tracer auto-power spectrum.

        Note: May not exist for every kind of tracer.
        """
        if not hasattr(self.hod, "cs_pairs"):
            raise AttributeError("The HOD being used has no satellite occupation")

        u = self.tracer_profile_ukm[:, self._tm]
        integ = (
            self.dndm[self._tm]
            * 2
            * self.hod.cs_pairs(self.m[self._tm])
            * u
            * self.m[self._tm]
        )

        if self.force_1halo_turnover:
            r = np.pi / self.k / 10  # The 10 is a complete heuristic hack.
            mmin = (
                4 * np.pi * r ** 3 * self.mean_density0 * self.halo_overdensity_mean / 3
            )
            mask = np.outer(self.m[self._tm], np.ones_like(self.k)) < mmin
            integ[mask.T] = 0

        c = intg.trapz(integ, dx=self.dlog10m * np.log(10))
        c /= self.mean_tracer_den ** 2

        return tools.ExtendedSpline(
            self.k,
            c,
            lower_func=tools._zero if self.force_1halo_turnover else "boundary",
            upper_func="power_law" if np.all(c[-10:] > 0) else tools._zero,
        )

    @property
    def power_1h_cs_auto_tracer(self):
        """The cen-sat part of the 1-halo term of the tracer auto-power spectrum.

        Note: May not exist for every kind of tracer.
        """
        return self.power_1h_cs_auto_tracer_fnc(self.k_hm)

    @cached_quantity
    def corr_1h_cs_auto_tracer_fnc(self):
        """The cen-sat part of the 1-halo term of the tracer auto-correlation function.

        Note: May not exist for every kind of tracer.
        """
        if not hasattr(self.hod, "cs_pairs"):
            raise AttributeError("The HOD being used has no satellite occupation")

        rho = self.tracer_profile_rho[:, self._tm]
        integ = (
            self.dndm[self._tm]
            * 2
            * self.hod.cs_pairs(self.m)[self._tm]
            * rho
            * self.m[self._tm]
        )
        c = intg.trapz(integ, dx=self.dlog10m * np.log(10))

        c = c / self.mean_tracer_den ** 2 - 1

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
        """
        Total 1-halo galaxy power.
        """
        try:
            return lambda k: self.power_1h_cs_auto_tracer_fnc(
                k
            ) + self.power_1h_ss_auto_tracer_fnc(k)
        except AttributeError:
            u = self.tracer_profile_ukm[:, self._tm]
            integ = u ** 2 * self.dndm[self._tm] * self.total_occupation[self._tm] ** 2

            if self.force_1halo_turnover:
                r = np.pi / self.k / 10  # The 10 is a complete heuristic hack.
                mmin = (
                    4
                    * np.pi
                    * r ** 3
                    * self.mean_density0
                    * self.halo_overdensity_mean
                    / 3
                )
                mask = np.outer(self.m[self._tm], np.ones_like(self.k)) < mmin
                integ[mask.T] = 0

            p = intg.simps(integ, self.m[self._tm]) / self.mean_tracer_den ** 2

            return tools.ExtendedSpline(
                self.k,
                p,
                lower_func=tools._zero if self.force_1halo_turnover else "boundary",
                upper_func=tools._zero,
            )

    @property
    def power_1h_auto_tracer(self):
        """Total 1-halo galaxy power."""
        return self.power_1h_auto_tracer_fnc(self.k_hm)

    @cached_quantity
    def corr_1h_auto_tracer_fnc(self):
        """The 1-halo term of the galaxy correlations."""
        if self.tracer_profile.has_lam:
            lam = self.tracer_profile_lam[:, self._tm]
            if hasattr(self.hod, "ss_pairs"):
                rho = self.tracer_profile_rho[:, self._tm]
                integ = (
                    self.m[self._tm]
                    * self.dndm[self._tm]
                    * (
                        self.hod.ss_pairs(self.m[self._tm]) * lam
                        + 2 * self.hod.cs_pairs(self.m[self._tm]) * rho
                    )
                )
                if self.hod._central:
                    integ *= self.central_occupation[self._tm]

            else:
                integ = (
                    self.m[self._tm]
                    * self.dndm[self._tm]
                    * (self.total_occupation[self._tm] ** 2 * lam)
                )
            c = intg.trapz(integ, dx=self.dlog10m * np.log(10))

            c /= self.mean_tracer_den ** 2

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
        """The 1-halo term of the galaxy correlations."""
        return self.corr_1h_auto_tracer_fnc(self.r)

    @cached_quantity
    def _power_2h_auto_tracer_primitive(self):
        """The 2-halo term of the tracer auto-power spectrum.

        This is 'primitive' because it can be 2D, i.e. it can have an r-based scale
        dependence based either on scale dependent bias or halo exclusion.
        """
        # It's possible that a better route for both scale-dep bias and halo-exclusion
        # is to use the scales r=2pi/k. But then you'd get correlation functions that
        # weren't necessarily the FT of the power...

        u = self.tracer_profile_ukm[:, self._tm]
        if self.sd_bias_model is not None:
            bias = np.outer(self.sd_bias_correction, self.halo_bias)[:, self._tm]
        else:
            bias = self.halo_bias[self._tm]

        inst = self.exclusion_model(
            m=self.m[self._tm],
            density=self.total_occupation[self._tm] * self.dndm[self._tm],
            Ifunc=self.total_occupation[self._tm]
            * self.dndm[self._tm]
            * u
            / self.mean_tracer_den,
            bias=bias,
            r=self._r_table,
            delta_halo=self.halo_overdensity_mean,
            mean_density=self.mean_density0,
            **self.exclusion_params,
        )

        if hasattr(inst, "density_mod"):
            self.__density_mod = inst.density_mod
        else:
            self.__density_mod = np.ones_like(self._r_table) * self.mean_tracer_den

        intg = inst.integrate()

        phh = self._power_halo_centres_fnc(self.k)

        if intg.ndim == 2:
            p = []
            for i, x in enumerate(intg):

                p.append(
                    tools.ExtendedSpline(
                        self.k,
                        x * phh,
                        lower_func=self.linear_power_fnc,
                        match_lower=True,
                        upper_func="power_law"
                        if (
                            self.exclusion_model == NoExclusion
                            and "filtered" not in self.hc_spectrum
                        )
                        else tools._zero,
                    )
                )
        else:
            p = tools.ExtendedSpline(
                self.k,
                intg * phh,
                lower_func=self.linear_power_fnc,
                match_lower=True,
                upper_func="power_law"
                if (
                    self.exclusion_model == NoExclusion
                    and "filtered" not in self.hc_spectrum
                )
                else tools._zero,
            )

        return p

    @property
    def power_2h_auto_tracer(self):
        """The 2-halo term of the tracer auto-power spectrum."""
        # If there's nothing modifying the scale-dependence, just return the original power.
        if self.exclusion_model is NoExclusion and self.sd_bias_model is None:
            return self._power_2h_auto_tracer_primitive(self.k_hm)

        # Otherwise, first calculate the correlation function.
        out = tools.hankel_transform(
            self.corr_2h_auto_tracer_fnc, self.k_hm, "k", h=0.001
        )

        # Everything below about k=1e-2 is essentially just the linear power biased,
        # and the hankel transform stops working at some small k.
        if np.any(self.k_hm < 1e-2):
            warnings.warn(
                "power_2h_auto_tracer for k < 1e-2 is not computed directly, but "
                "is rather just the linear power * effective bias."
            )
            out[self.k_hm < 1e-2] = (
                self.power[self.k_hm < 1e-2] * self.bias_effective_tracer
            )

        return out

    @cached_quantity
    def corr_2h_auto_tracer_fnc(self):
        """The 2-halo term of the tracer auto-correlation."""
        # Need to set h smaller here because this might need to be transformed back
        # to power.
        corr = tools.hankel_transform(
            self._power_2h_auto_tracer_primitive, self._r_table, "r", h=1e-4
        )

        # modify by the new density. This step is *extremely* sensitive to the exact
        # value of __density_mod at large
        # scales, where the ratio *should* be exactly 1.
        if self._r_table[-1] > 2 * self.halo_profile.halo_mass_to_radius(self.m[-1]):
            try:
                self.__density_mod *= self.mean_tracer_den / self.__density_mod[-1]
            except TypeError:
                pass

        corr = (self.__density_mod / self.mean_tracer_den) ** 2 * (1 + corr) - 1

        return tools.ExtendedSpline(
            self._r_table, corr, lower_func="power_law", upper_func=tools._zero
        )

    @property
    def corr_2h_auto_tracer(self):
        """The 2-halo term of the tracer auto-correlation."""
        return self.corr_2h_auto_tracer_fnc(self.r)

    @property
    def power_auto_tracer(self):
        """Auto-power spectrum of the tracer"""
        return self.power_1h_auto_tracer_fnc(self.k_hm) + self.power_2h_auto_tracer

    @cached_quantity
    def corr_auto_tracer_fnc(self):
        """The tracer auto correlation function"""
        return lambda r: self.corr_1h_auto_tracer_fnc(r) + self.corr_2h_auto_tracer_fnc(
            r
        )

    @property
    def corr_auto_tracer(self):
        """The tracer auto correlation function"""
        return self.corr_auto_tracer_fnc(self.r)

    # ===========================================================================
    # Cross-correlations
    # ===========================================================================
    @cached_quantity
    def power_1h_cross_tracer_matter_fnc(self):
        """
        Total 1-halo cross-power
        """
        ut = self.tracer_profile_ukm[:, self._tm]
        uh = self.halo_profile_ukm[:, self._tm]

        m = self.m[self._tm]

        integ = self.dndm[self._tm] * (
            uh * ut * m * self.total_occupation[self._tm]
            + uh * self.hod.satellite_occupation(m)
        )
        p = intg.simps(integ, m)

        p /= self.mean_tracer_den * self.mean_density
        return tools.ExtendedSpline(
            self.k, p, lower_func="power_law", upper_func="power_law"
        )

    @property
    def power_1h_cross_tracer_matter(self):
        """
        Total 1-halo cross-power
        """
        return self.power_1h_cross_tracer_matter_fnc(self.k_hm)

    @cached_quantity
    def corr_1h_cross_tracer_matter_fnc(self):
        """The 1-halo term of the cross correlation"""
        corr = tools.hankel_transform(
            self.power_1h_cross_tracer_matter_fnc, self._r_table, "r"
        )
        return tools.ExtendedSpline(
            self._r_table, corr, lower_func="power_law", upper_func=tools._zero
        )

    @property
    def corr_1h_cross_tracer_matter(self):
        """The 1-halo term of the cross correlation"""
        return self.corr_1h_cross_tracer_matter_fnc(self.r)

    @cached_quantity
    def power_2h_cross_tracer_matter_fnc(self):
        """The 2-halo term of the cross-power spectrum."""
        ut = self.tracer_profile_ukm[:, self._tm]
        um = self.halo_profile_ukm

        # if self.sd_bias_model is not None:
        #     bias = np.outer(self.sd_bias.bias_scale(), self.bias)[:, self._tm]
        # else:
        bias = self.halo_bias

        # Do this the simple way for now
        bt = intg.simps(
            self.dndm[self._tm] * bias[self._tm] * self.total_occupation[self._tm] * ut,
            self.m[self._tm],
        )
        bm = intg.simps(self.dndm * bias * self.m * um, self.m)

        power = (
            bt
            * bm
            * self._power_halo_centres_fnc(self.k)
            / (self.mean_tracer_den * self.mean_density)
        )

        return tools.ExtendedSpline(
            self.k,
            power,
            lower_func="power_law",
            upper_func="power_law"
            if "filtered" not in self.hc_spectrum
            else tools._zero,
        )

        # inst = self.exclusion_model(m=self.m[self._tm],
        #                             density=self.total_occupation[self._tm] * self.dndm[self._tm],
        #                             I=self.total_occupation[self._tm] * self.dndm[
        #                                 self._tm] * u / self.mean_tracer_den,
        #                             bias=bias, r=self.r, delta_halo=self.delta_halo,
        #                             mean_density=self.mean_density0,
        #                             **self.exclusion_params)

        # if hasattr(inst, "density_mod"):
        #     self.__density_mod = inst.density_mod
        # else:
        #     self.__density_mod = np.ones_like(self.r) * self.mean_tracer_den

        # return inst.integrate() * self._power_halo_centres

    @property
    def power_2h_cross_tracer_matter(self):
        """The 2-halo term of the cross-power spectrum."""
        return self.power_2h_cross_tracer_matter_fnc(self.k_hm)

    @cached_quantity
    def corr_2h_cross_tracer_matter_fnc(self):
        """The 2-halo term of the cross-correlation"""
        corr = tools.hankel_transform(
            self.power_2h_cross_tracer_matter_fnc, self._r_table, "r"
        )
        return tools.ExtendedSpline(
            self._r_table, corr, lower_func="power_law", upper_func=tools._zero
        )

    @property
    def corr_2h_cross_tracer_matter(self):
        """The 2-halo term of the cross-correlation"""
        return self.corr_2h_cross_tracer_matter_fnc(self.r)

    @cached_quantity
    def power_cross_tracer_matter_fnc(self):
        """Cross-power spectrum of tracer and matter"""
        return lambda k: self.power_1h_cross_tracer_matter_fnc(
            k
        ) + self.power_2h_cross_tracer_matter_fnc(k)

    @property
    def power_cross_tracer_matter(self):
        """Cross-power spectrum of tracer and matter"""
        return self.power_cross_tracer_matter_fnc(self.k_hm)

    @cached_quantity
    def corr_cross_tracer_matter_fnc(self):
        """Cross-correlation of tracer with matter"""
        return (
            lambda r: self.corr_1h_cross_tracer_matter_fnc(r)
            + self.corr_2h_cross_tracer_matter_fnc(r)
            + 1
        )

    @property
    def corr_cross_tracer_matter(self):
        """Cross-correlation of tracer with matter"""
        return self.corr_cross_tracer_matter_fnc(self.r)

    # ===========================================================================
    # Other utilities
    # ===========================================================================
    def _find_m_min(self, ng):
        """
        Calculate the minimum mass of a halo to contain a (central) galaxy
        based on a known mean galaxy density
        """

        self.power  # This just makes sure the power is gotten and copied
        c = deepcopy(self)
        c.update(hod_params={"M_min": self.Mmin}, dlog10m=0.01)

        integrand = c.m[c._tm] * c.dndm[c._tm] * c.total_occupation[c._tm]

        density_message = (
            f"Maximum mean galaxy density exceeded. User input required density of {ng}, "
            "but maximum density (with HOD M_min == DM Mmin) is {}. "
            f"Consider decreasing Mmin,or checking tracer_density."
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
            integral = intg.simps(integrand, dx=np.log(c.m[1] / c.m[0]))
            if integral < ng:
                raise NGException(density_message.format(integral))

            def model(mmin):
                c.update(hod_params={"M_min": mmin})
                integrand = c.m[c._tm] * c.dndm[c._tm] * c.total_occupation[c._tm]
                integral = intg.simps(integrand, dx=np.log(c.m[1] / c.m[0]))
                return abs(integral - ng)

            res = minimize(
                model, 12.0, tol=1e-3, method="Nelder-Mead", options={"maxiter": 200}
            )
            mmin = res.x[0]

        return mmin

    # =============================
    # For Compatibility
    # =============================
    @property
    def corr_gg_1h(self):
        warnings.warn("This method is deprecated in favour of corr_1h_auto_tracer")
        return self.corr_1h_auto_tracer

    @property
    def corr_gg_2h(self):
        warnings.warn("This method is deprecated in favour of corr_2h_auto_tracer")
        return self.corr_2h_auto_tracer

    @property
    def corr_gg(self):
        warnings.warn("This method is deprecated in favour of corr_auto_tracer")
        return self.corr_auto_tracer

    @property
    def power_gg_1h(self):
        warnings.warn("This method is deprecated in favour of power_1h_auto_tracer")
        return self.power_1h_auto_tracer

    @property
    def power_gg_2h(self):
        warnings.warn("This method is deprecated in favour of power_2h_auto_tracer")
        return self.power_2h_auto_tracer

    @property
    def power_gg(self):
        warnings.warn("This method is deprecated in favour of power_auto_tracer")
        return self.power_auto_tracer

    @property
    def corr_mm_1h(self):
        warnings.warn("This method is deprecated in favour of corr_1h_auto_matter")
        return self._corr_1h_auto_matter_table

    @property
    def corr_mm_2h(self):
        warnings.warn("This method is deprecated in favour of corr_2h_auto_matter")
        return self.corr_2h_auto_matter

    @property
    def corr_mm(self):
        warnings.warn("This method is deprecated in favour of corr_auto_matter")
        return self.corr_auto_matter

    @property
    def power_mm_1h(self):
        warnings.warn("This method is deprecated in favour of power_1h_auto_matter")
        return self.power_1h_auto_matter

    @property
    def power_mm_2h(self):
        warnings.warn("This method is deprecated in favour of power_2h_auto_matter")
        return self.power_2h_auto_matter

    @property
    def power_mm(self):
        warnings.warn("This method is deprecated in favour of power_auto_matter")
        return self.power_auto_matter


# For compatibility
HaloModel = TracerHaloModel


class NGException(Exception):
    pass
