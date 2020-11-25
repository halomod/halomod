"""
Module for defining HOD classes.

The HOD class exposes methods that deal directly with occupation statistics and don't interact with
the broader halo model. These include things like the average satellite/central occupation, total
occupation, and "pair counts".

The HOD concept is here meant to be as general as possible. While traditionally the HOD has been
thought of as a number count occupation, the base class here is just as amenable to "occupations"
that could be defined over the real numbers -- i.e. continuous occupations. This could be achieved
via each "discrete" galaxy being marked by some real quantity (eg. each galaxy is on average a
certain brightness, or contains a certain amount of gas), or it could be achieved without assuming
any kind of discrete tracer, and just assuming a matching of some real field to the underlying halo
mass. Thus  *all* kinds of occupations can be dealt with in these classes.

For the sake of consistency of implementation, all classes contain the notion that there may be a
"satellite" component of the occupation, and a "central" component. This is to increase fidelity in
cases where it is known that a discrete central object will necessarily be in the sample before any
other object, because it is inherently "brighter" (for whatever selection the sample uses). It is
not necessary to assume some distinct central component, so for models in which this does not make
sense, it is safe to set the central component to zero.

The most subtle/important thing to note about these classes are the assumptions surrounding the
satellite/central decomposition. So here are the assumptions:

1. The average satellite occupancy is taken to be the average over *all* haloes, with and without
   centrals. This has subtle implications for how to mock up the galaxy population, because if one
   requires a central before placing a satellite, then the avg. number of satellites placed into
   *available* haloes is increased if the central occupation is less than 1.

2. We provide the option to enforce a "central condition", that is, the requirement that a central
   be found in a halo before any satellites are observed. To enforce this, set ``central=True`` in
   the constructor of any HOD. This has some ramifications:

3. If the central condition is enforced, then for all HOD classes (except see point 5), the mean
   satellite occupancy is modified. If the defined occupancy is Ns', then the returned occupancy is
   Ns = Nc*Ns'. This merely ensures that Ns=0 when Nc=0. The user should note that this will change
   the interpretation of parameters in the Ns model, unless Nc is a simple step function.

4. The pair-wise counts involve a term <Nc*Ns>. When the central condition is enforced, this reduces
   trivially to <Ns>. However, if the central condition is not enforced we *assume* that the
   variates Nc and Ns are uncorrelated, and use <Nc*Ns> = <Nc><Ns>.

5. A HOD class that is defined with the central condition intrinsically satisfied, the class variable
   ``central_condition_inherent`` can be set to True in the class definition, which will avoid the
   extra modification. Do note that just because the class is specified such that the central
   condition can be satisfied (i.e. <Ns> is 0 when <Nc> is zero), and thus the
   ``central_condition_inherent`` is True, does not mean that it is entirely enforced.
   The pairwise counts still depend on whether the user assumes that the central condition is
   enforced or not, which must be set at instantiation.

6. By default, the central condition is *not* enforced.
"""


import numpy as np
import scipy.special as sp
from hmf import Component
from abc import ABCMeta, abstractmethod
from astropy.cosmology import Planck15
from .concentration import CMRelation
from .profiles import Profile
from hmf.halos.mass_definitions import MassDefinition, SOMean
from hmf._internals import pluggable
import scipy.constants as const
import astropy.constants as astroconst


@pluggable
class HOD(Component, metaclass=ABCMeta):
    """
    Halo Occupation Distribution model base class.

    This class should not be called directly. The user
    should call a derived class.

    As with all :class:`hmf._framework.Model` classes,
    each class should specify its parameters in a _defaults dictionary at
    class-level.

    The exception to this is the M_min parameter, which is defined for every
    model (it may still be defined to modify the default). This parameter acts
    as the one that may be set via the mean density given all the other
    parameters. If the model has a sharp cutoff at low mass, corresponding to
    M_min, the extra parameter sharp_cut may be set to True, allowing for simpler
    setting of M_min via this route.

    See the derived classes in this module for examples of how to define derived
    classes of :class:`HOD`.
    """

    _defaults = {"M_min": 11.0}
    sharp_cut = False
    central_condition_inherent = False

    def __init__(
        self,
        central: bool = False,
        cosmo=Planck15,
        cm_relation: [None, CMRelation] = None,
        profile: [None, Profile] = None,
        mdef: [None, MassDefinition] = SOMean(),
        **model_parameters
    ):
        self._central = central
        self.cosmo = cosmo
        self.cm_relation = cm_relation
        self.profile = profile
        self.mdef = mdef

        super(HOD, self).__init__(**model_parameters)

    @abstractmethod
    def nc(self, m):
        """Defines the average number of centrals at mass m.

        Useful for populating catalogues."""
        pass

    @abstractmethod
    def ns(self, m):
        """Defines the average number of satellites at mass m.

        Useful for populating catalogues."""
        pass

    @abstractmethod
    def _central_occupation(self, m):
        """The central occupation function of the tracer."""
        pass

    @abstractmethod
    def _satellite_occupation(self, m):
        """The satellite occupation function of the tracer."""
        pass

    @abstractmethod
    def ss_pairs(self, m):
        """The average amount of the tracer coupled with itself in haloes of mass m, <T_s T_s>."""
        pass

    @abstractmethod
    def cs_pairs(self, m):
        """The average amount of the tracer coupled with itself in haloes of mass m, <T_s T_c>."""
        pass

    @abstractmethod
    def sigma_satellite(self, m):
        """The standard deviation of the satellite tracer amount in haloes of mass m."""
        pass

    @abstractmethod
    def sigma_central(self, m):
        """The standard deviation of the central tracer amount in haloes of mass m."""
        pass

    def central_occupation(self, m):
        """The occupation function of the central component."""
        return self._central_occupation(m)

    def satellite_occupation(self, m):
        """The occupation function of the satellite (or profile-dependent) component."""
        if self._central and not self.central_condition_inherent:
            return self.nc(m) * self._satellite_occupation(m)
        else:
            return self._satellite_occupation(m)

    def total_occupation(self, m):
        """The total (average) occupation of the halo."""
        return self.central_occupation(m) + self.satellite_occupation(m)

    def total_pair_function(self, m):
        """The total weight of the occupation paired with itself."""
        return self.ss_pairs(m) + self.cs_pairs(m)

    def unit_conversion(self, cosmo, z):
        """A factor to convert the total occupation to a desired unit."""
        return 1.0

    @property
    def mmin(self):
        """Defines a reasonable minimum mass to set for this HOD to converge when integrated."""
        return self.params["M_min"]


class HODNoCentral(HOD, abstract=True):
    """Base class for all HODs which have no concept of a central/satellite split."""

    def __init__(self, **model_parameters):
        super(HODNoCentral, self).__init__(**model_parameters)
        self._central = False

    def nc(self, m):
        """Density of Central Tracer"""
        return 0

    def cs_pairs(self, m):
        """The average amount of the tracer coupled with itself in haloes of mass m, <T_s T_c>."""
        return 0

    def _central_occupation(self, m):
        """The occupation function of the central component."""
        return 0

    def sigma_central(self, m):
        """The standard deviation of the central tracer amount in haloes of mass m."""
        return 0


class HODBulk(HODNoCentral, abstract=True):
    """Base class for HODs with no discrete tracers, just an assignment of tracer to the halo."""

    def ns(self, m):
        """Density of Satellite Tracer"""
        return 0

    def ss_pairs(self, m):
        """The average amount of the tracer coupled with itself in haloes of mass m, <T_s T_s>."""
        return self.satellite_occupation(m) ** 2


class HODPoisson(HOD, abstract=True):
    """
    Base class for discrete HOD's with poisson-distributed satellite population.

    Also assumes that the amount of the tracer is statistically independent of the number
    counts, but its average is directly proportional to it.

    This accounts for all Poisson-distributed number-count HOD's (which is all traditional HODs).
    """

    def nc(self, m):
        """Amplitude of central tracer per tracer source"""
        return self.central_occupation(m) / self._tracer_per_central(m)

    def ns(self, m):
        """Amplitude of satellite tracer per tracer source"""
        return self.satellite_occupation(m) / self._tracer_per_satellite(m)

    def _tracer_per_central(self, m):
        """Number of tracer per central tracer source"""
        return 1

    def _tracer_per_satellite(self, m):
        """Number of tracer per satellite tracer source"""
        return self._tracer_per_central(m)

    def ss_pairs(self, m):
        """The average amount of the tracer coupled with itself in haloes of mass m, <T_s T_s>."""
        return self.satellite_occupation(m) ** 2

    def cs_pairs(self, m):
        """The average amount of the tracer coupled with itself in haloes of mass m, <T_c T_s>."""
        if self._central:
            return self.satellite_occupation(m) * self._tracer_per_central(m)
        else:
            return self.central_occupation(m) * self.satellite_occupation(m)

    def sigma_central(self, m):
        """The standard deviation of the central tracer amount in haloes of mass m."""
        co = self.central_occupation(m)
        return np.sqrt(co * (1 - co))

    def sigma_satellite(self, m):
        """The standard deviation of the satellite tracer amount in haloes of mass m."""
        return np.sqrt(self.satellite_occupation(m))


class Zehavi05(HODPoisson):
    r"""
    Three-parameter model of Zehavi (2005) [1]_.

    Parameters
    ----------
    M_min : float, default = 11.6222
        Minimum mass of halo that supports a central galaxy
    M_1 : float, default = 12.851
        Mass of a halo which on average contains 1 satellite
    alpha : float, default = 1.049
        Index of power law for satellite galaxies

    References
    ----------
    .. [1] Zehavi, I. et al., "The Luminosity and Color Dependence of the Galaxy Correlation
           Function ", https://ui.adsabs.harvard.edu/abs/2005ApJ...630....1Z.

    """

    _defaults = {"M_min": 11.6222, "M_1": 12.851, "alpha": 1.049}
    sharp_cut = True

    def _central_occupation(self, m):
        """
        Amplitude of central tracer at mass M
        """
        n_c = np.ones_like(m)
        n_c[m < 10 ** self.params["M_min"]] = 0

        return n_c

    def _satellite_occupation(self, m):
        """
        Amplitude of satellite tracer at mass M
        """
        return (m / 10 ** self.params["M_1"]) ** self.params["alpha"]


class Zheng05(HODPoisson):
    """
    Five-parameter model of Zheng (2005)

    Parameters
    ----------
    M_min : float, default = 11.6222
        Minimum mass of halo that supports a central galaxy
    M_1 : float, default = 12.851
        Mass of a halo which on average contains 1 satellite
    alpha : float, default = 1.049
        Index of power law for satellite galaxies
    sig_logm : float, default = 0.26
        Width of smoothed cutoff
    M_0 : float, default = 11.5047
        Minimum mass of halo containing satellites

    References
    ----------
    .. [1] Zheng, Z. et al., "Theoretical Models of the Halo Occupation Distribution:
           Separating Central and Satellite Galaxies ",
           https://ui.adsabs.harvard.edu/abs/2005ApJ...633..791Z.

    """

    _defaults = {
        "M_min": 11.6222,
        "M_1": 12.851,
        "alpha": 1.049,
        "M_0": 11.5047,
        "sig_logm": 0.26,
    }

    def _central_occupation(self, m):
        """
        Amplitude of central tracer at mass M
        """
        return 0.5 * (
            1 + sp.erf((np.log10(m) - self.params["M_min"]) / self.params["sig_logm"])
        )

    def _satellite_occupation(self, m):
        """
        Amplitude of satellite tracer at mass M
        """
        ns = np.zeros_like(m)
        ns[m > 10 ** self.params["M_0"]] = (
            (m[m > 10 ** self.params["M_0"]] - 10 ** self.params["M_0"])
            / 10 ** self.params["M_1"]
        ) ** self.params["alpha"]
        return ns

    @property
    def mmin(self):
        """Minimum turnover mass for tracer"""
        return self.params["M_min"] - 5 * self.params["sig_logm"]


class Contreras13(HODPoisson):
    """
    Nine-parameter model of Contreras (2013) [1]_.

    Parameters
    ----------
    M_min : float, default = 11.6222
        Minimum mass of halo that supports a central galaxy
    M_1 : float, default = 12.851
        Mass of a halo which on average contains 1 satellite
    alpha : float, default = 1.049
        Index of power law for satellite galaxies
    sig_logm : float, default = 0.26
        Width of smoothed cutoff
    M_0 : float, default = 11.5047
        Minimum mass of halo containing satellites
    fca : float, default = 0.5
        fca
    fcb : float, default = 0
        fcb
    fs : float, default = 1
        fs
    delta : float, default  = 1
        delta
    x : float, default = 1
        x

    References
    ----------
    .. [1] Contreras, C. et al., "The WiggleZ Dark Energy Survey:
           measuring the cosmicgrowth rate with the two-point galaxy correlation function",
           https://ui.adsabs.harvard.edu/abs/2013MNRAS.430..924C.

    """

    _defaults = {
        "M_min": 11.6222,
        "M_1": 12.851,
        "alpha": 1.049,
        "M_0": 11.5047,
        "sig_logm": 0.26,
        "fca": 0.5,
        "fcb": 0,
        "fs": 1,
        "delta": 1,
        "x": 1,
    }

    def _central_occupation(self, m):
        """
        Amplitude of central tracer at mass M
        """
        return self.params["fcb"] * (1 - self.params["fca"]) * np.exp(
            -np.log10(m / 10 ** self.params["M_min"]) ** 2
            / (2 * (self.params["x"] * self.params["sig_logm"]) ** 2)
        ) + self.params["fca"] * (
            1
            + sp.erf(
                np.log10(m / 10 ** self.params["M_min"])
                / self.params["x"]
                / self.params["sig_logm"]
            )
        )

    def _satellite_occupation(self, m):
        """
        Amplitude of satellite tracer at mass M
        """
        return (
            self.params["fs"]
            * (
                1
                + sp.erf(np.log10(m / 10 ** self.params["M_1"]) / self.params["delta"])
            )
            * (m / 10 ** self.params["M_1"]) ** self.params["alpha"]
        )


class Geach12(Contreras13):
    """
    8-parameter model of Geach et. al. (2012) [1]_.

    This is identical to :class:`Contreras13`, but with `x==1`.

    References
    ----------https://ui.adsabs.harvard.edu/abs/2005ApJ...631...41T
    .. [1] Geach, J. E. et al., "The clustering of Hα emitters at z=2.23 from HiZELS ",
           https://ui.adsabs.harvard.edu/abs/2012MNRAS.426..679G.

    """

    pass


class Tinker05(Zehavi05):
    """
    3-parameter model of Tinker et. al. (2005) [1]_.

    References
    ----------
    .. [1] Tinker, J. L. et al., "On the Mass-to-Light Ratio of Large-Scale Structure",
           https://ui.adsabs.harvard.edu/abs/2005ApJ...631...41T.

    """

    _defaults = {"M_min": 11.6222, "M_1": 12.851, "M_cut": 12.0}
    central_condition_inherent = True

    def _satellite_occupation(self, m):
        """
        Amplitude of satellite tracer at mass M
        """
        out = self.central_occupation(m)
        return (
            out
            * np.exp(-(10 ** self.params["M_cut"]) / (m - 10 ** self.params["M_min"]))
            * (m / 10 ** self.params["M_1"])
        )


class Zehavi05WithMax(Zehavi05):
    """Zehavi05 model in which a maximum halo mass for occupancy also exists."""

    _defaults = {
        "M_min": 11.6222,
        "M_1": 12.851,
        "alpha": 1.049,
        "M_max": 18,  # Truncation mass
    }

    def _central_occupation(self, m):
        """
        Amplitude of central tracer at mass M
        """
        n_c = np.zeros_like(m)
        n_c[
            np.logical_and(
                m >= 10 ** self.params["M_min"], m <= 10 ** self.params["M_max"]
            )
        ] = 1

        return n_c

    def _satellite_occupation(self, m):
        """
        Amplitude of satellite tracer at mass M
        """
        return (m / 10 ** self.params["M_1"]) ** self.params["alpha"]


class Zehavi05Marked(Zehavi05WithMax):
    """
    The Zehavi05 model, with a possibility that the quantity is not number counts.

    NOTE: this should not give different results to Zehavi05 for any normalised statistic.
    """

    _defaults = {
        "M_min": 11.6222,
        "M_1": 12.851,
        "logA": 0.0,
        "alpha": 1.049,
        "M_max": 18.0,
    }

    def sigma_central(self, m):
        """The standard deviation of the central tracer amount in haloes of mass m."""
        co = super(Zehavi05Marked, self)._central_occupation(m)
        return np.sqrt(self._tracer_per_central(m) * co * (1 - co))

    def _tracer_per_central(self, m):
        """Number of tracer per central tracer source"""
        return 10 ** self.params["logA"]

    def _central_occupation(self, m):
        """
        Amplitude of central tracer at mass M
        """
        return super(Zehavi05Marked, self)._central_occupation(
            m
        ) * self._tracer_per_central(m)

    def _satellite_occupation(self, m):
        """
        Amplitude of satellite tracer at mass M
        """
        return super(Zehavi05Marked, self)._satellite_occupation(
            m
        ) * self._tracer_per_satellite(m)


class ContinuousPowerLaw(HODBulk):
    """
    A continuous HOD which is tuned to match the Zehavi05 total occupation except for normalisation.
    """

    _defaults = {
        "M_min": 11.6222,
        "M_1": 12.851,
        "logA": 0.0,
        "alpha": 1.049,
        "M_max": 18.0,
        "sigma_A": 0,  # The (constant) standard deviation of the tracer
    }
    sharp_cut = True

    def _satellite_occupation(self, m):
        """
        Amplitude of satellite tracer at mass M
        """
        alpha = self.params["alpha"]
        M_1 = 10 ** self.params["M_1"]
        A = 10 ** self.params["logA"]
        M_min = 10 ** self.params["M_min"]
        M_max = 10 ** self.params["M_max"]

        return np.where(
            np.logical_and(m >= M_min, m <= M_max), A * ((m / M_1) ** alpha + 1.0), 0,
        )

    def sigma_satellite(self, m):
        """The standard deviation of the satellite tracer amount in haloes of mass m."""
        return np.ones_like(m) * self.params["sigma_A"]


class Constant(HODBulk):
    """A toy model HOD in which every halo has the same amount of the tracer on average."""

    _defaults = {"logA": 0, "M_min": 11.0, "sigma_A": 0}

    def _satellite_occupation(self, m):
        """
        Amplitude of satellite tracer at mass M
        """
        return np.where(m > 10 ** self.params["M_min"], 10 ** self.params["logA"], 0)

    def sigma_satellite(self, m):
        """The standard deviation of the satellite tracer amount in haloes of mass m."""
        return np.ones_like(m) * self.params["sigma_A"]


class Spinelli19(HODPoisson):
    """
    Six-parameter model of Spinelli et al. (2019) [1]_.

    Default is taken to be the z=1 case, however you should manually set `hm.z=1`.

    References
    ----------
    .. [1] Spinelli, M. et al.,
           "The atomic hydrogen content of the post-reionization Universe",
           https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.5434S/abstract.
    """

    _defaults = {
        "a1": 0.0016,  # gives HI mass amplitude of the power law
        "a2": 0.00011,  # gives HI mass amplitude of the power law
        "alpha": 0.56,  # slop of exponential break
        "beta": 0.43,  # slop of mass
        "M_min": 9,  # Truncation Mass
        "M_break": 11.86,  # Characteristic Mass
        "M_1": -2.99,  # mass of exponential cutoff
        "sigma_A": 0,  # The (constant) standard deviation of the tracer
        "M_max": 18,  # Truncation mass
        "M_0": 8.31,  # Amplitude of satellite HOD
        "M_break_sat": 11.4,  # characteristic mass for satellite HOD
        "alpha_sat": 0.84,  # slop of exponential cut-off for satellite
        "beta_sat": 1.10,  # slop of mass for satellite
        "M_1_counts": 12.851,
        "alpha_counts": 1.049,
        "M_min_counts": 11,  # Truncation Mass
        "M_max_counts": 15,  # Truncation Mass
        "a": 0.049,
        "b": 2.248,
        "eta": 1.0,
    }
    sharp_cut = False
    central_condition_inherent = False

    def _central_occupation(self, m):
        """
        Amplitude of central tracer at mass M
        """
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        m_1 = 10 ** self.params["M_1"]
        a1 = self.params["a1"]
        a2 = self.params["a2"]
        m_break = 10 ** self.params["M_break"]

        out = (
            m
            * (a1 * (m / 1e10) ** beta * np.exp(-((m / m_break) ** alpha)) + a2)
            * np.exp(-((m_1 / m) ** 0.5))
        )
        return out

    def _satellite_occupation(self, m):
        """
        Amplitude of satellite tracer at mass M
        """
        alpha = self.params["alpha_sat"]
        beta = self.params["beta_sat"]
        amp = 10 ** self.params["M_0"]
        m1 = 10 ** self.params["M_break_sat"]
        array = np.zeros_like(m)
        array[m >= 10 ** 11] = 1

        return amp * (m / m1) ** beta * np.exp(-((m1 / m) ** alpha)) * array
        # return 10**8

    def unit_conversion(self, cosmo, z):
        "A factor (potentially with astropy units) to convert the total occupation to a desired unit."
        A12 = 2.869e-15
        nu21cm = 1.42e9
        Const = (3.0 * A12 * const.h * const.c ** 3.0) / (
            32.0 * np.pi * (const.m_p + const.m_e) * const.Boltzmann * nu21cm ** 2
        )
        Mpcoverh_3 = ((astroconst.kpc.value * 1e3) / (cosmo.H0.value / 100.0)) ** 3
        hubble = cosmo.H0.value * cosmo.efunc(z) * 1.0e3 / (astroconst.kpc.value * 1e3)
        temp_conv = Const * ((1.0 + z) ** 2 / hubble)
        # convert to Mpc^3, solar mass
        temp_conv = temp_conv / Mpcoverh_3 * astroconst.M_sun.value
        return temp_conv

    def _tracer_per_central(self, M):
        """Number of tracer per central tracer source"""
        n_c = np.zeros_like(M)
        n_c[
            np.logical_and(
                M >= 10 ** self.params["M_min_counts"],
                M <= 10 ** self.params["M_max_counts"],
            )
        ] = 1
        return n_c

    def _tracer_per_satellite(self, M):
        """Number of tracer per satellite tracer source"""
        n_s = np.zeros_like(M)
        index = np.logical_and(
            M >= 10 ** self.params["M_min_counts"],
            M <= 10 ** self.params["M_max_counts"],
        )
        n_s[index] = (M[index] / 10 ** self.params["M_1_counts"]) ** self.params[
            "alpha_counts"
        ]

        return n_s
