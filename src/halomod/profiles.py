r"""
Module defining halo density profile.

The halo density profile is used to describe the density distribution of dark matter or a specific type of tracer within a dark matter halo. It is usually descirbed as a function

``rho(r|rho_s,r_s) = rho_s f(x=r/r_s)``

Here ``rho_s`` is the amplitude of the density, ``r_s = r_vir/c`` is related to the concentration and the virial radius of the halo.

The profile used in power spectrum calculation is usually the Fourier Transformed one, which is usually a function of ``u(K=kr_s)``.

Profile models are defined as :class:`~hmf.Component` instances -- that is,
they are flexible models that the user can subclass and use in the halo model framework.
See :class:`Profile` for instructions on how to use ``Profile`` models. The following notes
will mostly describe how to use various models provided in the literature.

All models are specified in terms of the ``f(x)``, and analytically transformed to Fourier space, if
an analytical formulae can be obtained.

As with all ``Component`` subclasses, arbitrary user-specified variables can be received
by defining them in the `_defaults` class-level dictionary.

The module also defines a :class:`ProfileInf`, which does not truncate the dark matter halo at ``x=c``. Mathematically, it does not require a concentration-mass relation as an input. Here, an arbitary :class:`~halomod.CMRelation` should be plugged in, and results will remain the same.

Examples
--------
Use NFW profile in a halo model::

    >>> from halomod import HaloModel
    >>> hm = HaloModel(halo_profile_model="NFW")

You can also specify a different profile for tracer if you're working with
:class:`~halomod.halo_model.TracerHaloModel` ::
    >>> from halomod import HaloModel
    >>> hm = HaloModel(halo_profile_model="NFW",tracer_profile_model="CoredNFW")

Notice that tracer density profile density should be used only in inverse volume or dimensionless unit.
"""
import numpy as np
import scipy.special as sp
import scipy.integrate as intg
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.interpolate import RectBivariateSpline
import mpmath
from hmf import Component
from scipy.special import gammainc, gamma
import os
import warnings
from scipy.special import sici
from hmf.halos.mass_definitions import SOMean
from astropy.cosmology import Planck15
import hankel
from scipy.integrate import quad
from hmf._internals import pluggable


def ginc(a, x):
    r"""
    ``gamma(a)*gammainc(a,x)``
    """
    return gamma(a) * gammainc(a, x)


@pluggable
class Profile(Component):
    """
    Halo radial density profiles.

    This class provides basic building blocks for all kinds of fun with halo
    radial density profiles. It is modeled on the system described in
    arXiv:2009.14066. This means that subclasses providing
    specific profiles shapes, f(x) must provide very minimal other information
    for a range of products to be available.

    The "main" quantities available are the halo_profile itself, its fourier pair,
    and its convolution (this is not available for every halo_profile). Furthermore,
    quantities such as the halo_concentration-mass relation are provided, along with
    tools such as those to generate a mock halo of the given halo_profile.

    Parameters
    ----------
    cm_relation : :class:`~halomod.CMRelation` instance
        Identifies which halo-concentration-mass relation to use.
    mdef : :class:`hmf.halos.mass_definitions.MassDefinition` instance
        A mass definition to interpret input masses with.
    z : float, default 0.0
        The redshift of the halo
    """

    _defaults = {}

    def __init__(
        self, cm_relation, mdef=SOMean(), z=0.0, cosmo=Planck15, **model_parameters
    ):

        self.mdef = mdef
        self.delta_halo = self.mdef.halo_overdensity_mean(z, cosmo)
        self.z = z
        self._cm_relation = cm_relation
        self.mean_dens = mdef.mean_density(z=z, cosmo=cosmo)
        self.mean_density0 = mdef.mean_density(0, cosmo=cosmo)
        self.has_lam = hasattr(self, "_l")

        super(Profile, self).__init__(**model_parameters)

    def halo_mass_to_radius(self, m, at_z=False):
        """Return the halo radius corresponding to ``m``.

        Note that this is the radius corresponding to the halo at redshift zero,
        even if the redshift of the profile is not zero.
        """
        # I'm not absolutely sure that it's correct to use mean_density0 here,
        # rather than mean_dens (i.e. a function of redshift). Using mean_density0
        # lines up with HMCode, which I kind of trust, but it seems odd to me that
        # the radius of a halo of a given mass at a given redshift should only depend on the
        # background density at z=0.
        dens = self.mean_dens if at_z else self.mean_density0
        return (3 * m / (4 * np.pi * self.delta_halo * dens)) ** (1.0 / 3.0)

    def halo_radius_to_mass(self, r, at_z=False):
        """Return the halo mass corresponding to ``r``."""
        dens = self.mean_dens if at_z else self.mean_density0

        return 4 * np.pi * r ** 3 * self.delta_halo * dens / 3

    def _rs_from_m(self, m, c=None, at_z=False):
        """
        Return the scale radius for a halo of mass m.

        Parameters
        ----------
        m : float
            mass of the halo
        c : float, default None
            halo_concentration of the halo (if None, use cm_relation to get it).
        """
        if c is None:
            c = self.cm_relation(m)
        r = self.halo_mass_to_radius(m, at_z=at_z)
        return r / c

    def scale_radius(
        self, m: [float, np.ndarray], at_z: bool = False
    ) -> [float, np.ndarray]:
        """
        Return the scale radius for a halo of mass m.

        The scale radius is defined as :math:`r_s = r_vir(m) / c(m).

        Parameters
        ----------
        m
            Mass of the halo(s), in units of M_sun / h.
        at_z
            If true, return the redshift-dependent configuration-space scale radius of
            the halo. Otherwise, return the redshift-independent Lagrangian-space scale
            radius (based on an initial density patch).

        Returns
        -------
        r_s
            The scale radius, same type as ``m``.
        """
        return self._rs_from_m(m=m, at_z=at_z)

    def virial_velocity(self, m=None, r=None):
        """
        Return the virial velocity for a halo of mass ``m``.

        Either `m` or `r` must be passed. If both are passed, ``m`` is preferentially used.

        Parameters
        ----------
        m : array_like, optional
            Masses of halos.
        r : array_like, optional
            Radii of halos.
        """
        if m is None and r is None:
            raise ValueError("Either m or r must be specified")
        if m is not None:
            r = self.halo_mass_to_radius(m)
        else:
            m = self.halo_radius_to_mass(r)
        return np.sqrt(6.673 * 1e-11 * m / r)

    def _h(self, c=None, m=None) -> [float, np.ndarray]:
        """
        The integral of f(x)*x^2 out to c

        .. note:: This function should be replaced with an analytic solution if
                  possible in derived classes.

        Parameters
        ----------
        c : float or array_like, optional
            The halo_concentration(s) of the halo(s). Used ONLY if m is not specified.
        m : float or array_like, optional
            The mass of the halo. Determines the halo_concentration if provided.
        """
        if c is None and m is None:
            raise ValueError("Either c or m must be provided.")
        if m is not None:
            c = self.cm_relation(m)

        x, dx = np.linspace(1e-6, np.max(c), 2000, retstep=True)
        integrand = self._f(x) * x ** 2

        integ = intg.cumtrapz(integrand, dx=dx, initial=0)

        if not hasattr(c, "__len__"):
            return integ[-1]
        else:
            sp = spline(x, integ, k=3)
            return sp(c)

    def _p(self, K: np.ndarray, c: np.ndarray):
        r"""
        The reduced dimensionless fourier-transform of the halo_profile

        This function should not need to be called by the user in general.

        Parameters
        ----------
        K : float or array_like
            The unit-less wavenumber k*r_s
        c : float or array_like
            The halo_concentration

        Notes
        -----
        .. note :: This should be replaced by an analytic function if possible

        The formula is

        .. math:: \int_0^c x \sin(\kappa x) / \kappa f(x) dx

        where :math:`kappa` is the unitless wavenumber `k*r_s`, and `x` is the unitless
        radial co-ordinate `r/r_s`. This is simply the scaled 3D fourier transform
        of the profile, taken to a Hankel transform.
        """

        c = np.atleast_1d(c)
        if K.ndim < 2:
            if len(K) != len(c):
                K = np.atleast_2d(K)  # should be len(k) * len(rs)
            else:
                K = np.atleast_2d(K).T

        assert K.ndim == 2
        assert K.shape[1] == len(c)

        sort_indx = np.argsort(c)

        # We get a shorter vector of different K's to find the integral for, otherwise
        # we need to do a full integral for every K (which is 2D, since K is different
        # for every c).
        kk = np.logspace(np.log10(K.min()), np.log10(K.max()), 100)

        intermediate_res = np.zeros((len(c), len(kk)))

        # To make it more efficient, we do the integral in parts cumulatively, so we
        # can get the value at each c in turn.
        for j, k in enumerate(kk):
            # Get all zeros up to the maximum c
            zeros = np.pi / k * np.arange(c.max() // (np.pi / k))
            for i, indx in enumerate(sort_indx):
                # Get the limits on c for this iteration.
                c_0 = 0 if not i else c[sort_indx[i - 1]]
                c_1 = c[indx]
                points = zeros[(c_0 < zeros) & (zeros < c_1)]
                integral = quad(
                    lambda x: x * self._f(x) * np.sin(k * x) / k,
                    c_0,
                    c_1,
                    points=points,
                    limit=max(50, len(points) + 1),
                )[0]

                # If its not the minimum c, add it to the previous integrand.
                if i:
                    intermediate_res[indx, j] = (
                        intermediate_res[sort_indx[i - 1], j] + integral
                    )
                else:
                    intermediate_res[indx, j] = integral

        # Now we need to interpolate onto the actual K values we have at each c.
        out = np.zeros_like(K)
        for ic, integral in enumerate(intermediate_res):
            spl = spline(kk, integral)
            out[:, ic] = spl(K[:, ic])

        return out

    def _rho_s(self, c, r_s=None, norm=None):
        """
        The amplitude factor of the halo_profile

        Parameters
        ----------
        c : float or array of floats
            The halo_concentration parameter
        norm : str or None, {None,"m","rho"}
            Normalisation for the amplitude. Can either be None (in which case
            the output is a density), "m" (in which case the output is inverse
            volume) or "rho" in which case the output is dimensionless.
        r_s : float or array of floats
            The scale radius. This is only required if ``norm`` is "m".
        """
        if norm is None:
            rho = c ** 3 * self.delta_halo * self.mean_dens / (3 * self._h(c))
        elif norm == "m":
            rho = 1.0 / (4 * np.pi * r_s ** 3 * self._h(c))
        elif norm == "rho":
            rho = c ** 3 * self.delta_halo / (3 * self._h(c))

        return self._reduce(rho)

    def rho(self, r, m, norm=None, c=None, coord="r"):
        """
        The density at radius r of a halo of mass m.

        Parameters
        ----------
        r : float or array of floats
            The radial location(s). The units vary according to :attr:`coord`
        m : float or array of floats
            The mass(es) of the halo(s)
        norm : str, {``m``,``rho``}
            Normalisation of the density.
        c : float or array of floats, default ``None``
            Concentration(s) of the halo(s). Must be same length as :attr:`m`.
        coord : str, {``r``,``x``,``s``}
            What the radial coordinate represents. ``r`` represents physical
            co-ordinates [units Mpc/h]. ``x`` is in units of the scale radius
            (x(r_vir) = c), and ``s`` is in units of the virial radius (s(r_vir) = 1).
        """
        c, r_s, x = self._get_r_variables(r, m, c, coord)
        rho = self._f(x) * self._rho_s(c, r_s, norm)
        rho[x > c] = 0.0

        return self._reduce(rho)

    def u(self, k, m, norm="m", c=None, coord="k"):
        """
        The (optionally normalised) Fourier-transform of the density halo_profile

        Parameters
        ----------
        k : float or array of floats
            The radial wavenumber(s). The units vary according to :attr:`coord`
        m : float or array of floats
            The mass(es) of the halo(s)
        norm : str, {``None``,``m``,``rho``}
            Normalisation of the density.
        c : float or array of floats, default ``None``
            Concentration(s) of the halo(s). Must be same length as :attr:`m`.
        coord : str, {``k``,``kappa``}
            What the radial coordinate represents. ``r`` represents physical
            wavenumbers [units h/Mpc]. ``kappa`` is in units of the scale radius,
            kappa = k*rs.
        """
        c, K = self._get_k_variables(k, m, c, coord)

        u = self._p(K, c) / self._h(c)

        if norm is None:
            u *= m
        elif norm != "m":
            raise ValueError(str(norm) + "is not a valid value for norm")

        return self._reduce(u)

    def lam(self, r, m, norm="m", c=None, coord="r"):
        """
        The density halo_profile convolved with itself.

        Parameters
        ----------
        r : float or array of floats
            The radial location(s). The units vary according to :attr:`coord`
        m : float or array of floats
            The mass(es) of the halo(s)
        norm : str, {``None``,``m``,``rho``}
            Normalisation of the density.
        c : float or array of floats, default ``None``
            Concentration(s) of the halo(s). Must be same length as :attr:`m`.
        coord : str, {``r``,``x``,``s``}
            What the radial coordinate represents. ``r`` represents physical
            co-ordinates [units Mpc/h]. ``x`` is in units of the scale radius
            (r_vir = c), and ``s`` is in units of the virial radius (r_vir = 1).
        """
        if self.has_lam:
            c, r_s, x = self._get_r_variables(r, m, c, coord)
            if norm in [None, "m"]:
                lam = self._l(x, c) * r_s ** 3 * self._rho_s(c, r_s, norm) ** 2
            else:
                raise ValueError("norm must be None or 'm'")
        else:
            raise AttributeError("this halo_profile has no self-convolution defined.")
        return self._reduce(lam)

    def cdf(self, r, c=None, m=None, coord="r"):
        """
        The cumulative distribution function, :math:`m(<x)/m_v`

        Parameters
        ----------
        r : float or array_like
            The radial location -- units defined by :attr:`coord`
        c : float or array_like, optional
            The halo_concentration. Only used if m not provided
        m : float or array_like, optional
            The mass of the halo. Defines the halo_concentration if provided.
        coord : str, {``"x"``, ``"r"``, ``"s"``}
            What the radial coordinate represents. ``r`` represents physical
            co-ordinates [units Mpc/h]. ``x`` is in units of the scale radius
            (r_vir = c), and ``s`` is in units of the virial radius (r_vir = 1).
        """
        c, r_s, x = self._get_r_variables(r, m, c, coord)
        return self._h(x) / self._h(c)

    def cm_relation(self, m: [float, np.ndarray]) -> [float, np.ndarray]:
        """
        The halo_concentration-mass relation
        """
        return self._cm_relation.cm(m, self.z)

    def _get_r_variables(self, r, m, c=None, coord="r"):
        """
        From a raw array in r, mass, returns halo_concentration,
        scale radius and x=r*c/rvir.

        Returns
        -------
        c : same shape as m
            halo_concentration
        r_s : same shape as m
            Scale radius
        x : 2d array
            Dimensionless scale parameter, shape (r,[m]).
        """
        if c is None:
            c = self.cm_relation(m)
        r_s = self._rs_from_m(m, c)

        if coord == "r":
            x = np.divide.outer(r, r_s)
        elif coord == "x":
            x = r
        elif coord == "s":
            x = np.outer(r, c)
        else:
            raise ValueError(f"coord must be one of 'r', 'x' or 's', got '{coord}'.")
        return c, r_s, x

    def _get_k_variables(self, k, m, c=None, coord="k"):
        """
        From a raw array in k, mass, returns halo_concentration,
        kappa.

        Returns
        -------
        c : same shape as m
            halo_concentration
        K : 1d or 2d array
            Dimensionless scale parameter, shape (r,[m]).
        """
        if c is None:
            c = self.cm_relation(m)
        r_s = self._rs_from_m(m, c)

        if coord == "k":
            K = np.outer(k, r_s) if np.iterable(k) and np.iterable(r_s) else k * r_s
        elif coord == "kappa":
            K = k

        return c, K

    def _reduce(self, x):
        x = np.squeeze(np.atleast_1d(x))
        if x.size == 1:
            try:
                return x[0]
            except IndexError:
                return x.dtype.type(x)
        else:
            return x

    def populate(self, n, m, c=None, centre=np.zeros(3)):
        """
        Populate a halo with the current halo profile of mass ``m`` with ``n`` tracers.

        Parameters
        ----------
        n : int
            Number of tracers to place down
        m : float
            Mass of the halo.
        c : float, optional
            Concentration of the halo. Will be calculated if not given.
        centre : 3-array
            (x,y,z) co-ordinates of centre of halo

        Returns
        -------
        pos : (N,3)-array
            Array of positions of the tracers, centred around (0,0,0).
        """
        c, r_s, x = self._get_r_variables(np.linspace(0, 1, 1000), m, c, coord="s")

        cdf = self.cdf(x, c, m, coord="x")
        spl = spline(cdf, x, k=3)

        rnd = np.random.uniform(size=n)
        x = spl(rnd)

        r = r_s * x
        pos = np.random.normal(size=(3, n))
        pos *= r / np.sqrt(np.sum(pos ** 2, axis=0))
        return pos.T + centre


class ProfileInf(Profile, abstract=True):
    """
    An extended halo_profile (not truncated at x=c)
    """

    def rho(self, r, m, norm=None, c=None, coord="r"):
        """
        The density at radius r of a halo of mass m.

        Parameters
        ----------
        r : float or array of floats
            The radial location(s). The units vary according to :attr:`coord`
        m : float or array of floats
            The mass(es) of the halo(s)
        norm : str, {``None``,``m``,``rho``}
            Normalisation of the density.
        c : float or array of floats, default ``None``
            Concentration(s) of the halo(s). Must be same length as :attr:`m`.
        coord : str, {``r``,``x``,``s``}
            What the radial coordinate represents. ``r`` represents physical
            co-ordinates [units Mpc/h]. ``x`` is in units of the scale radius
            (r_vir = c), and ``s`` is in units of the virial radius (r_vir = 1).
        """
        c, r_s, x = self._get_r_variables(r, m, c, coord)

        rho = self._f(x) * self._rho_s(c, r_s, norm)

        return self._reduce(rho)

    def u(self, k, m, norm="m", c=None, coord="k"):
        """
        The fourier-transform of the density halo_profile

        Parameters
        ----------
        k : float or array of floats
            The radial wavenumber(s). The units vary according to :attr:`coord`
        m : float or array of floats
            The mass(es) of the halo(s)
        norm : str, {``None``,``m``,``rho``}
            Normalisation of the density.
        c : float or array of floats, default ``None``
            Concentration(s) of the halo(s). Must be same length as :attr:`m`.
        coord : str, {``k``,``kappa``}
            What the radial coordinate represents. ``r`` represents physical
            co-ordinates [units Mpc/h]. ``x`` is in units of the scale radius
            (r_vir = c), and ``s`` is in units of the virial radius (r_vir = 1).
        """
        c, K = self._get_k_variables(k, m)

        u = self._p(K) / self._h(c)

        if norm is None:
            u *= m
        elif norm != "m":
            raise ValueError(str(norm) + "is not a valid value for norm")

        return self._reduce(u)

    def _p(self, K: np.ndarray, c: np.ndarray):
        """
        The dimensionless fourier-transform of the halo_profile

        This should be replaced by an analytic function if possible.
        """
        assert K.ndim == 2
        assert K.shape[0] == len(c)

        ft = hankel.SymmetricFourierTransform(ndim=3, N=640, h=0.005)

        out = np.zeros_like(K)

        # Go through each value of c
        for i, kk in enumerate(K):
            out[i] = ft.transform(self._f, k=K, ret_err=False, ret_cumsum=False)

        return out

    def lam(self, r, m, norm=None, c=None, coord="r"):
        """
        The density profile convolved with itself.

        Parameters
        ----------
        r : float or array of floats
            The radial location(s). The units vary according to :attr:`coord`
        m : float or array of floats
            The mass(es) of the halo(s)
        norm : str, {``None``,``m``,``rho``}
            Normalisation of the density.
        c : float or array of floats, default ``None``
            Concentration(s) of the halo(s). Must be same length as :attr:`m`.
        coord : str, {``r``,``x``,``s``}
            What the radial coordinate represents. ``r`` represents physical
            co-ordinates [units Mpc/h]. ``x`` is in units of the scale radius
            (r_vir = c), and ``s`` is in units of the virial radius (r_vir = 1).
        """
        c, r_s, x = self._get_r_variables(r, m, c, coord)
        if self.has_lam:
            if norm in [None, "m"]:
                lam = self._l(x) * r_s ** 3 * self._rho_s(c, r_s, norm) ** 2
            else:
                raise ValueError("norm must be None or 'm'")
        else:
            raise AttributeError("this halo_profile has no self-convolution defined.")
        return self._reduce(lam)


class NFW(Profile):
    r"""
    Canonical Density Profile of Navarro, Frenk & White(1997).

    See documentation for :class:`Profile` for information on input parameters. This
    model has no free parameters.

    Notes
    -----
    This is an empirical form proposed in [1]_ and [2]_, with the formula

    .. math:: \rho(r) = \frac{\rho_s}{r/R_s\big(1+r/R_s\big)^2}


    References
    ----------
    .. [1] Navarro, Julio F., Frenk, Carlos S. and White, Simon D. M., "The Structure of Cold Dark
           Matter Halos", https://ui.adsabs.harvard.edu/abs/1996ApJ...462..563N.
    .. [2] Navarro, Julio F., Frenk, Carlos S. and White, Simon D. M., "A Universal Density Profile
           from Hierarchical Clustering",
           https://ui.adsabs.harvard.edu/abs/1997ApJ...490..493N.
    """

    def _f(self, x):
        return 1.0 / (x * (1 + x) ** 2)

    def _h(self, c):
        return np.log(1.0 + c) - c / (1.0 + c)

    def _p(self, K, c=None):
        bs, bc = sp.sici(K)
        asi, ac = sp.sici((1 + c) * K)
        return (
            np.sin(K) * (asi - bs)
            - np.sin(c * K) / ((1 + c) * K)
            + np.cos(K) * (ac - bc)
        )

    def _l(self, x, c):
        x = np.atleast_1d(x)
        c = np.atleast_1d(c)
        result = np.zeros_like(x)

        if np.all(x > 2 * c):
            return result  # Stays as zero

        if x.ndim == 2:
            c = np.outer(np.ones(x.shape[0]), c)

        if x.ndim == 1:
            c = np.ones(x.shape[0]) * c

        # Get low values
        if np.any(x <= c):
            mask = x <= c
            x_lo = x[mask]
            # c_lo = c[mask]
            a_lo = 1.0 / c[mask]

            f2_lo = (
                -4 * (1 + a_lo) + 2 * a_lo * x_lo * (1 + 2 * a_lo) + (a_lo * x_lo) ** 2
            )
            f2_lo /= 2 * (x_lo * (1 + a_lo)) ** 2 * (2 + x_lo)
            f3_lo = (
                np.log((1 + a_lo - a_lo * x_lo) * (1 + x_lo) / (1 + a_lo)) / x_lo ** 3
            )
            f4 = np.log(1 + x_lo) / (x_lo * (2 + x_lo) ** 2)
            result[mask] = 4 * np.pi * (f2_lo + f3_lo + f4)
        # And high values
        if np.any(np.logical_and(x < 2 * c, x > c)):
            mask = np.logical_and(x > c, x <= 2 * c)
            x_hi = x[mask]
            a_hi = 1.0 / c[mask]

            f2_hi = np.log((1 + a_hi) / (a_hi + a_hi * x_hi - 1)) / (
                x_hi * (2 + x_hi) ** 2
            )
            f3_hi = (x_hi * a_hi ** 2 - 2 * a_hi) / (
                2 * x_hi * (1 + a_hi) ** 2 * (2 + x_hi)
            )
            result[mask] = 4 * np.pi * (f2_hi + f3_hi)

        return result


class NFWInf(NFW, ProfileInf):
    r"""
    The same as NFW profile, but not truncated at x=c.
    """

    def _p(self, K, c=None):
        bs, bc = sp.sici(K)
        return 0.5 * ((np.pi - 2 * bs) * np.sin(K) - 2 * np.cos(K) * bc)

    def _l(self, x, c=None):

        f1 = 8 * np.pi / (x ** 2 * (x + 2))
        f2 = ((x ** 2 + 2 * x + 2) * np.log(1 + x)) / (x * (x + 2)) - 1

        return f1 * f2


class Hernquist(Profile):
    r"""
    Halo Density Profile of Hernquist(1990).

    See documentation for :class:`Profile` for information on input parameters. This
    model has no free parameters.

    Notes
    -----
    This is an empirical form proposed in [1]_, with the formula

    .. math:: \rho(r) = \frac{\rho_s}{r/R_s\big(1+r/R_s\big)^3}


    References
    ----------
    .. [1] Hernquist, L., "An Analytical Model for Spherical Galaxies and Bulges",
    https://ui.adsabs.harvard.edu/abs/1990ApJ...356..359H.
    """

    def _f(self, x):
        return 1.0 / (x * (1 + x) ** 3)

    def _h(self, c):
        return c ** 2 / (2 * (1 + c) ** 2)

    def _p(self, K, c):

        sk, ck = sp.sici(K)
        skp, ckp = sp.sici(K + c * K)

        f1 = K * ck * np.sin(K) - K * np.cos(K) * sk - 1
        f2 = -((1 + c) * K * np.cos(c * K) + np.sin(c * K)) / (1 + c) ** 2
        f3 = K ** 2 * (ckp * np.sin(K) - np.cos(K) * skp)

        return (-K / 2 * f1 + 0.5 * (f2 + f3)) / K


class HernquistInf(Hernquist, ProfileInf):
    r"""
    The same as Hernquist profile, but not truncated at x=c.
    """

    def _p(self, K):
        si, ci = sp.sici(K)

        return 0.25 * (2 - K * (2 * ci * np.sin(K) + np.cos(K) * (np.pi - 2 * si)))

    def _l(self, x):

        h1 = (24 + 60 * x + 56 * x ** 2 + 24 * x ** 3 + 6 * x ** 4 + x ** 5) / (1 + x)
        h2 = 12 * (1 + x) * (2 + 2 * x + x ** 2) * np.log(1 + x) / x

        return 4 * np.pi * 4 * (h1 - h2) / (x ** 4 * (2 + x) ** 4)


class Moore(Profile):
    r"""
    Halo Density Profile of Moore(1998).

    See documentation for :class:`Profile` for information on input parameters. This
    model has no free parameters.

    Notes
    -----
    This is an empirical form proposed in [1]_ and [2]_, with the formula

    .. math:: \rho(r) = \frac{\rho_s}{\big(r/R_s\big)^{1.5}\big(1+r/R_s\big)^{1.5}}


    References
    ----------
    .. [1] Moore, B. et al., "Resolving the Structure of Cold Dark Matter Halos",
           https://ui.adsabs.harvard.edu/abs/1998ApJ...499L...5M.
    .. [2] Moore, B. et al., "Cold collapse and the core catastrophe ",
           https://ui.adsabs.harvard.edu/abs/1999MNRAS.310.1147M.
    """

    def _f(self, x):
        return 1.0 / (x ** 1.5 * (1 + x ** 1.5))

    def _h(self, c):
        return 2.0 * np.log(1 + c ** 1.5) / 3

    def cm_relation(self, m):
        c = super(Moore, self).cm_relation(m)
        c = (c / 1.7) ** 0.9

        return c

    def _rs_from_m(self, m, c=None):
        r_s = super(Moore, self)._rs_from_m(m, c)
        return r_s * c / (c / 1.7) ** 0.9


class MooreInf(Moore, ProfileInf):
    r"""
    The same with Moore, but not truncated at x=c.
    """

    def _p(self, K):
        def G(k):
            return mpmath.meijerg(
                [[1.0 / 2.0, 3.0 / 4.0, 1.0], []],
                [
                    [
                        1.0 / 12.0,
                        1.0 / 4.0,
                        5.0 / 12.0,
                        0.5,
                        3.0 / 4.0,
                        3.0 / 4.0,
                        1.0,
                    ],
                    [-1.0 / 12.0, 7.0 / 12.0],
                ],
                k ** 6 / 46656.0,
            ) / (4 * np.sqrt(3) * np.pi ** (5 / 2) * k)

        if K.ndim == 2:
            K1 = np.reshape(K, -1)
            K1.sort()
        else:
            K1 = K
        res = np.zeros(len(K[K < 10 ** 3.2]))
        for i, k in enumerate(K1[K1 < 10 ** 3.2]):
            res[i] = G(k)

        fit = spline(np.log(K1[K1 < 10 ** 3.2]), np.log(res), k=1)
        res = np.reshape(
            np.exp(fit(np.log(np.reshape(K, -1)))), (len(K[:, 0]), len(K[0, :]))
        )

        return res


class Constant(Profile):
    r"""
    A constant density profile ``rho=rho_s``.

    See documentation for :class:`Profile` for information on input parameters. This
    model has no free parameters.
    """

    def _f(self, x):
        return 1.0

    def _h(self, c):
        return c ** 3 / 3.0

    def _p(self, K, c):
        return (-c * K * np.cos(c * K) + np.sin(c * K)) / K ** 3


class GeneralizedNFW(Profile):
    r"""
    Generalized NFW profile.

    This module has an extra free parameter ``alpha``.

    Notes
    -----
    This is an empirical form which is a special case of the formula in [1]_:

    .. math:: \rho(r) = \frac{\rho_s}{\big(r/R_s\big)^{\alpha}\big(1+r/R_s\big)^{3-\alpha}}

    Other Parameters
    ----------------
    alpha: float
        The default value is ``1``.

    References
    ----------
    .. [1] Zhao, H., "Analytical models for galactic nuclei",
           https://ui.adsabs.harvard.edu/abs/1996MNRAS.278..488Z.
    """
    _defaults = {"alpha": 1}

    def _f(self, x):
        return 1.0 / (x ** self.params["alpha"] * (1 + x) ** (3 - self.params["alpha"]))

    # def _h(self, c=None, m=None):
    #     if c is None and m is None:
    #         raise ValueError("Either c or m must be provided.")
    #     if m is not None:
    #         c = self.cm_relation(m)
    #
    #     c = np.complex(c)
    #     f1 = -((-c) ** self.params['alpha']) * c ** self.params['alpha']
    #     f2 = mpmath.betainc(-c, 3 - self.params['alpha'], self.params['alpha'] - 2)
    #     return (f1 * f2).real


class GeneralizedNFWInf(GeneralizedNFW, ProfileInf):
    r"""
    The same with generalized NFW, but not truncated at x=c.
    """

    def _p(self, K):
        def G(k):
            return mpmath.meijerg(
                [
                    [
                        (self.params["alpha"] - 2) / 2.0,
                        (self.params["alpha"] - 1) / 2.0,
                    ],
                    [],
                ],
                [[0, 0, 0.5], [-0.5]],
                k ** 2 / 4,
            ) / (np.sqrt(np.pi) * sp.gamma(3 - self.params["alpha"]))

        if len(K.shape) == 2:
            K1 = np.reshape(K, -1)
            K1.sort()
        else:
            K1 = K
        res = np.zeros(len(K[K < 10 ** 3.2]))
        for i, k in enumerate(K1[K1 < 10 ** 3.2]):
            res[i] = G(k)

        fit = spline(np.log(K1[K1 < 10 ** 3.2]), np.log(res), k=1)
        res = np.reshape(
            np.exp(fit(np.log(np.reshape(K, -1)))), (len(K[:, 0]), len(K[0, :]))
        )

        return res


class Einasto(Profile):
    r"""
    An Einasto halo profile.

    It has two extra free parameters, ``alpha`` and ``use_interp``.

    This halo profile has no analytic Fourier Transform. The numerical FT has been
    pre-computed and is by default used to interpolate to the correct solution. If the
    full numerical calculation is preferred, set the model parameter ``use_interp`` to
    ``False``. The interpolation speeds up the calculation by at least 10 times.

    Notes
    -----
    This is an empirical form which is a special case of the formula in [1]_:

    .. math:: \rho(r) = \rho_s{\rm exp}\bigg[-\frac{2}{\alpha}\Big(\big(\frac{r}{r_s}\big)^\alpha-1\Big)\bigg]

    Other Parameters
    ----------------
    alpha : float
        The default value is ``0.18``.

    use_interp : boolean
        The default value is ``True``.
    References
    ----------
    .. [1] Einasto , J., "Kinematics and dynamics of stellar systems",
           Trudy Inst. Astrofiz. Alma-Ata 5, 87.
    """

    _defaults = {"alpha": 0.18, "use_interp": True}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.params["alpha"] != 0.18 and self.params["use_interp"]:
            warnings.warn(
                "Einasto interpolation for p(K,c) is only defined for alpha=0.18, switching off."
            )
            self.params["use_interp"] = False

    def _f(self, x):
        a = self.params["alpha"]
        return np.exp((-2.0 / a) * (x ** a - 1))

    def _h(self, c):
        a = self.params["alpha"]
        return (
            np.exp(2 / a)
            * (2 / a) ** (-3.0 / a)
            * ginc(3.0 / a, (2.0 / a) * c ** a)
            / a
        )

    def _p(self, K, c):
        if self.params["use_interp"]:
            data_path = os.path.join(os.path.dirname(__file__), "data")
            data = np.load(os.path.join(data_path, "uKc_einasto.npz"))

            pk = data["pk"]
            _k = data["K"]
            _c = data["c"]

            c = np.atleast_1d(c)
            if np.isscalar(K):
                K = np.atleast_2d(K)
            if K.ndim < 2:
                if len(K) != len(c):
                    K = np.atleast_2d(K).T  # should be len(rs) x len(k)
                else:
                    K = np.atleast_2d(K)
            pk[pk <= 0] = 1e-8

            spl = RectBivariateSpline(np.log(_k), np.log(_c), np.log(pk))
            cc = np.repeat(c, K.shape[0])
            return np.exp(
                self._reduce(spl.ev(np.log(K.flatten()), np.log(cc)).reshape(K.shape))
            )
        else:  # Numerical version.
            return super(Einasto, self)._p(K, c)


class CoredNFW(Profile):
    r"""
    Cored NFW profile.

    See documentation for :class:`Profile` for information on input parameters. This
    model has no free parameters.

    Notes
    -----
    This is an empirical form proposed in [1]_, with the formula

    .. math:: \rho(r) = \frac{\rho_s}{\big(r/R_s+0.75\big)\big(1+r/R_s\big)^2}


    References
    ----------
    .. [1] Maller, A. and Bullock, J., "Multiphase galaxy formation:high-velocity clouds and the
    missing baryon problem ",
    https://ui.adsabs.harvard.edu/abs/2004MNRAS.355..694M.
    """

    def _f(self, x):
        return 1.0 / (x + 0.75) / (x + 1) ** 2

    def _h(self, c):
        return (
            -4 * (-(2 * c + 3) / (c + 1) + 2 * np.log(c + 1))
            + 9 * np.log(c + 0.75)
            - 12
            - 9 * np.log(0.75)
        )

    def _p(self, K, c):
        def antideriv(k, x):
            si1, ci1 = sici(k * (x + 1))
            si2, ci2 = sici(k * (x + 0.75))

            return (1.0 / k) * (
                12 * (np.cos(k) * si1 - np.sin(k) * ci1)
                + 4 * k * (np.cos(k) * ci1 + np.sin(k) * si1)
                - 4 * np.sin(k * x) / (x + 1)
                + -12 * (np.cos(0.75 * k) * si2 - np.sin(0.75 * k) * ci2)
            )

        return antideriv(K, c) - antideriv(K, 0)


class PowerLawWithExpCut(ProfileInf):
    r"""
    A simple power law with exponential cut-off.

    Default is taken to be the `z=1` case of [1]_.


    Notes
    -----
    This is an empirical form proposed with the formula

    .. math:: \rho(r) = \rho_s * R_s^b / r^b * e^{-ar/R_s}


    References
    ----------
    .. [1] Spinelli, M. et al.,
           "The atomic hydrogen content of the post-reionization Universe",
           https://ui.adsabs.harvard.edu/abs/2020MNRAS.493.5434S/abstract.
    """

    _defaults = {"a": 0.049, "b": 2.248}

    def _f(self, x):
        return 1.0 / (x ** self.params["b"]) * np.exp(-self.params["a"] * x)

    def _h(self, c=None):
        return (
            gamma(3 - self.params["b"])
            * self.params["a"] ** (self.params["b"] - 3)
            * np.ones_like(c)
        )

    def _p(self, K, c=None):
        b = self.params["b"]
        a = self.params["a"]
        if b == 2:
            return np.arctan(K / a) / K
        else:
            return (
                -1
                / K
                * (
                    (a ** 2 + K ** 2) ** (b / 2 - 1)
                    * gamma(2 - b)
                    * np.sin((b - 2) * np.arctan(K / a))
                )
            )
