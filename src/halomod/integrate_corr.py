"""
Created on 10/03/2015

@author: Steven

Module for routines and _frameworks that intelligently integrate the real-space
correlation function
"""
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as _spline
from scipy.integrate import simps
from .halo_model import HaloModel
from hmf import cached_quantity, parameter
from hmf import Cosmology as csm
import warnings


class ProjectedCF(HaloModel):
    def __init__(
        self,
        rp_min=0.01,
        rp_max=50.0,
        rp_num=30,
        rp_log=True,
        proj_limit=None,
        **kwargs
    ):
        # Set default rnum
        if "rnum" not in kwargs:
            kwargs["rnum"] = 5 * rp_num

        super(ProjectedCF, self).__init__(**kwargs)

        self.proj_limit = proj_limit
        self.rp_min = rp_min
        self.rp_max = rp_max
        self.rp_num = rp_num
        self.rp_log = rp_log

    @parameter("switch")
    def rp_min(self, val):
        return val

    @parameter("option")
    def rp_log(self, val):
        return bool(val)

    @parameter("res")
    def rp_max(self, val):
        return val

    @parameter("res")
    def rp_num(self, val):
        if val < 0:
            raise ValueError("rp_num must be > 0")
        return int(val)

    @parameter("switch")
    def proj_limit(self, val):
        return val

    @cached_quantity
    def rp(self):
        if isinstance(self.rp_min, (list, np.ndarray)):
            rp = np.array(self.rp_min)
        else:
            if self.rp_log:
                rp = np.logspace(
                    np.log10(self.rp_min), np.log10(self.rp_max), self.rp_num
                )
            else:
                rp = np.linspace(self.rp_min, self.rp_max, self.rp_num)

        return rp

    @cached_quantity
    def rlim(self):
        if self.proj_limit is None:
            rlim = max(80.0, 5 * self.rp.max())
        else:
            rlim = self.proj_limit
        return rlim

    @cached_quantity
    def r(self):
        return np.logspace(np.log10(self.rp.min()), np.log10(self.rlim), self.rnum)

    @cached_quantity
    def projected_corr_gal(self):
        """
        Projected correlation function w(r_p).

        From Beutler 2011, eq 6.

        To integrate perform a substitution y = x - r_p.
        """
        return projected_corr_gal(self.r, self.corr_gg, self.rlim, self.rp)


def projected_corr_gal(
    r: np.ndarray, xir: np.ndarray, rlim: np.ndarray, rp_out: [None, np.ndarray] = None
):
    """
    Projected correlation function w(r_p).

    From Beutler 2011, eq 6.

    To integrate, we perform a substitution y = x - r_p.

    Parameters
    ----------
    r : float array
        Array of scales for the 3D correlation function, in [Mpc/h]
    xir : float array
        3D correlation function Array of xi(r), unitless.
    rlim :
    """
    if rp_out is None:
        rp_out = r

    lnr = np.log(r)
    lnxi = np.log(xir)

    p = np.zeros_like(rp_out)
    fit = _spline(r, xir, k=3)  # [self.corr_gal > 0] maybe?
    f_peak = 0.01
    a = 0

    for i, rp in enumerate(rp_out):
        if a != 1.3 and i < len(r) - 1:
            # Get log slope at rp
            ydiff = (lnxi[i + 1] - lnxi[i]) / (lnr[i + 1] - lnr[i])
            # if the slope is flatter than 1.3, it will converge faster, but to make
            # sure, we cut at 1.3
            a = max(1.3, -ydiff)
            theta = _get_theta(a)

        min_y = theta * f_peak ** 2 * rp

        # Get the upper limit for this rp
        ylim = rlim - rp

        # Set the y vector for this rp
        y = np.logspace(np.log(min_y), np.log(ylim), 1000, base=np.e)

        # Integrate
        integ_corr = fit(y + rp)
        integrand = (y + rp) * integ_corr / np.sqrt((y + 2 * rp) * y)
        p[i] = simps(integrand, y) * 2

    return p


def _get_theta(a):
    theta = (
        2 ** (1 + 2 * a)
        * (
            7
            - 2 * a ** 3
            + 3 * np.sqrt(5 - 8 * a + 4 * a ** 2)
            + a ** 2 * (9 + np.sqrt(5 - 8 * a + 4 * a ** 2))
            - a * (13 + 3 * np.sqrt(5 - 8 * a + 4 * a ** 2))
        )
        * ((1 + np.sqrt(5 - 8 * a + 4 * a ** 2)) / (a - 1)) ** (-2 * a)
    )
    theta /= (a - 1) ** 2 * (-1 + 2 * a + np.sqrt(5 - 8 * a + 4 * a ** 2))
    return theta


def flat_z_dist(zmin, zmax):
    def ret(z):
        z = np.atleast_1d(z)
        return np.where(np.logical_and(z >= zmin, z <= zmax), 1.0 / (zmax - zmin), 0)

    return ret


def dxdz(z, cosmo=csm().cosmo):
    """Derivative of comoving distance with redshift [Mpc/h]."""
    dh = cosmo.hubble_distance * cosmo.h
    return dh.value / cosmo.efunc(z)


class AngularCF(HaloModel):
    """
    Framework extension to angular correlation functions.

    Parameters
    ----------
    p1 : callable, optional
        The redshift distribution of the sample. This needs not
        be normalised to 1, as this will occur internally. May be
        either a function of radial distance [Mpc/h] or redshift.
        If a function of radial distance, `p_of_z` must be set to
        False. Default is a flat distribution in redshift.
    p2 : callable, optional
        See `p1`. This can optionally be a different function against
        which to cross-correlate. By default is equivalent to `p1`.
    theta_min, theta_max : float, optional
        min,max angular separations [Rad]
    theta_num : int, optional
        Number of steps in angular separation
    theta_log : bool, optional
        Whether to use logspace for theta values
    zmin, zmax : float, optional
        The redshift limits of the sample distribution. Note that
        this is in redshit, regardless of the value of `p_of_z`.
    znum : int, optional
        Number of steps in redshift grid.
    logu_min, logu_max : float, optional
        min,max of the log10 of radial separation grid [Mpc/h]. Must be large
        enough to let the integral over the 3D correlation function to converge.
    unum : int, optional
        Number of steps in the u grid.
    check_p_norm : bool, optional
        If False, cancels checking the normalisation of `p1` and `p2`.
    p_of_z : bool, optional
        Whether `p1` and `p2` are functions of redshift.
    kwargs : unpacked-dict
        Any keyword arguments passed down to :class:`halomod.HaloModel`.
    """

    def __init__(
        self,
        p1=None,
        p2=None,
        theta_min=1e-3 * np.pi / 180.0,
        theta_max=np.pi / 180.0,
        theta_num=30,
        theta_log=True,
        zmin=0.2,
        zmax=0.4,
        znum=100,
        logu_min=-4,
        logu_max=2.3,
        unum=100,
        check_p_norm=True,
        p_of_z=True,
        **kwargs
    ):
        super(AngularCF, self).__init__(**kwargs)

        if self.z < zmin or self.z > zmax:
            warnings.warn(
                "Your specified redshift (z=%s) is not within your selection function, z=(%s,%s)"
                % (self.z, zmin, zmax)
            )

        if p1 is None:
            p1 = flat_z_dist(zmin, zmax)

        self.p1 = p1
        self.p2 = p2
        self.zmin = zmin
        self.zmax = zmax
        self.znum = znum
        self.logu_min = logu_min
        self.logu_max = logu_max
        self.unum = unum
        self.check_p_norm = check_p_norm
        self.p_of_z = p_of_z

        self.theta_min = theta_min
        self.theta_max = theta_max
        self.theta_num = theta_num
        self.theta_log = theta_log

    @parameter("param")
    def p1(self, val):
        return val

    @parameter("param")
    def p2(self, val):
        return val

    @parameter("model")
    def p_of_z(self, val):
        return val

    @parameter("res")
    def theta_min(self, val):
        if val < 0:
            raise ValueError("theta_min must be > 0")
        return val

    @parameter("res")
    def theta_max(self, val):
        if val > 180.0:
            raise ValueError("theta_max must be < 180.0")
        return val

    @parameter("res")
    def theta_num(self, val):
        return val

    @parameter("res")
    def theta_log(self, val):
        return val

    @parameter("param")
    def zmin(self, val):
        return val

    @parameter("param")
    def zmax(self, val):
        return val

    @parameter("res")
    def znum(self, val):
        return val

    @parameter("res")
    def logu_min(self, val):
        return val

    @parameter("res")
    def logu_max(self, val):
        return val

    @parameter("res")
    def unum(self, val):
        return val

    @parameter("option")
    def check_p_norm(self, val):
        return val

    @cached_quantity
    def zvec(self):
        """
        Redshift distribution grid.
        """
        return np.linspace(self.zmin, self.zmax, self.znum)

    @cached_quantity
    def uvec(self):
        """Radial separation grid [Mpc/h]."""
        return np.logspace(self.logu_min, self.logu_max, self.unum)

    @cached_quantity
    def xvec(self):
        """Radial distance grid (corresponds to zvec) [Mpc/h]."""
        return self.cosmo.comoving_distance(self.zvec).value

    @cached_quantity
    def theta(self):
        """Angular separations, [Rad]."""
        if self.theta_min > self.theta_max:
            raise ValueError("theta_min must be less than theta_max")

        if self.theta_log:
            return np.logspace(
                np.log10(self.theta_min), np.log10(self.theta_max), self.theta_num
            )
        else:
            return np.linspace(self.theta_min, self.theta_max, self.theta_num)

    @cached_quantity
    def r(self):
        """Physical separation grid [Mpc/h]."""
        rmin = np.sqrt(
            (10 ** self.logu_min) ** 2 + self.theta.min() ** 2 * self.xvec.min() ** 2
        )
        rmax = np.sqrt(
            (10 ** self.logu_max) ** 2 + self.theta.max() ** 2 * self.xvec.max() ** 2
        )
        return np.logspace(np.log10(rmin), np.log10(rmax), self.rnum)

    @cached_quantity
    def angular_corr_gal(self):
        """The angular correlation function w(theta).

        From Blake+08, Eq. 33
        """

        def xi(r):
            s = _spline(self.r, self.corr_gg)
            return s(r)

        return angular_corr_gal(
            self.theta,
            xi,
            self.p1,
            self.zmin,
            self.zmax,
            self.logu_min,
            self.logu_max,
            znum=self.znum,
            unum=self.unum,
            p2=self.p2,
            check_p_norm=self.check_p_norm,
            cosmo=self.cosmo,
            p_of_z=self.p_of_z,
        )

    @cached_quantity
    def angular_corr_matter(self):
        """
        The angular correlation function w(theta).

        From Blake+08, Eq. 33
        """

        def xi(r):
            s = _spline(self.r, self.corr_mm)
            return s(r)

        return angular_corr_gal(
            self.theta,
            xi,
            self.p1,
            self.zmin,
            self.zmax,
            self.logu_min,
            self.logu_max,
            znum=self.znum,
            unum=self.unum,
            p2=self.p2,
            check_p_norm=self.check_p_norm,
            cosmo=self.cosmo,
            p_of_z=self.p_of_z,
        )


def _check_p(p, z):
    if hasattr(p, "integral"):
        integ = p.integral(z.min(), z.max())
    else:
        integ = simps(p(z), z)
    if not np.isclose(integ, 1.0, rtol=0.01):
        print(
            "WARNING: Filter function p(x) did not integrate to 1 (%s). Tentatively re-normalising."
            % integ
        )
        return lambda z: p(z) / integ
    else:
        return p


def angular_corr_gal(
    theta,
    xi,
    p1,
    zmin,
    zmax,
    logu_min,
    logu_max,
    znum=100,
    unum=100,
    p2=None,
    check_p_norm=True,
    cosmo=None,
    p_of_z=True,
    **xi_kw
):
    """
    Calculate the angular correlation function w(theta).

    From Blake+08, Eq. 33. That is, this uses the Limber approximation.
    This does not hold either for wide angles, or thin radial distributions.

    Parameters
    ----------
    theta : array_like
        Angles at which to calculate the angular correlation. In radians.
    xi : callable
        A function of one variable: r [Mpc/h], which returns
        the 3D correlation function at the scale r.
    p1: callable
        The redshift distribution of sources. Should integrate to 1 between
        `logz_min` and `logz_max`. A callable function of a single variable, z.
    zmin, zmax : float
        The redshift limits of the sample distribution. Note that
        this is in redshift, regardless of the value of `p_of_z`.
    logu_min, logu_max : float
        min,max of the log10 of radial separation grid [Mpc/h]. Must be large
        enough to let the integral over the 3D correlation function to converge.
    znum : int, optional
        Number of steps in redshift grid.
    unum : int, optional
        Number of steps in the u grid.
    p2 : callable, optional
        The same as `p1`, but for a second, cross-correlating dataset. If not
        provided, defaults to `p1` (i.e. auto-correlation).
    check_p_norm : bool, optional
        If False, cancels checking the normalisation of `p1` and `p2`.
    p_of_z : bool, optional
        Whether `p1` and `p2` are functions of redshift.
    cosmo : `hmf.cosmo.Cosmology` instance, optional
        A cosmology, used to generate comoving distance from redshift. Default
        is the default cosmology of the `hmf` package.
    xi_kw : unpacked-dict
        Any arguments to `xi` other than r,z.

    Returns
    -------
    wtheta : array_like
        The angular correlation function corresponding to `theta`.
    """
    if cosmo is None:
        cosmo = csm().cosmo

    # Arrays
    u = np.logspace(logu_min, logu_max, unum)

    if p_of_z:
        z = np.linspace(zmin, zmax, znum)
        x = (cosmo.comoving_distance(z) * cosmo.h).value
    else:
        xmin = (cosmo.comoving_distance(zmin) * cosmo.h).value
        xmax = (cosmo.comoving_distance(zmax) * cosmo.h).value
        x = np.linspace(xmin, xmax, znum)

    if check_p_norm:
        p1 = _check_p(p1, z if p_of_z else x)

    if p2 is None:
        p2 = p1
    elif check_p_norm:
        p2 = _check_p(p2, z if p_of_z else x)

    p_integ = p1(z) * p2(z) / dxdz(z, cosmo) if p_of_z else p1(x) * p2(x)
    R = np.sqrt(np.add.outer(np.outer(theta ** 2, x ** 2), u ** 2)).flatten()

    integrand = np.einsum(
        "kij,i->kij", xi(R, **xi_kw).reshape((len(theta), len(x), len(u))), p_integ
    )

    return 2 * simps(simps(integrand, u), x)
