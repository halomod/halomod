'''
Created on 10/03/2015

@author: Steven

Module for routines and _frameworks that intelligently integrate the real-space
correlation function
'''
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.integrate import simps
from halo_model import HaloModel
from hmf._cache import cached_property, parameter
from astropy import units as u
from tools import dblsimps

class ProjectedCF(HaloModel):
    def __init__(self, rp_min=0.01, rp_max=50.0, rp_num=30, rp_log=True, proj_limit=None, **kwargs):
        # Set default rnum
        if "rnum" not in kwargs:
            kwargs['rnum'] = 5*rp_num

        super(ProjectedCF, self).__init__(**kwargs)

        self.proj_limit = proj_limit
        self.rp_min = rp_min
        self.rp_max = rp_max
        self.rp_num = rp_num
        self.rp_log = rp_log

    @parameter
    def rp_min(self, val):
        return val

    @parameter
    def rp_log(self, val):
        return bool(val)

    @parameter
    def rp_max(self, val):
        return val

    @parameter
    def rp_num(self, val):
        if val < 0:
            raise ValueError("rp_num must be > 0")
        return int(val)

    @parameter
    def proj_limit(self, val):
        return val


    @cached_property("rp_min", "rp_max", "rp_num")
    def rp(self):
        if type(self.rp_min) == list or type(self.rp_min) == np.ndarray:
            rp = np.array(self.rp_min)
        else:
            if self.rp_log:
                rp = np.logspace(np.log10(self.rp_min), np.log10(self.rp_max), self.rp_num)
            else:
                rp = np.linspace(self.rp_min, self.rp_max, self.rp_num)

        return rp * u.Mpc / self._hunit

    @cached_property("proj_limit", "rp_max")
    def rlim(self):
        if self.proj_limit is None:
            rlim = max(80.0, 5 * self.rp.value.max())
        else:
            rlim = self.proj_limit
        return rlim * u.Mpc / self._hunit

    @cached_property("rp_min", "rlim", "rnum")
    def r(self):
        return np.logspace(np.log10(self.rp.min().value), np.log10(self.rlim.value), self.rnum) * u.Mpc / self._hunit

    @cached_property("r", "corr_gal", "rlim", "rp")
    def projected_corr_gal(self):
        """
        Projected correlation function w(r_p).

        From Beutler 2011, eq 6.

        To integrate perform a substitution y = x - r_p.
        """
        return projected_corr_gal(self.r, self.corr_gal, self.rlim, self.rp)

def projected_corr_gal(r, xir, rlim, rp_out=None):
    """
    Projected correlation function w(r_p).

    From Beutler 2011, eq 6.

    To integrate, we perform a substitution y = x - r_p.

    Parameters
    ----------
    r : float array
        Array of scales, in [Mpc/h]

    xir : float array
        Array of xi(r), unitless
    """
    if rp_out is None:
        rp_out = r

    lnr = np.log(r.value)
    lnxi = np.log(xir)

    p = np.zeros_like(rp_out)
    fit = spline(r, xir, k=3)  # [self.corr_gal > 0] maybe?
    f_peak = 0.01
    a = 0

    for i, rp in enumerate(rp_out):
        if a != 1.3 and i < len(r) - 1:
            # Get log slope at rp
            ydiff = (lnxi[i + 1] - lnxi[i]) / (lnr[i + 1] - lnr[i])
            # if the slope is flatter than 1.3, it will converge faster, but to make sure, we cut at 1.3
            a = max(1.3, -ydiff)
            theta = _get_theta(a)

        min_y = theta * f_peak ** 2 * rp

        # Get the upper limit for this rp
        ylim = rlim - rp

        # Set the y vector for this rp
        y = np.logspace(np.log(min_y.value), np.log(ylim.value), 1000, base=np.e) * rp.unit

        # Integrate
        integ_corr = fit(y + rp)
        integrand = (y + rp) * integ_corr / np.sqrt((y + 2 * rp) * y)
        p[i] = simps(integrand, y) * 2 * r.unit

    return p

def _get_theta(a):
    theta = 2 ** (1 + 2 * a) * (7 - 2 * a ** 3 + 3 * np.sqrt(5 - 8 * a + 4 * a ** 2) + a ** 2 * (9 + np.sqrt(5 - 8 * a + 4 * a ** 2)) -
                       a * (13 + 3 * np.sqrt(5 - 8 * a + 4 * a ** 2))) * ((1 + np.sqrt(5 - 8 * a + 4 * a ** 2)) / (a - 1)) ** (-2 * a)
    theta /= (a - 1) ** 2 * (-1 + 2 * a + np.sqrt(5 - 8 * a + 4 * a ** 2))
    return theta


class AngularCF(HaloModel):
    """
    Framework extension to angular correlation functions.

    Parameters
    ----------
    f : float array
        A float array specifying the pdf of galaxy counts as a function of
        comoving distance from the Galaxy.

    x : float array
        The comoving distances which correspond to f.

    theta_min : float, 0.01
        Minimum theta value (in degrees)

    theta_max : float, 10.0
        Maximum theta value (in degrees)

    theta_num : int, 30
        Number of theta values

    theta_log : bool, True
        Whether to use logspace for theta values

    NOTE: this is definitely not optimal yet. How should f be passed??
    """
    umin = -3
    umax = 2.2
    def __init__(self, f, x, theta_min=0.01, theta_max=10.0, theta_num=30, theta_log=True,
                 **kwargs):
        super(AngularCF, self).__init__(**kwargs)

        self.f = f
        self.x = x
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.theta_num = theta_num
        self.theta_log = theta_log

    @parameter
    def f(self, val):
        return val

    @parameter
    def x(self, val):
        return val

    @parameter
    def theta_min(self, val):
        if val < 0:
            raise ValueError("theta_min must be > 0")
        return val

    @parameter
    def theta_max(self, val):
        if val > 180.0:
            raise ValueError("theta_max must be < 180.0")
        return val

    @parameter
    def theta_num(self, val):
        return val

    @parameter
    def theta_log(self, val):
        return val


    @cached_property("theta_min", "theta_max", "theta_num", "theta_log")
    def theta(self):
        if self.theta_min > self.theta_max:
            raise ValueError("theta_min must be greater than theta_max")

        if self.theta_log:
            return np.logspace(np.log10(self.theta_min), np.log10(self.theta_max), self.theta_num)
        else:
            return np.linspace(self.theta_min, self.theta_max, self.theta_num)

    @cached_property("theta", "x", "rnum")
    def r(self):
        rmin = np.sqrt((10 ** self.umin) ** 2 + self.theta.min() ** 2 * self.x.min() ** 2)
        rmax = np.sqrt((10 ** self.umax) ** 2 + self.theta.max() ** 2 * self.x.max() ** 2)
        return np.logspace(np.log10(rmin), np.log10(rmax), self.rnum) * u.Mpc / self._hunit

    @cached_property("f", "x", "theta", "r", "corr_gal")
    def angular_corr_gal(self):
        """
        Calculate the angular correlation function w(theta).

        From Blake+08, Eq. 33
        """
        return angular_corr_gal(self.f, self.x, self.theta, self.r,
                                self.corr_gal, self.umin, self.umax)


def angular_corr_gal(f, x, theta, r, corr_gal, umin, umax):
    """
    Calculate the angular correlation function w(theta).

    From Blake+08, Eq. 33

    Parameters
    ----------
    f : function
        A function of a single variable which returns the normalised
        quantity of sources at a given comoving distance x.

    x : float array
        The comoving distances which correspond to f.

    theta : float array
        Values of the angular separation (degrees)

    r : float array
        Real-space separation

    corr_gal : float array
        Real-space correlation function corresponding to ``r``.

    umin : float
        The minimum value of the integral variable

    umax : float
        The maximum value of the integral variable
    """

    # Initialise result
    w = np.zeros_like(theta)

    # Setup vectors u^2,x^2 and f(x)
    u = np.logspace(umin, umax, 500)
    u2 = u * u
    x2 = x * x
    du = u[1] - u[0]
    dx = x[1] - x[0]
    xfx = x * f ** 2  # multiply by x because of log integration

    # Set up spline for xi(r)
    xi = spline(r, corr_gal, k=3)

    for i, th in enumerate(theta):
        # Set up matrix integrand (for double-integration).
        r = np.sqrt(np.add.outer(th * th * x2, u2)).flatten()  # # needs to be 1d for spline eval
        integrand = np.einsum("ij,i,j->ij", xi(r).reshape((len(x2), len(u2))), xfx, u)  # reshape here for dblsimps, mult by u for log int
        w[i] = dblsimps(integrand, dx, du)

    return w * 2 * np.log(10) ** 2
