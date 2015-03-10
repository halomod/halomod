'''
Created on 10/03/2015

@author: Steven

Module for routines that intelligently integrate the correlation function
'''
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.integrate import simps

def projected_corr_gal(r, xir, rp_out=None):
    """
    Projected correlation function w(r_p).

    From Beutler 2011, eq 6.

    To integrate perform a substitution y = x - r_p.
    
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
            # Get slope at rp (== index of power law at rp)
            ydiff = (lnxi[i + 1] - lnxi[i]) / (lnr[i + 1] - lnr[i])
            # if the slope is flatter than 1.3, it will converge faster, but to make sure, we cut at 1.3
            a = max(1.3, -ydiff)
            theta = _get_theta(a)

        min_y = theta * f_peak ** 2 * rp

        # Get the upper limit for this rp
        lim = r[-1] - rp

        # Set the y vector for this rp
        y = np.logspace(np.log(min_y.value), np.log(lim.value), 1000, base=np.e) * rp.unit

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

def angular_corr_gal(self, f, theta_min, theta_max, theta_num, logtheta=True,
                     x_min=0, x_max=10000):
    """
    Calculate the angular correlation function w(theta).
    
    From Blake+08, Eq. 33
    
    Parameters
    ----------
    f : function
        A function of a single variable which returns the normalised
        quantity of sources at a given comoving distance x.
        
    theta_min : float
        Minimum theta value (in radians)
        
    theta_max : float
        Maximum theta value (in radians)
        
    theta_num : int
        Number of theta values
        
    logtheta : bool, True
        Whether to use logspace for theta values
    """
    # Set up theta values
    if theta_min <= 0 or theta_min >= theta_max:
        raise ValueError("theta_min must be > 0 and < theta_max")
    if theta_max >= 180.0:
        raise ValueError("theta_max must be < pi")

    if logtheta:
        theta = 10 ** np.linspace(np.log10(theta_min), np.log10(theta_max), theta_num)
    else:
        theta = np.linspace(theta_min, theta_max, theta_num)

    if x_min <= 0 or x_min >= x_max:
        raise ValueError("x_min must be >0 and < x_max")


    # Initialise result
    w = np.zeros_like(theta)

    umin = -3
    umax = 2.2

    rmax = np.sqrt((10 ** umax) ** 2 + theta_max ** 2 * x_max ** 2)
    if rmax > 1.2 * self.rmax:
        print "WARNING: likely bad extrapolation in angular c.f. rmax = %s >> %s" % (rmax, self.rmax)

    # Setup vectors u^2,x^2 and f(x)
    u = np.logspace(umin, umax, 500)
    u2 = u * u
    x = np.logspace(np.log10(x_min), np.log10(x_max), 500)
    x2 = x * x
    du = u[1] - u[0]
    dx = x[1] - x[0]
    xfx = x * f(x) ** 2  # multiply by x because of log integration

    # Set up spline for xi(r)
    xi = spline(self.r, self.corr_gal, k=3)

    for i, th in enumerate(theta):
        # Set up matrix integrand (for double-integration).
        r = np.sqrt(np.add.outer(th * th * x2, u2)).flatten()  # # needs to be 1d for spline eval
        integrand = np.einsum("ij,i,j->ij", xi(r).reshape((len(x2), len(u2))), xfx, u)  # reshape here for dblsimps, mult by u for log int
        w[i] = dblsimps(integrand, dx, du)

    return w * 2 * np.log(10) ** 2, theta
