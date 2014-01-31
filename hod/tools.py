'''
Created on Sep 9, 2013

@author: Steven
'''
import numpy as np
import scipy.integrate as intg
import pycamb
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.stats import poisson
import profiles
def power_to_corr(power_func, R):
    """
    Calculate the correlation function given a power spectrum
    
    Parameters
    ----------
    power_func : callable
        A callable function which returns the natural log of power given lnk
        
    R : array_like
        The values of separation/scale to calculate the correlation at.
        
    """
    if not np.iterable(R):
        R = [R]

    corr = np.zeros_like(R)

    # the number of steps to fit into a half-period at high-k. 6 is better than 1e-4.
    minsteps = 6

    # set min_k, 1e-6 should be good enough
    mink = 1e-6

    temp_min_k = 1.0

    for i, r in enumerate(R):
        # getting maxk here is the important part. It must be an odd multiple of
        # pi/r to be at a "zero", it must be >1 AND it must have a number of half
        # cycles > 38 (for 1E-5 precision).

        min_k = (2 * np.ceil((temp_min_k * r / np.pi - 1) / 2) + 1) * np.pi / r
        maxk = max(39 * np.pi / r, min_k)


        # Now we calculate the requisite number of steps to have a good dk at hi-k.
        nk = np.ceil(np.log(maxk / mink) / np.log(maxk / (maxk - np.pi / (minsteps * r))))

        lnk, dlnk = np.linspace(np.log(mink), np.log(maxk), nk, retstep=True)
        P = np.exp(power_func(lnk))
        integ = P * np.exp(lnk) ** 2 * np.sin(np.exp(lnk) * r) / r

        corr[i] = (0.5 / np.pi ** 2) * intg.simps(integ, dx=dlnk)

    return corr


def non_linear_power(lnk_out=None, **camb_kwargs):
    """
    Calculates the non-linear power spectrum from camb + halofit and outputs
    it at the given lnk_out if given.
    
    INPUT
    lnk_out: [None] The values of ln(k) at which the power spectrum should be output
    **camb_kwargs: any argument for CAMB
    
    OUTPUT
    lnk: The lnk values at which the power is evaluated (k in units of k/h)
         If lnk_out is given, will be equivalent to lnk_out
    lnp: The log of the nonlinear power from halofit. 
    """

    k, P = pycamb.matter_power(NonLinear=1, **camb_kwargs)

    if lnk_out is not None:
        # FIXME: I have to put this disgusting cut (lnk>-5.3), because sometimes P comes
        # out with a massive drop below some k
        power_func = spline(np.log(k), np.log(P), k=1)
        P = np.exp(power_func(lnk_out[lnk_out > -5.3]))
        k = np.exp(lnk_out[lnk_out > -5.3])

    return np.log(k), np.log(P)

def virial_mass(r, mean_dens, delta_halo):
    """
    Returns the virial mass of a given halo radius
    """
    return 4 * np.pi * r ** 3 * mean_dens * delta_halo / 3

def virial_radius(m, mean_dens, delta_halo):
    """
    Returns the virial mass of a given halo radius
    """
    return ((3 * m) / (4 * np.pi * mean_dens * delta_halo)) ** (1. / 3.)

def overlapping_halo_prob(r, rv1, rv2):
    """
    The probability of non-overlapping ellipsoidal haloes (Tinker 2005 Appendix B)
    """
    if np.isscalar(rv1) and np.isscalar(rv2):
        x = r / (rv1 + rv2)
    else:
        x = r / np.add.outer(rv1, rv2)
    y = (x - 0.8) / 0.29

    if np.isscalar(y):
        if y <= 0:
            return 0
        elif y >= 1:
            return 1

    res = 3 * y ** 2 - 2 * y ** 3
    res[y <= 0] = 0.0
    res[y >= 1] = 1.0
    return res

def exclusion_window(k, r):
    """Top hat window function"""
    x = k * r
    return 3 * (np.sin(x) - x * np.cos(x)) / x ** 3

def dblsimps(X, dx, dy):
    """
    Perform double integration using simpsons rule
    """
    if len(X.shape) != 2:
        raise ValueError("dblsimps takes a matrix")
    if X.shape[0] % 2 != 1 or X.shape[1] != 1:
        # For now, knowing what's going in here, we just cut off the last value
        X = X[:-1, :-1]

    W = np.ones_like(X)

    W[range(1, len(W[:, 0]) - 1, 2), :] *= 4
    W[:, range(1, len(W[0, :]) - 1, 2)] *= 4
    W[range(2, len(W[:, 0]) - 1, 2), :] *= 2
    W[:, range(2, len(W[0, :]) - 1, 2)] *= 2

    return dx * dy * np.sum(W * X) / 9.0


def populate(centres, masses, delta_halo, omegam, z, profile, cm_relation,
             hodmod, truncate=True):
    """
    Populate a series of DM halos with galaxies given a HOD model.
    
    Parameters
    ----------
    centres : (N,3)-array 
        The cartesian co-ordinates of the centres of the halos
        
    masses : array_like
        The masses (in M_sun/h) of the halos
        
    delta_halo : float
        The overdensity of a halo w.r.t. mean density.
        
    omegam : float
        The fractional matter density
        
    z : float
        The redshift
        
    profile : str
        A density profile to use. Calls a subclass of :class:`profile.Profile`.
        
    cm_relation : str
        A concentration-mass relation to use
        
    hodmod : object of type :class:`hod.HOD`
        A HOD model to use to populate the dark matter.
        
    truncate : bool, default True
        Whether the profile should be truncated at the virial radius.
    
    Returns
    -------
    array :
        (N,3)-array of positions of galaxies.
        
    """

    # Define which halos have central galaxies.
    cgal = np.zeros_like(masses)
    masses = np.array(masses)

    cgal[np.random.rand() < hodmod.nc(masses)] = 1.0
    # Calculate the number of satellite galaxies in halos
    sgal = np.zeros_like(masses)
    sgal[cgal != 0.0] = poisson.rvs(hodmod.ns(masses[cgal != 0.0]))

    # Get an array ready, hopefully speeds things up a bit
    nhalos_with_gal = np.sum(cgal)

    allpos = np.empty((np.sum(sgal) + nhalos_with_gal, 3))
    allpos[:nhalos_with_gal, :] = centres[cgal > 0]
    prof = profiles.get_profile(profile, omegam, 1 - omegam, -1, delta_halo, cm_relation, truncate)
    begin = nhalos_with_gal
    mask = sgal > 0
    sgal = sgal[mask]
    centres = centres[mask]
    M = masses[mask]
    # Now go through each halo and calculate galaxy positions
    import time
    start = time.time()
    for i, m in enumerate(M):
        # print i, m
        end = begin + sgal[i]
        allpos[begin:end, :] = prof.populate(sgal[i], m, ba=1, ca=1, z=z) + centres[i, :]
        begin = end
    print "Took ", time.time() - start, " seconds, or ", (time.time() - start) / nhalos_with_gal, " each."
    print "z: ", z, "Centrals: ", float(len(masses)) / np.sum(cgal), "MeanGal: ", np.mean(sgal + 1) , "MostGal: ", sgal.max() + 1
    return allpos

