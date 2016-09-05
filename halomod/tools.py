'''
Created on Sep 9, 2013

@author: Steven
'''
import numpy as np
import scipy.integrate as intg
from scipy.stats import poisson
import time
from scipy.interpolate import InterpolatedUnivariateSpline as spline

def power_to_corr_ogata(power, k, r, N=640, h=0.005):
    """
    Use Ogata's method for Hankel Transforms in 3D for nu=0 (nu=1/2 for 2D)
    to convert a given power spectrum to a correlation function.
    """
    lnk = np.log(k)
    spl = spline(lnk, power)
    roots = np.arange(1, N + 1)
    t = h*roots
    s = np.pi*np.sinh(t)
    x = np.pi*roots*np.tanh(s/2)

    dpsi = 1 + np.cosh(s)
    dpsi[dpsi != 0] = (np.pi*t*np.cosh(t) + np.sinh(s))/dpsi[dpsi != 0]
    sumparts = np.pi*np.sin(x)*dpsi*x

    allparts = sumparts*spl(np.log(np.divide.outer(x, r))).T
    return np.sum(allparts, axis=-1)/(2*np.pi**2*r**3)


def power_to_corr_ogata_matrix(power, k, r, N=640, h=0.005):
    """
    Use Ogata's method for Hankel Transforms in 3D for nu=0 (nu=1/2 for 2D)
    to convert a given power spectrum to a correlation function.

    In this case, `power` is a (r,k) matrix, and the computation is slightly
    faster for less recalculations than looping over the original.
    """
    lnk = np.log(k)
    roots = np.arange(1, N + 1)
    t = h*roots
    s = np.pi*np.sinh(t)
    x = np.pi*roots*np.tanh(s/2)

    dpsi = 1 + np.cosh(s)
    dpsi[dpsi != 0] = (np.pi*t*np.cosh(t) + np.sinh(s))/dpsi[dpsi != 0]
    sumparts = np.pi*np.sin(x)*dpsi*x

    out = np.zeros(len(r))
    for ir, rr in enumerate(r):
        spl = spline(lnk, power[ir, :])
        allparts = sumparts*spl(np.log(x/rr))
        out[ir] = np.sum(allparts)/(2*np.pi**2*rr**3)
    return out


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
    minsteps = 8

    # set min_k, 1e-6 should be good enough
    mink = 1e-6

    temp_min_k = 1.0

    for i, r in enumerate(R):
        # getting maxk here is the important part. It must be a half multiple of
        # pi/r to be at a "zero", it must be >1 AND it must have a number of half
        # cycles > 38 (for 1E-5 precision).

        min_k = (2*np.ceil((temp_min_k*r/np.pi - 1)/2) + 0.5)*np.pi/r
        maxk = max(501.5*np.pi/r, min_k)

        # Now we calculate the requisite number of steps to have a good dk at hi-k.
        nk = np.ceil(np.log(maxk/mink)/np.log(maxk/(maxk - np.pi/(minsteps*r))))

        lnk, dlnk = np.linspace(np.log(mink), np.log(maxk), nk, retstep=True)
        P = power_func(lnk)
        integ = P*np.exp(lnk)**2*np.sin(np.exp(lnk)*r)/r

        corr[i] = (0.5/np.pi**2)*intg.simps(integ, dx=dlnk)

    return corr


def exclusion_window(k, r):
    """Top hat window function"""
    x = k*r
    return 3*(np.sin(x) - x*np.cos(x))/x**3


def populate(centres, masses, profile, hodmod):
    """
    Populate a series of DM halos with galaxies given a HOD model.

    Parameters
    ----------
    centres : (N,3)-array
        The cartesian co-ordinates of the centres of the halos

    masses : array_like
        The masses (in M_sun/h) of the halos

    profile : type :class:`profile.Profile`
        A density profile to use.

    hodmod : object of type :class:`hod.HOD`
        A HOD model to use to populate the dark matter.

    Returns
    -------
    array :
        (N,3)-array of positions of galaxies.
    """

    masses = np.array(masses)

    # Define which halos have central galaxies.
    cgal = np.random.binomial(1,hodmod.nc(masses))
    mask = cgal>0

    # Clear some memory
    masses = masses[mask]
    centres = centres[mask]

    # Calculate the number of satellite galaxies in halos
    sgal = poisson.rvs(hodmod.ns(masses))

    # Get an array ready, hopefully speeds things up a bit
    nhalos_with_gal = np.sum(cgal)
    allpos = np.empty((np.sum(sgal) + nhalos_with_gal, 3))

    # Assign central galaxy positions
    allpos[:nhalos_with_gal, :] = centres[mask]

    # Clean up some more memory
    del cgal

    mask = sgal > 0
    sgal = sgal[mask]
    centres = centres[mask]
    masses = masses[mask]


    # Now go through each halo and calculate galaxy positions
    begin = nhalos_with_gal
    start = time.time()
    for i, (m,n,ctr) in enumerate(zip(masses,sgal,centres)):
        end = begin + sgal[i]
        allpos[begin:end, :] = profile.populate(n, m, ba=1, ca=1) + ctr
        begin = end

    print "Took ", time.time() - start, " seconds, or ", (time.time() - start)/nhalos_with_gal, " each."
    print "MeanGal: ", np.mean(sgal + 1), "MostGal: ", sgal.max() + 1
    return allpos
