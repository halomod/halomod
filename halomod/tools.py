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


def populate(centres, masses, halomodel=None, profile=None, hodmod=None, edges=None):
    """
    Populate a series of DM halos with galaxies given a HOD model.

    Parameters
    ----------
    centres : (N,3)-array
        The cartesian co-ordinates of the centres of the halos

    masses : array_like
        The masses (in M_sun/h) of the halos

    halomodel : type :class:`halomod.HaloModel`
        A HaloModel object pre-instantiated. One can either use this, or
        *both* `profile` and `hodmod` arguments.

    profile : type :class:`profile.Profile`
        A density profile to use.

    hodmod : object of type :class:`hod.HOD`
        A HOD model to use to populate the dark matter.

    edges : float, len(2) iterable, or (2,3)-array
        Periodic box edges. If float, defines the upper limit of cube, with lower limit at zero.
        If len(2) iterable, defines edges of cube.
        If (2,3)-array, specifies edges of arbitrary rectangular prism.

    Returns
    -------
    pos : array
        (N,3)-array of positions of galaxies.

    halo : array
        (N)-array of associated haloes (by index)

    H : int
        Number of haloes with galaxies. The first H galaxies in pos/halo correspond to centrals.
    """
    if halomodel is not None:
        profile = halomodel.profile
        hodmod = halomodel.hod

    masses = np.array(masses)

    # Define which halos have central galaxies.
    cgal = np.random.binomial(1, hodmod.nc(masses))
    mask = cgal > 0
    central_halos = np.arange(len(masses))[mask]

    # Clear some memory
    masses = masses[mask]
    centres = centres[mask]

    # Calculate the number of satellite galaxies in halos
    sgal = poisson.rvs(hodmod.ns(masses))

    # Get an array ready, hopefully speeds things up a bit
    nhalos_with_gal = np.sum(cgal)
    pos = np.empty((np.sum(sgal) + nhalos_with_gal, 3))
    halo = np.empty(np.sum(sgal) + nhalos_with_gal)

    # Assign central galaxy positions
    pos[:nhalos_with_gal, :] = centres
    halo[:nhalos_with_gal] = central_halos

    # Clean up some more memory
    del cgal

    mask = sgal > 0
    sat_halos = central_halos[np.arange(len(masses))[mask]]

    sgal = sgal[mask]
    centres = centres[mask]
    masses = masses[mask]

    # Now go through each halo and calculate galaxy positions
    begin = nhalos_with_gal
    start = time.time()
    for i, (m, n, ctr) in enumerate(zip(masses, sgal, centres)):
        end = begin + sgal[i]
        pos[begin:end, :] = profile.populate(n, m, ba=1, ca=1, centre=ctr)
        halo[begin:end] = sat_halos[i]
        begin = end

    print "Took ", time.time() - start, " seconds, or ", (time.time() - start)/nhalos_with_gal, " each halo."
    print "NhalosWithGal: ", nhalos_with_gal, "NumGal: ", len(halo), "MeanGal: ", float(
        len(halo))/nhalos_with_gal, "MostGal: ", sgal.max() + 1

    if edges is None:
        pass
    elif np.isscalar(edges):
        edges = np.array([[0, 0, 0], [edges, edges, edges]])
    elif np.array(edges).shape == (2,):
        edges = np.array([[edges[0]]*3, [edges[1]]*3])

    if edges is not None:
        for j in range(3):
            d = pos[:, j] - edges[0][j]
            pos[d < 0, j] = edges[1][j] + d[d < 0]
            d = pos[:, j] - edges[1][j]
            pos[d > 0, j] = edges[0][j] + d[d > 0]

    return pos, halo.astype("int"), nhalos_with_gal
