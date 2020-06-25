"""
Created on Sep 9, 2013

@author: Steven
"""
import numpy as np
import scipy.integrate as intg
from scipy.stats import poisson
import time
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from .profiles import Profile
from .hod import HOD

try:
    from pathos import multiprocessing as mp

    HAVE_POOL = True
except ImportError:
    HAVE_POOL = False


def power_to_corr_ogata(
    power: np.ndarray, k: np.ndarray, r: np.ndarray, n=640, h=0.005
):
    """
    Convert a 3D power spectrum to a correlation function.

    Uses Ogata's method (Ogata 2005) for Hankel Transforms in 3D.

    Parameters
    ----------
    power : np.ndarray
        The power spectrum to convert -- either 1D or 2D. If 1D, it is the power as a
        function of k. If 2D, the first dimension should be ``len(r)``, and the power
        to integrate is considered to be different for each ``r``.
    k : np.ndarray
        Array of same length as the last axis of ``power``, giving the fourier-space
        co-ordinates.
    r : np.ndarray
        The real-space co-ordinates to which to transform.
    n : int, optional
        The number of subdivisions in the integral.
    h : int, optional
        Controls the spacing of the intervals (note the intervals are not equispaced).
        Smaller numbers give smaller intervals.

    Notes
    -----
    See the `hankel <https://hankel.readthedocs.io>`_ documentation for details on the
    implementation here. This particular function is restricted to a spherical transform.
    """
    lnk = np.log(k)
    roots = np.arange(1, n + 1)
    t = h * roots
    s = np.pi * np.sinh(t)
    x = np.pi * roots * np.tanh(s / 2)

    dpsi = 1 + np.cosh(s)
    dpsi[dpsi != 0] = (np.pi * t * np.cosh(t) + np.sinh(s)) / dpsi[dpsi != 0]
    sumparts = np.pi * np.sin(x) * dpsi * x

    if power.ndim == 1:
        spl = spline(lnk, power)
        allparts = sumparts * spl(np.log(np.divide.outer(x, r))).T
        return np.sum(allparts, axis=-1) / (2 * np.pi ** 2 * r ** 3)
    else:
        out = np.zeros(len(r))
        for ir, rr in enumerate(r):
            spl = spline(lnk, power[ir, :])
            allparts = sumparts * spl(np.log(x / rr))
            out[ir] = np.sum(allparts) / (2 * np.pi ** 2 * rr ** 3)
        return out


def corr_to_power_ogata(corr, r, k, n=640, h=0.005):
    """
    Convert an isotropic 3D correlation function to a power spectrum.

    Uses Ogata's method (Ogata 2005) for Hankel Transforms in 3D.

    Parameters
    ----------
    corr : np.ndarray
        The correlation function to convert as a function of ``r``.
    r : np.ndarray
        Array of same length as ``corr``, giving the real-space co-ordinates.
    k : np.ndarray
        The fourier-space co-ordinates to which to transform.
    n : int, optional
        The number of subdivisions in the integral.
    h : int, optional
        Controls the spacing of the intervals (note the intervals are not equispaced).
        Smaller numbers give smaller intervals.

    Notes
    -----
    See the `hankel <https://hankel.readthedocs.io>`_ documentation for details on the
    implementation here. This particular function is restricted to a spherical transform.
    """
    return 8 * np.pi ** 3 * power_to_corr_ogata(corr, r, k, n, h)


def power_to_corr(power_func: callable, r: np.ndarray) -> np.ndarray:
    """
    Calculate the isotropic 3D correlation function given an isotropic power spectrum.

    Parameters
    ----------
    power_func : callable
        A callable function which returns the natural log of power given lnk
    r : array_like
        The values of separation/scale to calculate the correlation at.

    Notes
    -----
    This uses standard Simpson's Rule integration, but in which the number of subdivisions
    is chosen with some care to ensure that zeros of the Bessel function are captured.

    See Also
    --------
    power_to_corr_ogata :
        A faster, smarter algorithm for doing the same thing.
    """
    if not np.iterable(r):
        r = [r]

    corr = np.zeros_like(r)

    # the number of steps to fit into a half-period at high-k. 6 is better than 1e-4.
    minsteps = 8

    # set min_k, 1e-6 should be good enough
    mink = 1e-6

    temp_min_k = 1.0

    for i, rr in enumerate(r):
        # getting maxk here is the important part. It must be a half multiple of
        # pi/r to be at a "zero", it must be >1 AND it must have a number of half
        # cycles > 38 (for 1E-5 precision).

        min_k = (2 * np.ceil((temp_min_k * rr / np.pi - 1) / 2) + 0.5) * np.pi / rr
        maxk = max(501.5 * np.pi / rr, min_k)

        # Now we calculate the requisite number of steps to have a good dk at hi-k.
        nk = np.ceil(
            np.log(maxk / mink) / np.log(maxk / (maxk - np.pi / (minsteps * rr)))
        )

        lnk, dlnk = np.linspace(np.log(mink), np.log(maxk), int(nk), retstep=True)
        P = power_func(lnk)
        integ = P * np.exp(lnk) ** 2 * np.sin(np.exp(lnk) * rr) / rr

        corr[i] = (0.5 / np.pi ** 2) * intg.simps(integ, dx=dlnk)

    return corr


def exclusion_window(k: np.ndarray, r: float) -> np.ndarray:
    """Fourier-space top-hat window function.

    Parameters
    ----------
    k : np.ndarray
        The fourier-space wavenumbers
    r : float
        The size of the top-hat window.

    Returns
    -------
    W : np.ndarray
        The top-hat window function in fourier space.
    """
    x = k * r
    return 3 * (np.sin(x) - x * np.cos(x)) / x ** 3


def populate(
    centres: np.ndarray,
    masses: np.ndarray,
    halomodel=None,
    profile: [None, Profile] = None,
    hodmod: [None, HOD] = None,
    edges: [None, np.ndarray] = None,
):
    """
    Populate a series of DM halos with a tracer, given a HOD model.

    Parameters
    ----------
    centres : (N,3)-array
        The cartesian co-ordinates of the centres of the halos
    masses : array_like
        The masses (in M_sun/h) of the halos
    halomodel : type :class:`halomod.HaloModel`
        A HaloModel object. One can either use this, or *both* `halo_profile` and
        `hodmod` arguments.
    profile : type :class:`halo_profile.Profile`, optional
        A density halo_profile to use. Only required if ``halomodel`` not given.
    hodmod : object of type :class:`hod.HOD`, optional
        A HOD model to use to populate the dark matter. Only required if ``halomodel``
        not given.
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
        Number of central galaxies. The first H galaxies in pos/halo correspond to centrals.
    """
    if halomodel is not None:
        profile = halomodel.halo_profile
        hodmod = halomodel.hod

    masses = np.array(masses)

    # Define which halos have central galaxies.
    cgal = np.random.binomial(1, hodmod.central_occupation(masses))
    cmask = cgal > 0
    central_halos = np.arange(len(masses))[cmask]

    if hodmod._central:
        masses = masses[cmask]
        centres = centres[cmask]

    # Calculate the number of satellite galaxies in halos
    # Using ns gives the correct answer for both central condition and not.
    # Note that other parts of the algorithm also need to be changed if central condition
    # is not true.
    sgal = poisson.rvs(hodmod.ns(masses))

    # Get an array ready, hopefully speeds things up a bit
    ncen = np.sum(cgal)
    nsat = np.sum(sgal)

    pos = np.empty((ncen + nsat, 3))
    halo = np.empty(ncen + nsat)

    # Assign central galaxy positions
    halo[:ncen] = central_halos
    if hodmod._central:
        pos[:ncen, :] = centres
    else:
        pos[:ncen, :] = centres[cmask]

    smask = sgal > 0
    # if hodmod._central:
    #     sat_halos = central_halos[np.arange(len(masses[cmask]))[smask]]
    # else:
    if hodmod._central:
        sat_halos = central_halos[np.arange(len(masses))[smask]]
    else:
        sat_halos = np.arange(len(masses))[smask]

    sgal = sgal[smask]
    centres = centres[smask]
    masses = masses[smask]

    # Now go through each halo and calculate galaxy positions
    start = time.time()
    halo[ncen:] = np.repeat(sat_halos, sgal)
    indx = np.concatenate(([0], np.cumsum(sgal))) + ncen

    def fill_array(i):
        m, n, ctr = masses[i], sgal[i], centres[i]
        pos[indx[i] : indx[i + 1], :] = profile.populate(n, m, centre=ctr)

    if HAVE_POOL:
        mp.ProcessingPool(mp.cpu_count()).map(fill_array, list(range(len(masses))))
    else:
        for i in range(len(masses)):
            fill_array(i)

    nhalos_with_gal = len(set(central_halos.tolist() + sat_halos.tolist()))

    print(
        f"Took {time.time() - start} seconds, or "
        f"{(time.time() - start) / nhalos_with_gal} each halo."
    )
    print(
        f"NhalosWithGal: {nhalos_with_gal}, Ncentrals: {ncen}, NumGal: {len(halo)}, "
        f"MeanGal: {float(len(halo)) / nhalos_with_gal}, "
        f"MostGal: {sgal.max() + 1 if len(sgal) > 0 else 1}"
    )

    if edges is not None:
        if np.isscalar(edges):
            edges = np.array([[0, 0, 0], [edges, edges, edges]])
        elif np.array(edges).shape == (2,):
            edges = np.array([[edges[0]] * 3, [edges[1]] * 3])

        for j in range(3):
            d = pos[:, j] - edges[0][j]
            pos[d < 0, j] = edges[1][j] + d[d < 0]
            d = pos[:, j] - edges[1][j]
            pos[d > 0, j] = edges[0][j] + d[d > 0]

    return pos, halo.astype("int"), ncen
