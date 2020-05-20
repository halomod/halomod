"""
Provides a simple wrapper for the 2-halo term function, which is written in
fortran (as it is one of the most compute-intensive parts of the calculation).
"""
from twohalo import twohalo_calc as thalo
import numpy as np


def twohalo_wrapper(
    excl,
    sdb,
    m,
    bias,
    ntot,
    dndm,
    lnk,
    dmpower,
    u,
    r,
    dmcorr,
    nbar,
    dhalo,
    rhob,
    ncores,
):
    """
    A simple wrapper for the 2-halo term calculation.

    Parameters
    ----------
    excl : str
        String identifier for halo-exclusion method (None,schneider,sphere,
        ng_matched,ellipsoid)

    sdb : bool
        Whether to use scale-dependent bias or not.

    m : array
        Array of masses

    bias : array
        The scale-independent bias function

    ntot : array
        The total average galaxies per halo of mass m

    dndm : array
        The mass function

    lnk : array
        Logarithmic wavenumbers

    dmpower : array
        Matter power spectrum

    u : (nk,nm)-array
        The normalised fourier transform of the density halo_profile as a function
        of m and k.

    r : array
        The scales at which to evaluate the 2-halo term

    dmcorr : array
        The matter correlation function

    nbar : float
        Mean galaxy number density

    dhalo : float
        Definition of a halo -- overdensity with respect to background.

    rhob : float
        The background density

    ncores : int
        The number of cores to use in the calculation. NOTE: not much of a
        speedup is gained, if any.
    """
    u = np.asfortranarray(u.T)

    exc_type = {"None": 1, "schneider": 2, "sphere": 3, "ng_matched": 4, "ellipsoid": 5}

    corr = thalo.twohalo(
        m,
        bias,
        ntot,
        dndm,
        lnk,
        dmpower,
        u,
        r,
        dmcorr,
        nbar,
        dhalo,
        rhob,
        exc_type[excl],
        sdb,
        ncores,
    )

    return corr


def dblsimps(X, dx, dy):
    return thalo.dblsimps(X, dx, dy)
