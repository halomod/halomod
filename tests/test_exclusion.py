"""
Various direct tests of the halo exclusion classes.
"""
from halomod.halo_exclusion import (
    NoExclusion,
    Sphere,
    DblSphere,
    DblEllipsoid,
    NgMatched,
)
import numpy as np
import pytest


def test_no_exclusion():
    m = np.logspace(10, 15, 200)
    integrand = np.outer(np.ones(3), (m / 1e10) ** -2)  # shape (k, m)
    density = np.ones_like(m)
    bias = np.ones_like(m)

    # Solution should be the integral of x^-2 from 10^10 to 10^15 squared
    excl = NoExclusion(
        m=m,
        density=density,
        Ifunc=integrand,
        bias=bias,
        r=None,
        delta_halo=None,
        mean_density=1,
    )

    assert np.allclose(excl.integrate(), 0.9999e20, rtol=1e-4)


def test_spherical_exclusion():
    """Test a simple integrand, and that we get back an analytic result from spherical exclusion."""
    m = np.logspace(10, 15, 1000)
    integrand = np.outer(np.ones(1), (m / 1e10) ** -2)  # shape (k, m)
    density = np.ones_like(m)
    bias = np.ones_like(m)
    delta_h = 200

    r = np.array([100, 1000, 1500])

    mlim = 4 * np.pi * (r / 2) ** 3 * delta_h / 3
    print(mlim)
    analytic = np.clip((1e10 - 1e20 / mlim), a_min=0, a_max=np.inf) ** 2

    excl = Sphere(
        m=m,
        density=density,
        Ifunc=integrand,
        bias=bias,
        r=r,
        delta_halo=delta_h,
        mean_density=1,
    )

    num = excl.integrate()
    print(num.shape)

    assert np.allclose(num.flatten(), analytic, rtol=1e-3)


def test_dbl_sphere():
    """Test simple uniform integral for double-spherical exclusion."""
    m = np.logspace(10, 15, 1000)
    integrand = np.ones((1, len(m)))  # shape (k, m)
    density = np.ones_like(m)
    bias = np.ones_like(m)
    delta_h = 200
    mean_density = 1e11

    r = np.array([1, 100])

    excl = DblSphere(
        m=m,
        density=density,
        Ifunc=integrand,
        bias=bias,
        r=r,
        delta_halo=delta_h,
        mean_density=mean_density,
    )

    # The r = 100 should be equivalent to just using no exclusion.
    no_excl = NoExclusion(
        m=m,
        density=density,
        Ifunc=integrand,
        bias=bias,
        r=r,
        delta_halo=delta_h,
        mean_density=mean_density,
    )

    intg = excl.integrate().flatten()
    assert np.isclose(intg[-1], no_excl.integrate().flatten()[-1], rtol=1e-3)

    # The following tests for r = 1, in which case some masses are masked.
    # The idea is that the integral should just be the area of a portion of a circle
    # out to mlim, where the inside has perpendicular straight edges above m[0].
    # However, this doesn't seem to work for some reason.
    # a = 4 * np.pi * delta_h * mean_density / 3
    # analytic = np.pi*(mlim[0]**2) / 4 - m[0]**2 - 2*(mlim[0] - m[0])*m[0]
    # assert np.isclose(intg[0], analytic, rtol=1e-3)


def test_dbl_ellipsoid_large_r():
    m = np.logspace(10, 15, 200)
    integrand = np.outer(np.ones(3), (m / 1e10) ** -2)  # shape (k, m)
    density = np.ones_like(m)
    bias = np.ones_like(m)

    # Solution should be the integral of x^-2 from 10^10 to 10^15 squared
    no_excl = NoExclusion(
        m=m,
        density=density,
        Ifunc=integrand,
        bias=bias,
        r=np.array([100]),
        delta_halo=200,
        mean_density=1e11,
    )

    excl = DblEllipsoid(
        m=m,
        density=density,
        Ifunc=integrand,
        bias=bias,
        r=np.array([100]),
        delta_halo=200,
        mean_density=1e11,
    )

    assert np.allclose(no_excl.integrate().flatten(), excl.integrate().flatten())


@pytest.mark.skip("Too hard to get the analytic answer")
def test_dbl_ellipsoid_small_r():
    """Test simple uniform integral for double-ellipsoidal exclusion."""
    m = np.logspace(10, 15, 1000)
    integrand = np.ones((1, len(m)))  # shape (k, m)
    density = np.ones_like(m)
    bias = np.ones_like(m)
    delta_h = 200
    mean_density = 1e11

    r = np.array([1])

    excl = DblEllipsoid(
        m=m,
        density=density,
        Ifunc=integrand,
        bias=bias,
        r=r,
        delta_halo=delta_h,
        mean_density=mean_density,
    )

    excl.integrate().flatten()


def test_ng_matched_large_r():
    m = np.logspace(10, 15, 200)
    integrand = np.outer(np.ones(3), (m / 1e10) ** -2)  # shape (k, m)
    density = np.ones_like(m)
    bias = np.ones_like(m)

    # Solution should be the integral of x^-2 from 10^10 to 10^15 squared
    no_excl = NoExclusion(
        m=m,
        density=density,
        Ifunc=integrand,
        bias=bias,
        r=np.array([100]),
        delta_halo=200,
        mean_density=1e11,
    )

    excl = NgMatched(
        m=m,
        density=density,
        Ifunc=integrand,
        bias=bias,
        r=np.array([100]),
        delta_halo=200,
        mean_density=1e11,
    )

    assert np.allclose(no_excl.integrate().flatten(), excl.integrate().flatten())
