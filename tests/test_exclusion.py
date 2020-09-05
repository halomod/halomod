"""
Various direct tests of the halo exclusion classes.
"""
from halomod.halo_exclusion import (
    makeW,
    outer,
    dblsimps_,
    dbltrapz,
    dbltrapz_,
    NoExclusion,
    Sphere,
    DblSphere,
    DblSphere_,
    DblEllipsoid,
    DblEllipsoid_,
    NgMatched,
    NgMatched_,
    cumsimps,
)
import numpy as np
import pytest
from scipy.integrate import simps


def test_makeW():
    arr = np.array([[1.0, 4.0, 1.0], [4.0, 16.0, 4.0], [1.0, 4.0, 1.0]])
    assert np.all(makeW(3, 3) == arr)


def test_cumsimps():
    y = np.array([1, 2, 3])
    assert isinstance(cumsimps(y, 0.001), np.ndarray)


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


@pytest.mark.parametrize("integ", (dblsimps_, dbltrapz_))
def test_dbl_simps_(integ):
    """Test a simple integration"""
    arr1 = np.outer(np.arange(7).astype("float64"), np.arange(7).astype("float64"))
    arr2 = np.outer(np.arange(8).astype("float64"), np.arange(8).astype("float64"))
    num1 = integ(arr1, dx=1, dy=1)
    num2 = integ(arr2, dx=1, dy=1)

    assert np.allclose(num1, 324, rtol=1e-4)
    assert np.allclose(num2, 600.25, rtol=1e-4)


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


@pytest.mark.parametrize("dbl_sphere", (DblSphere, DblSphere_))
def test_dbl_sphere(dbl_sphere):
    """Test simple uniform integral for double-spherical exclusion."""
    m = np.logspace(10, 15, 1001)
    integrand = np.ones((1, len(m)))  # shape (k, m)
    density = np.ones_like(m)
    bias = np.ones_like(m)
    delta_h = 200
    mean_density = 1e11

    r = np.array([1, 100])

    excl = dbl_sphere(
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

    den = np.sqrt(
        simps(
            simps(
                np.outer(density * m, np.ones_like(density)), dx=excl.dlnx, even="first"
            ),
            dx=excl.dlnx,
            even="first",
        )
    )

    assert np.isclose(intg[-1], no_excl.integrate().flatten()[-1], rtol=1e-3)
    assert np.allclose(den, excl.density_mod[-1])


@pytest.mark.parametrize("dbl_ellipsoid", (DblEllipsoid, DblEllipsoid_))
def test_dbl_ellipsoid_large_r(dbl_ellipsoid):
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

    excl = dbl_ellipsoid(
        m=m,
        density=density,
        Ifunc=integrand,
        bias=bias,
        r=np.array([100]),
        delta_halo=200,
        mean_density=1e11,
    )
    den = np.sqrt(
        dbltrapz(
            outer(np.ones_like(np.array([100])), np.outer(density * m, density * m)),
            excl.dlnx,
        )
    )

    assert np.allclose(
        no_excl.integrate().flatten(), excl.integrate().flatten(), rtol=1e-3
    )
    assert np.allclose(den, excl.density_mod, rtol=1e-3)


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


@pytest.mark.parametrize("ng_matched", (NgMatched, NgMatched_))
def test_ng_matched_large_r(ng_matched):
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
