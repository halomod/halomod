"""Various direct tests of the halo exclusion classes."""

import numpy as np
import pytest

from halomod import TracerHaloModel
from halomod.halo_exclusion import (
    DblEllipsoid,
    DblEllipsoid_,
    DblSphere,
    DblSphere_,
    Exclusion,
    NgMatched,
    NgMatched_,
    NoExclusion,
    Sphere,
    cumsimps,
    dblsimps_,
    dbltrapz_,
    integrate_dblsphere,
    makeW,
)


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
        m=m, density=density, power_integrand=integrand, bias=bias, r=None, halo_density=None
    )

    assert np.allclose(excl.integrate(), 0.9999e20, rtol=1e-4)


@pytest.mark.parametrize("integ", [dblsimps_, dbltrapz_])
def test_dbl_simps_(integ):
    """Test a simple integration."""
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
    analytic = np.clip((1e10 - 1e20 / mlim), a_min=0, a_max=np.inf) ** 2

    excl = Sphere(
        m=m, density=density, power_integrand=integrand, bias=bias, r=r, halo_density=200.0
    )

    num = excl.integrate()

    assert np.allclose(num.flatten(), analytic, rtol=1e-3)


@pytest.mark.parametrize(
    "excl", [NgMatched, NgMatched_, DblEllipsoid, DblEllipsoid_, DblSphere, DblSphere_, Sphere]
)
@pytest.mark.parametrize("z", (0, 1))
def test_halo_exclusion_extreme_r(excl: Exclusion, z: float):
    kw = {
        "z": z,
        "rmin": 1e-2,
        "rmax": 300.0,
        "rnum": 15,
        "dr_table": 0.05,
        "dlog10m": 0.15,
        "dlnk": 0.05,
        "hod_model": "Zheng05",
        "hod_params": {"central": True},
        "bias_model": "Tinker10PBSplit",
        "hmf_model": "Tinker10",
        "tracer_concentration_model": "Duffy08",
        "halo_profile_model": "NFW",
        "hc_spectrum": "linear",
        "force_unity_dm_bias": False,
        "transfer_params": {"extrapolate_with_eh": True},
    }
    noexc = TracerHaloModel(exclusion_model="NoExclusion", **kw)
    with_exc = TracerHaloModel(exclusion_model=excl, **kw)

    mexc = with_exc._matter_exclusion
    texc = with_exc._tracer_exclusion

    # Restrict ourselves to large scales, above which all the halo masses have
    # smaller radii.
    rmax = mexc.r_halo.max() * 2
    rmask = mexc.r > rmax

    # At high-r, none of the masses should be masked out.
    if mexc.mask is not None:
        assert not np.any(mexc.mask[rmask])
        assert not np.any(texc.mask[rmask])

    # We should also get that the density_mod is unity at large scales.
    if hasattr(mexc.density_mod, "__len__"):
        # density_mod can be simply just "1" because some exclusions don't use it.
        np.testing.assert_allclose(mexc.density_mod[rmask], 1, atol=1e-4)
        np.testing.assert_allclose(texc.density_mod[rmask], 1, atol=1e-4)

    # Finally, the integrated power should be equal to the effective bias at large
    # scales.
    intg = mexc.integrate()
    intg = intg[rmask, 0] if intg.shape[0] == len(rmask) else intg[0, 0]

    np.testing.assert_allclose(intg, with_exc.bias_effective_matter**2, rtol=1e-3)

    intg = texc.integrate()
    intg = intg[rmask, 0] if intg.shape[0] == len(rmask) else intg[0, 0]
    np.testing.assert_allclose(
        texc.integrate()[rmask, 0], with_exc.bias_effective_tracer**2, rtol=1e-3
    )

    # the matter and tracer correlation functions should be unaffected
    # by halo-exclusion on the largest scales.
    mask = noexc.r > rmax
    np.testing.assert_allclose(
        noexc.corr_2h_auto_matter[mask],
        with_exc.corr_2h_auto_matter[mask],
        rtol=3e-4,
    )
    np.testing.assert_allclose(
        noexc.corr_2h_auto_tracer[mask], with_exc.corr_2h_auto_tracer[mask], rtol=3e-4
    )

    # the matter and tracer correlation functions should be zero on the smallest scales.
    # by halo-exclusion on the largest scales.
    mask = noexc.r < 0.1
    assert np.all(with_exc.corr_2h_auto_matter[mask] < 0.2 * noexc.corr_2h_auto_matter[mask])
    assert np.all(with_exc.corr_2h_auto_tracer[mask] < 0.2 * noexc.corr_2h_auto_tracer[mask])


# ============================================================================
# Tests for smooth xmin boundary (spline integration) in exclusion models
# ============================================================================


def _make_excl_fixtures(n_k=3, n_mass=200, n_r=3):
    """Return shared arrays used by the xmin/2D-bias tests."""
    m = np.logspace(10, 15, n_mass)
    integrand = np.outer(np.ones(n_k), (m / 1e10) ** -2)  # (k, m)
    density = np.ones_like(m)
    r = np.array([100.0, 1000.0, 1500.0])[:n_r]
    bias_1d = np.ones_like(m)
    bias_2d = np.outer(np.ones_like(r), bias_1d)  # (r, m)
    return m, integrand, density, r, bias_1d, bias_2d


def test_no_exclusion_spline_path_1d_bias():
    """Setting xmin exercises _spline_integrate for 1D bias in NoExclusion."""
    m, integrand, density, r, bias_1d, _ = _make_excl_fixtures()
    xmin = m[20]  # well above m[0] so the spline path is taken

    excl_full = NoExclusion(
        m=m, density=density, power_integrand=integrand, bias=bias_1d, r=r, halo_density=None
    )
    excl_xmin = NoExclusion(
        m=m,
        density=density,
        power_integrand=integrand,
        bias=bias_1d,
        r=r,
        halo_density=None,
        xmin=xmin,
    )

    result_full = excl_full.integrate()
    result_xmin = excl_xmin.integrate()

    # Both should be (1, k) because bias is 1D.
    assert result_full.shape[0] == 1
    assert result_xmin.shape[0] == 1
    # Cutting off the lower-mass tail must strictly reduce the integral.
    assert np.all(result_xmin < result_full)


def test_no_exclusion_2d_bias():
    """NoExclusion.integrate() must work with a 2D bias array (scale-dependent bias).

    This was broken before the fix: the 2D path accidentally passed an unknown
    ``dx`` keyword argument to ``_spline_integrate``, raising a TypeError.
    """
    m, integrand, density, r, bias_1d, bias_2d = _make_excl_fixtures()

    excl_1d = NoExclusion(
        m=m, density=density, power_integrand=integrand, bias=bias_1d, r=r, halo_density=None
    )
    excl_2d = NoExclusion(
        m=m, density=density, power_integrand=integrand, bias=bias_2d, r=r, halo_density=None
    )

    result_1d = excl_1d.integrate()
    result_2d = excl_2d.integrate()

    # 2D bias returns shape (r, k); 1D bias returns (1, k).
    assert result_2d.shape == (len(r), integrand.shape[0])
    # For uniform bias=1, all r slices of the 2D result equal the 1D result.
    np.testing.assert_allclose(result_2d, np.broadcast_to(result_1d, result_2d.shape), rtol=1e-6)


def test_no_exclusion_2d_bias_with_xmin():
    """NoExclusion with 2D bias and xmin must give the same result as 1D bias + xmin."""
    m, integrand, density, r, bias_1d, bias_2d = _make_excl_fixtures()
    xmin = m[20]

    excl_1d = NoExclusion(
        m=m,
        density=density,
        power_integrand=integrand,
        bias=bias_1d,
        r=r,
        halo_density=None,
        xmin=xmin,
    )
    excl_2d = NoExclusion(
        m=m,
        density=density,
        power_integrand=integrand,
        bias=bias_2d,
        r=r,
        halo_density=None,
        xmin=xmin,
    )

    result_2d = excl_2d.integrate()
    result_1d = excl_1d.integrate()
    np.testing.assert_allclose(result_2d, np.broadcast_to(result_1d, result_2d.shape), rtol=1e-6)


def test_integrate_dblsphere_xmin():
    """integrate_dblsphere with xmin uses the spline outer integral."""
    m = np.logspace(10, 15, 100)
    r = np.array([100.0, 1000.0])
    nk = 2
    integ_arr = np.broadcast_to((m / 1e10) ** -2, (len(r), nk, len(m))).copy()
    mask = np.zeros((len(r), len(m), len(m)), dtype=bool)
    dx = np.log(m[1] / m[0])
    xmin = m[10]

    result_full = integrate_dblsphere(integ_arr, mask, dx)
    result_xmin = integrate_dblsphere(integ_arr, mask, dx, m=m, xmin=xmin)

    # Both should return (r, k).
    assert result_full.shape == (len(r), nk)
    assert result_xmin.shape == (len(r), nk)
    # Cutting mass below xmin must reduce the integral.
    assert np.all(result_xmin <= result_full)


def test_dblellipsoid_density_mod_xmin():
    """DblEllipsoid.density_mod uses spline integration when xmin is set."""
    m = np.logspace(10, 15, 100)
    r = np.array([100.0, 1000.0, 1500.0])
    nk = 2
    integrand = np.outer(np.ones(nk), (m / 1e10) ** -2)
    density = np.ones_like(m)
    bias = np.ones_like(m)
    halo_density = 200.0
    xmin = m[10]

    excl_full = DblEllipsoid(
        m=m, density=density, power_integrand=integrand, bias=bias, r=r, halo_density=halo_density
    )
    excl_xmin = DblEllipsoid(
        m=m,
        density=density,
        power_integrand=integrand,
        bias=bias,
        r=r,
        halo_density=halo_density,
        xmin=xmin,
    )

    dm_full = excl_full.density_mod
    dm_xmin = excl_xmin.density_mod

    # density_mod should be a 1D vector over r.
    assert dm_full.shape == (len(r),)
    assert dm_xmin.shape == (len(r),)
    # Cutting lower masses must reduce the modified density.
    assert np.all(dm_xmin <= dm_full)
