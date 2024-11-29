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
