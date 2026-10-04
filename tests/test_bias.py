"""Unit tests of the bias module."""

from __future__ import annotations

import numpy as np
import pytest
from hmf import MassFunction
from hmf.halos.mass_definitions import SOMean

from halomod import DMHaloModel, bias


@pytest.fixture(scope="module")
def hmf():
    """Simple hmf object that gives us reasonable defaults for the bias."""
    return MassFunction(transfer_model="EH")


def test_Jing98(hmf: MassFunction):
    """Just to see if it works."""
    b = bias.Jing98(
        nu=hmf.nu,
        n=hmf.n,
        n_eff=2.89,
        delta_c=hmf.delta_c,
        m=hmf.m,
        mstar=hmf.mass_nonlinear,
        cosmo=hmf.cosmo,
        sigma_8=hmf.sigma_8,
        delta_halo=200,
        z=hmf.z,
    )
    assert isinstance(b.bias(), np.ndarray)


def test_PBSplit(hmf: MassFunction):
    """Test if interpolation of parameters works."""
    b = bias.Tinker10PBSplit(
        nu=hmf.nu,
        n=hmf.n,
        delta_c=hmf.delta_c,
        m=hmf.m,
        mstar=hmf.mass_nonlinear,
        cosmo=hmf.cosmo,
        sigma_8=hmf.sigma_8,
        delta_halo=250,
        z=hmf.z,
    )
    assert isinstance(b.bias(), np.ndarray)


@pytest.mark.parametrize("bias_model", list(bias.Bias._models.values()))
def test_monotonic_bias(bias_model, hmf: MassFunction):
    if bias_model.__name__ in ["Jing98", "Seljak04"]:
        pytest.skip("Known to be non-monotonic.")

    # Test that all bias models are monotonic
    b = bias_model(
        nu=hmf.nu,
        n=hmf.n,
        delta_c=hmf.delta_c,
        m=hmf.m,
        mstar=hmf.mass_nonlinear,
        cosmo=hmf.cosmo,
        sigma_8=hmf.sigma_8,
        delta_halo=200,
        z=hmf.z,
    )
    dlog10m = np.log10(b.m[1] / b.m[0])
    diff = np.diff(b.bias() / dlog10m)
    print(diff.min())
    assert diff.min() >= -2e-2


@pytest.mark.filterwarnings("ignore:Your input mass definition")
@pytest.mark.filterwarnings("ignore:Astropy cosmology class contains massive neutrinos")
@pytest.mark.parametrize(
    ("hmf_bias", "col_bias"),
    [
        (bias.Mo96, "cole89"),
        (bias.Jing98, "jing98"),
        (bias.SMT01, "sheth01"),
        (bias.Seljak04, "seljak04"),
        (bias.Pillepich10, "pillepich10"),
        (bias.Tinker10, "tinker10"),
    ],
)
def test_bias_against_colossus(hmf_bias, col_bias):
    # don't care that the mdef isn't compatible with the HMF, because we're not testing
    # the HMF.
    if col_bias in ["seljak04", "jing98"]:
        pytest.skip("Uses nonlinear mass which has to be investigated.")

    cbias = bias.make_colossus_bias(col_bias, mdef=SOMean())

    hm = DMHaloModel(transfer_model="EH", mdef_model=SOMean, bias_model=hmf_bias)
    col = DMHaloModel(transfer_model="EH", mdef_model=SOMean, bias_model=cbias)

    assert np.allclose(hm.halo_bias, col.halo_bias, rtol=1e-2)


@pytest.mark.parametrize("bias_model", ["Tinker10", "SMT01", "Mo96", "Jing98"])
def test_high_z_bias_without_nonlinear_mass(bias_model):
    """Bias models that don't use M* work at z=20, where M* is undefined (#265).

    At z=20 every halo on the mass grid is a rare peak, so its bias exceeds unity, and
    on large scales the halo-model matter power reduces to the linear power.
    """
    hm = DMHaloModel(z=20, transfer_model="EH", bias_model=bias_model)
    with pytest.raises(ValueError, match="Cannot find the nonlinear mass"):
        _ = hm.mass_nonlinear

    assert hm.bias.mstar is None
    assert np.all(np.isfinite(hm.halo_bias))
    assert np.all(hm.halo_bias > 1)

    p = hm.power_auto_matter
    assert np.all(np.isfinite(p))
    k = 0.01
    assert np.interp(k, hm.k_hm, p) == pytest.approx(np.interp(k, hm.k, hm.power), rel=1e-2)


@pytest.mark.parametrize("bias_model", ["Seljak04", "Seljak04Cosmo"])
def test_seljak04_requires_nonlinear_mass(bias_model):
    """Seljak04 is defined in terms of M*, so it must raise when M* is undefined."""
    hm = DMHaloModel(z=20, transfer_model="EH", bias_model=bias_model)
    with pytest.raises(ValueError, match="Cannot find the nonlinear mass"):
        _ = hm.halo_bias

    hm0 = DMHaloModel(z=0, transfer_model="EH", bias_model=bias_model)
    assert hm0.bias.mstar == hm0.mass_nonlinear
    assert np.all(np.isfinite(hm0.halo_bias))


def test_seljak04_at_nonlinear_mass():
    """At m = M*, Seljak04 reduces to a + b + d/(e+1) + f (constructor still takes mstar)."""
    b = bias.Seljak04(nu=np.array([1.0]), m=np.array([1e12]), mstar=1e12)
    p = b.params
    expected = p["a"] + p["b"] + p["d"] / (p["e"] + 1) + p["f"]
    assert b.bias()[0] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("bias_model", "expected"),
    [
        # Recorded on hmf 3.7.1 before #265 changed how mstar is passed to bias models.
        ("Seljak04", (0.6691316610205197, 1.1026721589147641, 6.820381127433977)),
        ("Seljak04Cosmo", (0.7108376610205197, 1.08743148698864, 6.721728455507853)),
    ],
)
def test_seljak04_z0_unchanged(bias_model, expected):
    """Only computing M* for models that need it doesn't change Seljak04 at z=0."""
    hm = DMHaloModel(z=0, transfer_model="EH", bias_model=bias_model)
    b = np.interp([11.0, 13.0, 15.0], np.log10(hm.m), hm.halo_bias)
    np.testing.assert_allclose(b, expected, rtol=1e-6)


@pytest.mark.parametrize("bias_model", ["Seljak04", "Jing98"])
def test_required_inputs_update_with_z(bias_model):
    """Inputs only computed for some bias models still invalidate on update."""
    hm = DMHaloModel(z=0, transfer_model="EH", bias_model=bias_model)
    b0 = hm.halo_bias
    hm.update(z=1.0)

    fresh = DMHaloModel(z=1.0, transfer_model="EH", bias_model=bias_model)
    np.testing.assert_allclose(hm.halo_bias, fresh.halo_bias, rtol=1e-10)
    # Massive haloes of fixed mass are rarer, hence more biased, at higher z.
    massive = hm.m >= 1e12
    assert np.all(hm.halo_bias[massive] > b0[massive])
    if bias_model == "Seljak04":
        assert hm.bias.mstar == pytest.approx(fresh.mass_nonlinear, rel=1e-10)
        assert hm.bias.mstar < 0.1 * DMHaloModel(z=0, transfer_model="EH").mass_nonlinear
    else:
        np.testing.assert_allclose(hm.bias.n_eff, fresh.n_eff, rtol=1e-10)
