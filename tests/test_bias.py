"""Unit tests of the bias module."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from hmf import MassFunction
from hmf.halos.mass_definitions import SOMean

from halomod import DMHaloModel, TracerHaloModel, bias


@pytest.fixture(scope="module")
def hmf():
    """Simple hmf object that gives us reasonable defaults for the bias."""
    return MassFunction(transfer_model="EH")


def test_Jing98(hmf: MassFunction):
    """Just to see if it works."""
    b = bias.Jing98(
        nu=hmf.nu2,
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
        nu=hmf.nu2,
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
        nu=hmf.nu2,
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


def test_halo_model_bias_no_nu_deprecation():
    """Regression test for #263: computing the bias must not touch deprecated ``nu``.

    hmf>=3.7 deprecates ``MassFunction.nu`` (which is the *squared* peak height) in
    favour of ``nu2``. The bias models must receive ``nu2`` so that their numbers are
    unchanged, without triggering the deprecation warning.
    """
    hm = TracerHaloModel(transfer_model="EH")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        halo_bias = hm.halo_bias
        power = hm.power_auto_tracer
        bias_nu = hm.bias.nu

    assert np.all(np.isfinite(halo_bias))
    assert np.all(np.isfinite(power))

    # The bias models are written in terms of the squared peak height.
    np.testing.assert_array_equal(bias_nu, hm.nu2)
    np.testing.assert_allclose(bias_nu, (hm.delta_c / hm.sigma) ** 2, rtol=1e-12, atol=0)
    np.testing.assert_allclose(bias_nu, hm.peak_height**2, rtol=1e-12, atol=0)
