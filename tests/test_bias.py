"""Unit tests of the bias module."""

import pytest
import numpy as np
from hmf import MassFunction
from halomod import bias
from hmf.halos.mass_definitions import SOMean
from halomod import DMHaloModel


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


@pytest.mark.parametrize(
    "hmf_bias,col_bias",
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
    if col_bias in ["seljak04", "jing98"]:
        pytest.skip("Uses nonlinear mass which has to be investigated.")

    cbias = bias.make_colossus_bias(col_bias, mdef=SOMean())

    hm = DMHaloModel(transfer_model="EH", mdef_model=SOMean, bias_model=hmf_bias)
    col = DMHaloModel(transfer_model="EH", mdef_model=SOMean, bias_model=cbias)

    assert np.allclose(hm.halo_bias, col.halo_bias, rtol=1e-2)
