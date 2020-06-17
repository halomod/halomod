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
    return MassFunction()


@pytest.mark.parametrize("bias_model", list(bias.Bias._models.values()))
def test_monotonic_bias(bias_model, hmf: MassFunction):
    if bias_model.__name__ in ["Jing98", "Seljak04", "Seljak04Cosmo"]:
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

    assert np.all(np.diff(b.bias()) >= 0)


def test_bias_against_colossus():
    cole89 = bias.make_colossus_bias("cole89", mdef=SOMean())

    hm = DMHaloModel(transfer_model="EH", mdef_model=SOMean, bias_model=bias.Mo96)
    col = DMHaloModel(transfer_model="EH", mdef_model=SOMean, bias_model=cole89)

    assert np.allclose(hm.halo_bias, col.halo_bias, rtol=1e-2)
