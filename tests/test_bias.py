"""Unit tests of the bias module."""

import pytest
import numpy as np
from hmf import MassFunction
from halomod import bias


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
        h=hmf.cosmo.h,
        Om0=hmf.cosmo.Om0,
        sigma_8=hmf.sigma_8,
        delta_halo=200,
        z=hmf.z,
    )

    assert np.all(np.diff(b.bias()) >= 0)
