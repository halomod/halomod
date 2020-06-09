"""
Test various halo profile properties.
"""
from halomod.concentration import Bullock01Power
from halomod import profiles as pf
import pytest
import numpy as np

bullock = Bullock01Power(ms=1e12)
m = np.logspace(10, 15, 100)
r = np.logspace(-2, 2, 20)


@pytest.mark.parametrize(
    "profile",
    (
        pf.NFW,
        pf.NFWInf,
        pf.CoredNFW,
        pf.Einasto,
        pf.GeneralizedNFW,
        pf.GeneralizedNFWInf,
        pf.Hernquist,
        pf.Moore,
        pf.MooreInf,
    ),
)
def test_decreasing_profile(profile):
    prof = profile(bullock)
    assert np.all(np.diff(prof.rho(r, m=1e12, norm="rho")) <= 0)


@pytest.mark.parametrize(
    "profile",
    (
        pf.NFW,
        pf.NFWInf,
        pf.CoredNFW,
        pf.Einasto,
        pf.GeneralizedNFW,
        pf.GeneralizedNFWInf,
        pf.Hernquist,
        pf.Moore,
        pf.MooreInf,
    ),
)
def test_increasing_cdf(profile):
    prof = profile(bullock)
    assert np.all(np.diff(prof.cdf(r, m=1e12)) >= 0)


@pytest.mark.parametrize(
    "profile",
    (
        pf.NFW,
        pf.NFWInf,
        pf.CoredNFW,
        pf.Einasto,
        pf.GeneralizedNFW,
        pf.GeneralizedNFWInf,
        pf.Hernquist,
        pf.Moore,
        pf.MooreInf,
    ),
)
def test_decreasing_convolution(profile):
    prof = profile(bullock)
    if not prof.has_lam:
        pytest.skip("This profile doesn't have a convolution defined.")
    else:
        assert np.all(np.diff(prof.lam(r, m=1e12)) <= 0)
