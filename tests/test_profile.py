"""
Test various halo profile properties.
"""
from halomod.concentration import Bullock01Power
from halomod import profiles as pf
from halomod import TracerHaloModel
import pytest
import numpy as np

bullock = Bullock01Power(ms=1e12)
m = np.logspace(10, 15, 100)
r = np.logspace(-2, 2, 20)


class NFWnum(pf.Profile):
    """Test the numerical integration against analytical."""

    def _f(self, x):
        return 1.0 / (x * (1 + x) ** 2)


class NFWnumInf(pf.ProfileInf):
    """Test the numerical integration against analytical."""

    def _f(self, x):
        return 1.0 / (x * (1 + x) ** 2)


@pytest.fixture(scope="module")
def thm():
    return TracerHaloModel(rmin=0.01, rmax=50, rnum=20)


@pytest.fixture(scope="module")
def thmnum():
    return TracerHaloModel(rmin=0.01, rmax=50, rnum=20, halo_profile_model=NFWnum)


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
        pf.PowerLawWithExpCut,
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
        pf.PowerLawWithExpCut,
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
        # pf.PowerLawWithExpCut,
    ),
)
def test_decreasing_convolution(profile):
    prof = profile(bullock)
    if not prof.has_lam:
        pytest.skip("This profile doesn't have a convolution defined.")
    else:
        assert np.all(np.diff(prof.lam(r, m=1e12)) <= 0)


@pytest.mark.parametrize(
    "profile",
    (
        pf.NFW,
        #        pf.NFWInf,  infinite profile can't be normalised by mass.
        pf.CoredNFW,
        pf.Einasto,
        pf.GeneralizedNFW,
        #        pf.GeneralizedNFWInf,
        pf.Hernquist,
        pf.Moore,
        #        pf.MooreInf,
    ),
)
def test_ukm_low_k(profile):
    """Test that all fourier transforms, when normalised by mass, are 1 at low k"""
    k = np.array([1e-10])
    m = np.logspace(10, 18, 100)

    prof = profile(bullock)

    assert np.allclose(prof.u(k, m, norm="m"), 1, rtol=1e-3)


def test_virial_velocity(thm, thmnum):
    m = np.array([1e12, 1e13])
    r = np.array([0.1, 1.0])
    attnum = getattr(thmnum.halo_profile, "virial_velocity")
    att = getattr(thm.halo_profile, "virial_velocity")
    assert np.allclose(att(m), attnum(m), rtol=1e-2)
    assert np.allclose(att(None, r), attnum(None, r), rtol=1e-2)
    with pytest.raises(ValueError):
        attnum()


def test_h(thm, thmnum):
    m = np.array([1e12, 1e13])
    attnum = getattr(thmnum.halo_profile, "_h")
    att = getattr(thm.halo_profile, "_h")
    cm = getattr(thm.halo_profile, "cm_relation")
    assert np.allclose(att(cm(m)), attnum(m=m), rtol=1e-2)
    with pytest.raises(ValueError):
        attnum()


def test_u(thm, thmnum):
    k = np.array([0.1, 1])
    m = np.array([1e12, 1e13])
    attnum = getattr(thmnum.halo_profile, "u")
    att = getattr(thm.halo_profile, "u")
    assert np.allclose(att(k, m, norm=None), attnum(k, m, norm=None), rtol=1e-2)
    with pytest.raises(ValueError):
        attnum(k, m, norm="r")


def test_lam(thmnum, thm):
    r = np.array([0.1, 1])
    m = np.array([1e12, 1e13])
    attnum = getattr(thmnum.halo_profile, "lam")
    att = getattr(thm.halo_profile, "lam")
    with pytest.raises(AttributeError):
        attnum(r, m, norm="r")
    with pytest.raises(ValueError):
        att(r, m, norm="r")


def test_get_r_variable(thmnum):
    r = np.array([0.1, 1])
    m = np.array([1e12, 1e13])
    attnum = getattr(thmnum.halo_profile, "_get_r_variables")
    with pytest.raises(ValueError):
        attnum(r, m, coord="m")


def test_get_k_variable(thmnum, thm):
    k = np.array([0.1, 1])
    m = np.array([1e12, 1e13])
    attnum = getattr(thmnum.halo_profile, "_get_k_variables")
    att = getattr(thm.halo_profile, "_get_k_variables")
    assert np.allclose(att(k, m, coord="kappa"), attnum(k, m, coord="kappa"), rtol=1e-2)
