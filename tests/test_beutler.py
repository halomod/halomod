"""
Direct tests of the halo model code against known values from Beutler+2013, with
intermediate data provided by David Palomara using his own halo model code.
"""

from halomod.integrate_corr import ProjectedCF
import numpy as np
import pytest
from pathlib import Path

pytestmark = pytest.mark.skip(
    "These tests are not passing and I don't have access to the data source."
)

direc = Path(__file__).parent / "data/beutler"
david_wprp = np.genfromtxt(direc / "wp_dump.txt")
david_xi = np.genfromtxt(direc / "xir_dump.txt")

# These are values provided directly by David Palomara
david_nden = 3.673788e-04
david_fsat = 0.150
david_beff = 1.488


@pytest.fixture(scope="module")
def h() -> ProjectedCF:
    return ProjectedCF(
        sd_bias_model="TinkerSD05",
        exclusion_model="NgMatched",
        rp_min=david_wprp[0, 0],
        rp_max=david_wprp[-1, 0],
        rp_num=len(david_wprp[:, 0]),
        rp_log=True,
        rmin=david_xi[0, 0],
        rmax=david_xi[-1, 0],
        rnum=len(david_xi[:, 0]),
        rlog=True,
        lnk_min=-15,
        lnk_max=10,
        dlnk=0.01,
        cosmo_params={"Ob0": 0.04545, "Om0": 0.2732, "H0": 70.0},
        n=0.966,
        sigma_8=0.8,
        z=0.0585,
        hc_spectrum="nonlinear",
        proj_limit=50.0,
        transfer_model="EH",
        dlog10m=0.01,
        halo_concentration_model="Duffy08",
        hmf_model="Tinker10",
        bias_model="Tinker10",
        hod_model="Zheng05",
        hod_params={
            "alpha": 1.396,
            "M_1": 14.022,
            "M_min": 12.6753,
            "sig_logm": 0.001,
            "M_0": 0,
        },
        takahashi=False,
    )


def test_mean_gal_den(h: ProjectedCF):
    assert np.isclose(h.mean_tracer_den / h.cosmo.h ** 3, david_nden, rtol=0.1)


def test_mean_mass_eff(h: ProjectedCF):
    assert np.isclose(h.bias_effective_tracer, david_beff, rtol=0.05)


def test_sat_frac(h: ProjectedCF):
    assert np.isclose(h.satellite_fraction, david_fsat, rtol=0.1)


@pytest.mark.parametrize(
    "q,thing,indx",
    [
        ("projected_corr_gal", david_wprp, 1),
        ("corr_1h_cs_auto_tracer", david_xi, 1),
        ("corr_1h_ss_auto_tracer", david_xi, 2),
        ("corr_1h_auto_tracer", david_xi, 3),
        ("corr_2h_auto_tracer", david_xi, 4),
        ("corr_auto_tracer", david_xi, 5),
    ],
)
def test_corr(h, q, thing, indx):
    assert np.allclose(getattr(h, q), thing[:, indx], rtol=0.1)
