"""
Test a specific model against data provided by Chris Blake from his own halo modelling
code, used in Blake+08 (modelling of SDSS sources).
"""
import numpy as np
import pytest
from halomod import TracerHaloModel
from halomod.hod import Zehavi05
from scipy.interpolate import InterpolatedUnivariateSpline as spline


@pytest.fixture(scope="module")
def hod():
    return TracerHaloModel(
        dlog10m=0.02,
        lnk_min=np.log(1e-8),
        lnk_max=np.log(20000),
        dlnk=0.01,
        cosmo_params={"Ob0": 0.04545, "Om0": 0.2732, "H0": 70.0},
        n=0.966,
        sigma_8=0.8,
        z=0.0369,
        hod_model=Zehavi05,
        hod_params={"alpha": 1.214, "M_1": 13.396, "M_min": 12.0478},
        hc_spectrum="nonlinear",
        hmf_model="Jenkins",
        bias_model="Tinker05",
        sd_bias_model="TinkerSD05",
        exclusion_model="Sphere",
        halo_concentration_model="Duffy08",
        takahashi=0,
        transfer_model="EH",
    )


# ===============================================================================
# Iterate through quantities
# ===============================================================================
@pytest.mark.parametrize("q", ["linearpk", "nonlinpk", "m_vs_nu", "biasfn", "massfn",])
def test_blake_quantity(hod, datadir, q):
    if not q.startswith("xir"):
        chris = np.genfromtxt(datadir / "blake" / (q + ".txt"))

    if q == "linearpk":
        steve = spline(hod.k, hod.power)(chris[:, 0])
    if q == "nonlinpk":
        steve = spline(hod.k, hod.nonlinear_power)(chris[:, 0])
    if q == "m_vs_nu":
        steve = spline(hod.m, hod.nu)(chris[:, 0])
    if q == "biasfn":
        steve = spline(hod.m, hod.halo_bias)(chris[:, 0])
    if q == "massfn":
        chris[:, 0] = 10 ** chris[:, 0]
        steve = spline(hod.m, hod.dndlog10m)(chris[:, 0])

    assert np.allclose(steve, chris[:, 1], rtol=0.1)
