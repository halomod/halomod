"""
This module tests halomod against hmcode as run in a particular configuration.

We explicitly install and run HMCode as part of the test setup to make it clear what
parameters are being used etc.

See devel/HMCode.ipynb for a worded work-through of these automatic tests (may be helpful
if these tests break). Note that in the notebook, a forked version of the HMCode repo is
used (https://github.com/steven-murray/hmcode) which has a branch in which more
information is written out.
"""

import pytest
import numpy as np
from halomod import DMHaloModel
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from pathlib import Path


def read_power(fname: Path):
    """Read the power.dat file from HMcode"""
    # Each column is the power at a different redshift.
    with open(fname, "r") as fl:
        line = fl.readline().split("#####")[-1].split("        ")[1:]
        redshifts = [float(x) for x in line]

    data = np.genfromtxt(fname, skip_header=1)
    k = data[:, 0]
    return k, redshifts, data[:, 1:]


@pytest.fixture(scope="module")
def hmcode_data(datadir):
    k, z, data = read_power(datadir / "hmcode_power.dat")
    return {"k": k, "z": z, "p": data}


hm = DMHaloModel(
    exclusion_model=None,
    sd_bias_model=None,
    transfer_model="EH_BAO",
    cosmo_params={
        "Tcmb0": 2.725,  # Line 596
        "Om0": 0.3,  # Line 587
        "Ob0": 0.05,  # Line 589
        "H0": 70.0,  # Line 591
    },
    hc_spectrum="linear",
    halo_concentration_model="Bullock01",
    halo_concentration_params={"K": 4, "F": 0.01},  # Line 376
    hmf_model="SMT",
    sigma_8=0.8,  # Line 593
    n=0.96,  # Line 594
    Mmin=2,  # Line 795
    Mmax=18,  # Line 796,
    lnk_min=np.log(1e-3),  # Line 50
    lnk_max=np.log(1e2),  # Line 51
    dlnk=0.01,
    dlog10m=16 / 256,
    mdef_model="SOMean",
    disable_mass_conversion=True,
)


@pytest.mark.parametrize("iz", range(16))
def test_hmcode(hmcode_data, iz):
    z = hmcode_data["z"][iz]

    hm.update(z=z)

    halomod = (
        hm.power_auto_matter_fnc(hmcode_data["k"])
        * hmcode_data["k"] ** 3
        / (2 * np.pi ** 2)
    )

    assert np.allclose(halomod, hmcode_data["p"][:, iz], rtol=3e-2)
