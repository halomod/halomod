"""
This module tests halomod against hmcode as run in a particular configuration.

We explicitly install and run HMCode as part of the test setup to make it clear what
parameters are being used etc.

See devel/HMCode.ipynb for a worded work-through of these automatic tests (may be helpful
if these tests break). Note that in the notebook, a forked version of the HMCode repo is
used (https://github.com/steven-murray/hmcode) which has a branch in which more
information is written out.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
from halomod import DMHaloModel
from hmf import MassFunction
from matplotlib import pyplot

MassFunction.ERROR_ON_BAD_MDEF = False


def read_power(fname: Path):
    """Read the power.dat file from HMcode."""
    # Each column is the power at a different redshift.
    with fname.open() as fl:
        line = fl.readline().split("#####")[-1].split("        ")[1:]
        redshifts = [float(x) for x in line]

    data = np.genfromtxt(fname, skip_header=1)
    k = data[:, 0]
    return k, redshifts, data[:, 1:]


@pytest.fixture(scope="module")
def hmcode_data(datadir):
    k, z, data = read_power(datadir / "hmcode_power.dat")
    return {"k": k, "z": z, "p": data}


@pytest.fixture(scope="module")
def hm():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", category=UserWarning, message="Your input mass definition"
        )
        return DMHaloModel(
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
            force_1halo_turnover=False,
        )


@pytest.mark.filterwarnings("ignore:Requested mass definition")
@pytest.mark.filterwarnings("ignore:You are using an un-normalized mass function")
@pytest.mark.parametrize("iz", range(16))
def test_hmcode(hm, hmcode_data, iz, plt):
    z = hmcode_data["z"][iz]

    fac = hmcode_data["k"] ** 3 / (2 * np.pi**2)
    hm.update(z=z)
    halomod = hm.power_auto_matter_fnc(hmcode_data["k"]) * fac

    if plt == pyplot:
        fig, ax = plt.subplots(2, 1, sharex=True)

        ax[0].plot(hmcode_data["k"], hmcode_data["p"][:, iz])
        ax[0].plot(hmcode_data["k"], halomod, ls="--")
        ax[0].plot(
            hmcode_data["k"],
            fac * hm.power_1h_auto_matter_fnc(hmcode_data["k"]),
            color="r",
            ls="--",
        )

        ax[0].plot(
            hmcode_data["k"],
            fac * hm.power_2h_auto_matter_fnc(hmcode_data["k"]),
            color="g",
            ls="--",
        )

        ax[0].set_xscale("log")
        ax[0].set_yscale("log")
        ax[0].set_title(f"z={z}")
        ax[0].set_ylim(1e-10, 1e3)

        ax[1].plot(
            hmcode_data["k"],
            hmcode_data["p"][:, iz] / (fac * hm.power_auto_matter_fnc(hmcode_data["k"])) - 1,
        )
        ax[1].axhline(0.03, color="k", ls="--")
        ax[1].axhline(-0.03, color="k", ls="--")

    np.testing.assert_allclose(halomod, hmcode_data["p"][:, iz], rtol=3e-2)
