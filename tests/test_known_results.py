"""
Test the halo_model results against known 'correct' results for regression
testing.
"""


from halomod import HaloModel
import numpy as np
import pytest
from pathlib import Path
import itertools

pytestmark = pytest.mark.skip

DATA = Path(__file__).parent / "data"


def rms_diff(vec1, vec2, tol):
    mask = np.logical_and(
        np.logical_not(np.isnan(vec1)), np.logical_not(np.isnan(vec2))
    )
    vec1 = vec1[mask]
    vec2 = vec2[mask]
    err = np.sqrt(np.mean(((vec1 - vec2) / vec2) ** 2))
    print("RMS Error: ", err, "(> ", tol, ")")
    return err < tol


def max_diff_rel(vec1, vec2, tol):
    mask = np.logical_and(
        np.logical_not(np.isnan(vec1)), np.logical_not(np.isnan(vec2))
    )
    mask = np.logical_and(mask, vec2 != 0)
    vec1 = vec1[mask]
    vec2 = vec2[mask]
    err = np.max(np.abs((vec1 - vec2) / vec2))
    print("Max Diff: ", err, "(> ", tol, ")")
    return err < tol


def max_diff(vec1, vec2, tol):
    mask = np.logical_and(
        np.logical_not(np.isnan(vec1)), np.logical_not(np.isnan(vec2))
    )
    vec1 = vec1[mask]
    vec2 = vec2[mask]
    err = np.max(np.abs((vec1 - vec2)))
    print("Max Diff: ", err, "(> ", tol, ")")
    return err < tol


# ===============================================================================
# The Test Classes
# ===============================================================================
class TestKnown(object):

    H0 = 70.0

    r = np.zeros(100)
    for i in range(100):
        r[i] = 0.05 * 1.0513 ** i

    maxm = 1.02e10
    dm = maxm / 15
    for i in range(224):
        maxm += dm
        dm = maxm / 15
    maxm /= 1.03

    m = 10 ** 11.8363 * 100 / H0

    dm = m / 100
    ms = []
    while m < maxm:
        m += dm / 2
        ms.append(m)
        m += dm / 2
        dm = m / 100

    ms = np.log10(np.array(ms) * H0 / 100)

    lnk = np.zeros(55)
    for i in range(55):
        lnk[i] = 1e-3 * 1.25 ** i

    lnk = np.log(lnk)
    hod = HaloModel(
        lnk_min=np.log(1e-3),
        lnk_max=np.log(1e-3 * 1.25 ** 54),
        dlnk=np.log(1.25),
        Mmin=ms.min(),
        Mmax=ms.max(),
        dlog10m=np.log10(ms[1] / ms[0]),
        rmin=r,
        transfer_model="EH",
        hmf_model="SMT",
        bias_model="SMT01",
        hc_spectrum="linear",
        exclusion_model=None,
        halo_concentration_model="Zehavi11",
        sd_bias_model=None,
        z=0.0,
        cosmo_params={"H0": H0, "Ob0": 0.05, "Om0": 0.3,},
        sigma_8=0.8,
        n=1.0,
        hod_params={
            "M_1": 12.8510,
            "M_0": 11.5047,
            "sig_logm": 0.26,
            "M_min": 11.6222,
            "alpha": 1.049,
        },
        hod_model="Zheng05",
        mdef_model="SOCritical",
        halo_profile_model="NFW",
    )

    # Initialize the hod with most stuff it needs.
    hod.corr_auto_tracer

    @pytest.mark.parametrize(
        "z,prop",
        itertools.product([0.0, 2.0], ["rho", "lam", "rhor", "lamr", "uk", "um"]),
    )
    def test_profile(self, z, prop):
        data = np.genfromtxt(DATA / f"{prop}z{z}")

        # NOTE TO FUTURE SELF: need to put update here rather than in loop in test_profile or
        # else it doesn't work!!!
        hod = self.hod.clone(z=z)
        if prop == "rho":
            assert max_diff_rel(
                hod.halo_profile.rho(hod.r[0], hod.hmf.m, norm="m"),
                data[:, 1] / hod.cosmo.h ** 3,
                0.01,
            )
        elif prop == "lam":
            assert max_diff_rel(
                hod.halo_profile.lam(hod.r[0], hod.hmf.m),
                data[:, 1] / hod.cosmo.h,
                0.1,
            )

        elif prop == "rhor":
            m = data[0, 2] * hod.cosmo.h
            assert max_diff_rel(
                hod.halo_profile.rho(hod.r, m, norm="m"),
                data[:, 1] / hod.cosmo.h ** 3,
                0.01,
            )
        elif prop == "lamr":
            m = data[0, 2] * hod.cosmo.h
            assert max_diff_rel(
                hod.halo_profile.lam(hod.r, m), data[:, 1] / hod.cosmo.h, 0.02,
            )

        elif prop == "uk":
            m = data[0, 2] * hod.cosmo.h

            assert max_diff_rel(
                hod.halo_profile.u(hod.transfer.k, m, norm="m"), data[:, 1], 0.01,
            )
        elif prop == "um":
            assert max_diff_rel(
                hod.halo_profile.u(hod.transfer.k[0], hod.hmf.m, norm="m"),
                data[:, 1],
                0.01,
            )

    @pytest.mark.parametrize("z", [0.0, 2.0])
    def test_bias(self, z):
        data = np.genfromtxt(DATA / f"bias_STz{z}")
        hod = self.hod.clone(z=z)
        assert max_diff_rel(hod.bias, data[:, 1], 0.01)

    @pytest.mark.parametrize(
        "z,prop", [(0.0, "ncen"), (0.0, "nsat"), (2.0, "ncen"), (2.0, "nsat")]
    )
    def test_hod(self, z, prop):
        hod = self.hod.clone(z=z)
        data = np.genfromtxt(DATA / f"{prop}z{z}")
        if prop == "ncen":
            assert max_diff_rel(hod.central_occupation, data[:, 1], 0.01)
        elif prop == "nsat":
            assert max_diff_rel(hod.satellite_occupation, data[:, 1], 0.01)

    @pytest.mark.parametrize(
        "z,nonlinear,halo_exclusion,prop",
        itertools.product(
            [0.0], [False, True], [None, "schneider"], ["1h", "2h", "tot"]
        ),
    )
    def test_corr(self, z, nonlinear, halo_exclusion, prop):
        if halo_exclusion:
            pytest.skip("Halo Exclusion currently causing memory errors!")

        if nonlinear:
            ltag = "NL"
        else:
            ltag = "L"
        if halo_exclusion != "None":
            ex = "_eX"
        else:
            ex = ""
        data = np.genfromtxt(DATA / f"xirz{z}_EH_ST_ST_{ltag}{ex}.dat")

        # Update the hod object
        hod = self.hod.clone(
            z=z,
            hc_spectrum="nonlinear" if nonlinear else "linear",
            exclusion_model="NgMatched" if halo_exclusion else None,
        )

        if prop == "1h":
            assert max_diff_rel(hod.corr_1h_auto_tracer, data[:, 2], 0.01)
        elif prop == "2h":
            assert max_diff_rel(hod.corr_2h_auto_tracer, data[:, 1], 0.01)
        elif prop == "tot":
            assert max_diff_rel(hod.corr_auto_tracer, data[:, 3], 0.01)
