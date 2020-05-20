"""
Test the halo_model results against known 'correct' results for regression
testing.
"""

from hod import HaloModel
import numpy as np
import inspect
import os
import sys

PLOT = True
if PLOT:
    import matplotlib.pyplot as plt

    pref = "/Users/Steven/Documents/PhD/TestCharlesPlots/"
    from os.path import join

LOCATION = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


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
    def __init__(self):
        """
        Set up the main parameters of the run to be the same as the comparison
        code
        """
        self.H0 = 70.0

        self.r = np.zeros(100)
        for i in range(100):
            self.r[i] = 0.05 * 1.0513 ** i

        self.maxm = 1.02e10
        dm = self.maxm / 15
        for i in range(224):
            self.maxm += dm
            dm = self.maxm / 15
        self.maxm /= 1.03

        self.m = 10 ** 11.8363 * 100 / self.H0

        dm = self.m / 100
        self.ms = []
        while self.m < self.maxm:
            self.m += dm / 2
            self.ms.append(self.m)
            self.m += dm / 2
            dm = self.m / 100

        self.ms = np.log10(np.array(self.ms) * self.H0 / 100)

        lnk = np.zeros(55)
        for i in range(55):
            lnk[i] = 1e-3 * 1.25 ** i

        lnk = np.log(lnk)
        self.hod = HaloModel(
            lnk=lnk,
            M=self.ms,
            r=self.r,
            transfer_fit="EH",
            mf_fit="SMT",
            bias_model="ST",
            nonlinear=False,
            halo_exclusion="None",
            cm_relation="zehavi",
            scale_dependent_bias=False,
            z=0.0,
            H0=self.H0,
            omegab=0.05,
            omegac=0.25,
            omegav=0.7,
            sigma_8=0.8,
            n=1.0,
            M_1=12.8510,
            M_0=11.5047,
            gauss_width=0.26,
            M_min=11.6222,
            alpha=1.049,
            delta_wrt="crit",
            halo_profile="NFW",
        )

    def check_profile(self, z, prop):
        """Check halo_profile-related quantities"""
        data = np.genfromtxt(LOCATION + "/data/" + prop + "z" + str(z))

        # NOTE TO FUTURE SELF: need to put update here rather than in loop in test_profile or
        # else it doesn't work!!!
        self.hod.update(z=z)
        if prop == "rho":
            if PLOT:
                plt.clf()
                plt.plot(
                    self.hod.hmf.M,
                    self.hod.halo_profile.rho(self.hod.r[0], self.hod.hmf.M, norm="m"),
                    label="mine",
                )
                plt.plot(
                    data[:, 0] * self.hod.cosmo.h,
                    data[:, 1] / self.hod.cosmo.h ** 3,
                    label="charles",
                )
                plt.legend()
                plt.xscale("log")
                plt.yscale("log")
                plt.savefig(join(pref, "rho" + prop + "z" + str(z) + ".pdf"))
            assert max_diff_rel(
                self.hod.halo_profile.rho(self.hod.r[0], self.hod.hmf.M, norm="m"),
                data[:, 1] / self.hod.cosmo.h ** 3,
                0.01,
            )
        elif prop == "lam":
            if PLOT:
                plt.clf()
                plt.plot(
                    self.hod.hmf.M,
                    self.hod.halo_profile.lam(self.hod.r[0], self.hod.hmf.M),
                    label="mine",
                )
                plt.plot(
                    data[:, 0] * self.hod.cosmo.h,
                    data[:, 1] / self.hod.cosmo.h,
                    label="charles",
                )
                plt.legend()
                plt.xscale("log")
                plt.yscale("log")
                plt.savefig(join(pref, "lam" + prop + "z" + str(z) + ".pdf"))
            assert max_diff_rel(
                self.hod.halo_profile.lam(self.hod.r[0], self.hod.hmf.M),
                data[:, 1] / self.hod.cosmo.h,
                0.1,
            )

        elif prop == "rhor":
            m = data[0, 2] * self.hod.cosmo.h
            if PLOT:
                plt.clf()
                plt.plot(
                    self.hod.r,
                    self.hod.halo_profile.rho(self.hod.r, m, norm="m"),
                    label="mine",
                )
                plt.plot(
                    data[:, 0] * self.hod.cosmo.h,
                    data[:, 1] / self.hod.cosmo.h ** 3,
                    label="charles",
                )
                plt.legend()
                plt.xscale("log")
                plt.yscale("log")
                plt.savefig(join(pref, "rhor" + prop + "z" + str(z) + ".pdf"))
            assert max_diff_rel(
                self.hod.halo_profile.rho(self.hod.r, m, norm="m"),
                data[:, 1] / self.hod.cosmo.h ** 3,
                0.01,
            )
        elif prop == "lamr":
            m = data[0, 2] * self.hod.cosmo.h
            if PLOT:
                plt.clf()
                plt.plot(
                    self.hod.r, self.hod.halo_profile.lam(self.hod.r, m), label="mine"
                )
                plt.plot(
                    data[:, 0] * self.hod.cosmo.h,
                    data[:, 1] / self.hod.cosmo.h,
                    label="charles",
                )
                plt.legend()
                plt.xscale("log")
                plt.yscale("log")
                plt.savefig(join(pref, "lamr" + prop + "z" + str(z) + ".pdf"))
            assert max_diff_rel(
                self.hod.halo_profile.lam(self.hod.r, m),
                data[:, 1] / self.hod.cosmo.h,
                0.02,
            )

        elif prop == "uk":
            m = data[0, 2] * self.hod.cosmo.h
            if PLOT:
                plt.clf()
                plt.plot(
                    np.exp(self.hod.transfer.lnk),
                    self.hod.halo_profile.u(np.exp(self.hod.transfer.lnk), m, norm="m"),
                    label="mine",
                )
                plt.plot(data[:, 0] / self.hod.cosmo.h, data[:, 1], label="charles")
                plt.legend()
                plt.xscale("log")
                plt.yscale("log")
                plt.savefig(join(pref, "uk" + prop + "z" + str(z) + ".pdf"))

            assert max_diff_rel(
                self.hod.halo_profile.u(np.exp(self.hod.transfer.lnk), m, norm="m"),
                data[:, 1],
                0.01,
            )
        elif prop == "um":
            if PLOT:
                plt.clf()
                plt.plot(
                    self.hod.hmf.M,
                    self.hod.halo_profile.u(
                        np.exp(self.hod.transfer.lnk[0]), self.hod.hmf.M, norm="m"
                    ),
                    label="mine",
                )
                plt.plot(data[:, 0] / self.hod.cosmo.h, data[:, 1], label="charles")
                plt.legend()
                plt.xscale("log")
                plt.yscale("log")
                plt.savefig(join(pref, "um" + prop + "z" + str(z) + ".pdf"))
            assert max_diff_rel(
                self.hod.halo_profile.u(
                    np.exp(self.hod.transfer.lnk[0]), self.hod.hmf.M, norm="m"
                ),
                data[:, 1],
                0.01,
            )

    def test_profile(self):
        for z in [0.0, 2.0]:
            for prop in ["rho", "lam", "rhor", "lamr", "uk", "um"]:
                yield self.check_profile, z, prop

    def check_bias(self, z):
        data = np.genfromtxt(LOCATION + "/data/" + "bias_STz" + str(z))
        self.hod.update(z=z)
        if PLOT:
            plt.clf()
            plt.plot(self.hod.hmf.M, self.hod.bias.bias, label="mine")
            plt.plot(data[:, 0] * self.hod.cosmo.h, data[:, 1], label="charles")
            plt.legend()
            plt.xscale("log")
            plt.yscale("log")
            plt.savefig(join(pref, "biasz" + str(z) + ".pdf"))
        assert max_diff_rel(self.hod.bias.bias, data[:, 1], 0.01)

    def test_bias(self):
        for z in [0.0, 2.0]:
            yield self.check_bias, z

    def check_hod(self, z, prop):
        data = np.genfromtxt(LOCATION + "/data/" + prop + "z" + str(z))
        if prop == "ncen":
            if PLOT:
                plt.clf()
                plt.plot(self.hod.hmf.M, self.hod.n_cen, label="mine")
                plt.plot(data[:, 0] * self.hod.cosmo.h, data[:, 1], label="charles")
                plt.legend()
                plt.xscale("log")
                plt.yscale("log")
                plt.savefig(join(pref, "ncen" + prop + "z" + str(z) + ".pdf"))
            assert max_diff_rel(self.hod.n_cen, data[:, 1], 0.01)
        elif prop == "nsat":
            if PLOT:
                plt.clf()
                plt.plot(self.hod.hmf.M, self.hod.n_sat, label="mine")
                plt.plot(data[:, 0] * self.hod.cosmo.h, data[:, 1], label="charles")
                plt.legend()
                plt.xscale("log")
                plt.yscale("log")
                plt.savefig(join(pref, "nsat" + prop + "z" + str(z) + ".pdf"))
            assert max_diff_rel(self.hod.n_sat, data[:, 1], 0.01)

    def test_hod(self):
        for z in [0.0, 2.0]:
            self.hod.update(z=z)
            for prop in ["ncen", "nsat"]:
                yield self.check_hod, z, prop

    def check_corr(self, z, nonlinear, halo_exclusion, prop):
        if nonlinear:
            ltag = "NL"
        else:
            ltag = "L"
        if halo_exclusion != "None":
            ex = "_EX"
        else:
            ex = ""
        data = np.genfromtxt(
            LOCATION + "/data/" + "xirz" + str(z) + "_EH_ST_ST_" + ltag + ex + ".dat"
        )

        # Update the hod object
        self.hod.update(z=z, nonlinear=nonlinear, halo_exclusion=halo_exclusion)

        if prop == "1h":
            if PLOT:
                plt.clf()
                plt.plot(self.hod.r, self.hod.corr_gal_1h, label="mine")
                plt.plot(data[:, 0], data[:, 2], label="charles")
                plt.legend()
                plt.xscale("log")
                plt.yscale("log")
                plt.savefig(
                    join(pref, "1h" + str(z) + "_EH_ST_ST_" + ltag + ex + ".pdf")
                )
            assert max_diff_rel(self.hod.corr_gal_1h, data[:, 2], 0.01)
        elif prop == "2h":
            if PLOT:
                plt.clf()
                plt.plot(self.hod.r, self.hod.corr_gal_2h, label="mine")
                plt.plot(data[:, 0], data[:, 1], label="charles")
                plt.legend()
                plt.xscale("log")
                plt.yscale("log")
                plt.savefig(
                    join(pref, "2h" + str(z) + "_EH_ST_ST_" + ltag + ex + ".pdf")
                )
            assert max_diff_rel(self.hod.corr_gal_2h, data[:, 1], 0.01)
        elif prop == "tot":
            if PLOT:
                plt.clf()
                plt.plot(self.hod.r, self.hod.corr_gal, label="mine")
                plt.plot(data[:, 0], data[:, 3], label="charles")
                plt.legend()
                plt.xscale("log")
                plt.yscale("log")
                plt.savefig(
                    join(pref, "corr" + str(z) + "_EH_ST_ST_" + ltag + ex + ".pdf")
                )
            assert max_diff_rel(self.hod.corr_gal, data[:, 3], 0.01)

    def test_corr(self):
        for z in [0.0]:  # [0.0, 2.0]:
            for nonlinear in [False, True]:  # [True, False]:
                for halo_exclusion in ["None", "schneider"]:  # ["None", "schneider"]:
                    for prop in ["1h", "2h", "tot"]:
                        yield self.check_corr, z, nonlinear, halo_exclusion, prop


if __name__ == "__main__":
    t = TestKnown()
    print(t.m)
    print(t.r)

    t.test_bias()
    print("all done..")
