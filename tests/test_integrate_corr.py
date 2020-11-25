"""
Simple tests for the integration scheme for ProjectedCF.
"""
import numpy as np
from halomod import ProjectedCF
from halomod import projected_corr_gal
from halomod.integrate_corr import angular_corr_gal, AngularCF
from mpmath import hyp2f1
from scipy.special import gamma
import pytest
from scipy.integrate import dblquad, quad
from astropy.cosmology import Planck15, z_at_value
from astropy.units import Mpc

hyp2f1A = np.frompyfunc(lambda a, b, c, z: float(hyp2f1(a, b, c, z)), 4, 1)


def wprp_power_law(rp, r0, g):
    return np.sqrt(np.pi) * rp * (rp / r0) ** -g * gamma((g - 1) / 2.0) / gamma(g / 2.0)


def wprp_power_law_lim(rp, r0, g, rmax):
    return (
        (1 / gamma(g / 2))
        * (rp * rmax / r0) ** -g
        * gamma((g - 1) / 2)
        * (
            gamma(0.5) * rp * rmax ** g
            - rp ** g
            * rmax
            * gamma(g / 2)
            * hyp2f1A(0.5, (g - 1) / 2, (g + 1) / 2, rp ** 2 / rmax ** 2)
        )
    ).astype("float")


class TestProjCorr:
    rp = np.logspace(-2, 0.5, 50)
    gamma = 1.9
    r0 = 3.0

    def test_auto_rlim(self):
        h = ProjectedCF(rp_min=self.rp)  # This should imitate an "infinite" upper bound
        xir = (h.r / self.r0) ** -self.gamma

        wprp_anl = wprp_power_law(self.rp, self.r0, self.gamma)
        wprp = projected_corr_gal(h.r, xir, h.rlim, self.rp)
        assert np.allclose(wprp, wprp_anl, rtol=5e-2)

    def test_fixed_rlim(self):
        h = ProjectedCF(rp_min=self.rp, proj_limit=50.0)
        xir = (h.r / self.r0) ** -self.gamma

        wprp_anl = wprp_power_law_lim(self.rp, self.r0, self.gamma, 50.0)
        wprp = projected_corr_gal(h.r, xir, h.rlim, self.rp)
        assert np.allclose(wprp, wprp_anl, rtol=5e-2)


class TestAngularCF:
    @classmethod
    def power_law_solution(cls, theta, r0, gamma):
        """Solution of the angular correlation function (Blake08 Eq 33) for power-law xi."""
        # Let f(x) be a uniform distribution between 0 and 1000
        return [
            2
            * dblquad(
                lambda u, x: ((u ** 2 + x ** 2 * t ** 2) / r0 ** 2) ** (-gamma / 2),
                a=1000,
                b=2000,
                gfun=0,
                hfun=np.inf,
            )[0]
            / 1e6
            for t in theta
        ]  # 1e6 to account for integral of f(x)

    @classmethod
    def power_law_integral_over_u(cls, theta, x, r0, gamma):
        """Solution of the angular correlation function (Blake08 Eq 33) for power-law xi."""
        # Let f(x) be a uniform distribution between 0 and 1000
        return np.array(
            [
                [
                    2
                    * quad(
                        lambda u: ((u ** 2 + xx ** 2 * t ** 2) / r0 ** 2)
                        ** (-gamma / 2),
                        a=0,
                        b=np.inf,
                    )[0]
                    / 1e6
                    for t in theta  # 1e6 to account for integral of f(x)
                ]
                for xx in x
            ]
        ).T

    def test_power_law(self):
        theta = np.logspace(-2, -1, 5)

        r0 = 3  # Mpc
        gamma = 1.8

        zmin = z_at_value(Planck15.comoving_distance, 1000 * Mpc / Planck15.h)
        zmax = z_at_value(Planck15.comoving_distance, 2000 * Mpc / Planck15.h)

        num = angular_corr_gal(
            theta=theta,
            xi=lambda r: (r / r0) ** -gamma,
            p1=lambda x: 1 / 1000.0 * np.ones_like(x),
            zmin=zmin,
            zmax=zmax,
            logu_min=-6,
            logu_max=4,
            unum=1000,
            znum=500,
            cosmo=Planck15,
            p_of_z=False,
        )
        anl = self.power_law_solution(theta, r0, gamma)

        assert np.allclose(num, anl, rtol=5e-2)

    def test_against_blake(self):
        """Simple order-of-magnitude test of ACF against Blake+08 (Fig 4)"""
        acf = AngularCF(
            z=0.475,
            zmin=0.45,
            zmax=0.5,
            transfer_model="EH",
            hod_model="Zheng05",
            hod_params={
                "M_min": 12.98,
                "M_0": -10,
                "M_1": 14.09,
                "sig_logm": 0.21,
                "alpha": 1.57,
            },
            theta_min=1e-3 * np.pi / 180.0,
            theta_max=np.pi / 180.0,
            theta_num=2,
        )

        assert 10 < acf.angular_corr_gal[0] < 100
        assert 0.01 < acf.angular_corr_gal[1] < 0.1
