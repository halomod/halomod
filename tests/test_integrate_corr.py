"""
Simple tests for the integration scheme for ProjectedCF.
"""
import numpy as np
from halomod import ProjectedCF
from halomod import projected_corr_gal
from mpmath import hyp2f1
from scipy.special import gamma
import pytest

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
