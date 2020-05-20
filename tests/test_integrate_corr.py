"""
Simple tests for the integration scheme for ProjectedCF (thus far). Doesn't work
as yet, since it tries to import the libraries from *this* folder, rather than
installation (which doesn't work because the fortran code isn't installed.)
"""

# LOCATION = "/".join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split("/")[:-1])


# sys.path.insert(0, LOCATION)
import numpy as np
from src.halomod import ProjectedCF
from src.halomod import projected_corr_gal
from mpmath import gamma, hyp2f1

hyp2f1A = np.frompyfunc(lambda a, b, c, z: float(hyp2f1(a, b, c, z)), 4, 1)


def wprp_pl(rp, r0, g):
    return rp * (rp / r0) ** -g * gamma(0.5) * gamma((g - 1) / 2.0) / gamma(g / 2.0)


def wprp_pl_lim(rp, r0, g, rmax):
    return (
        (1 / gamma(g / 2))
        * (rp * rmax / r0) ** -g
        * gamma((g - 1) / 2)
        * (
            gamma(0.5) * rp * rmax ** g
            - rp ** g
            * rmax
            * gamma(g / 2)
            * hyp2f1A(0.5, (g - 1) / 2, (g + 1) / 2, rp.value ** 2 / rmax.value ** 2)
        )
    )


class TestProjCorr:
    def __init__(self):
        self.rp = np.logspace(-2, 1.2, 50)
        self.gamma = 1.85  # Values from S1 sample of Beutler+2011
        self.r0 = 5.14

        self.wprp_anl = wprp_pl(self.rp, self.r0, self.gamma)
        self.wprp_anl_rlim = wprp_pl_lim(self.rp, self.r0, self.gamma, 50.0)

    def test_auto_rlim(self):
        h = ProjectedCF(rp_min=self.rp)  # This should imitate an "infinite" upper bound
        xir = (h.r.value / self.r0) ** -self.gamma

        wprp_anl = wprp_pl(self.rp, self.r0, self.g)
        wprp = projected_corr_gal(h.r, xir, h.rlim, self.rp)
        assert np.all(abs(wprp - wprp_anl) / wprp_anl < 0.01)

    def test_fixed_rlim(self):
        h = ProjectedCF(rp_min=self.rp, proj_limit=50.0)
        xir = (h.r.value / self.r0) ** -self.gamma

        wprp_anl = wprp_pl_lim(self.rp, self.r0, self.g, 50.0)
        wprp = projected_corr_gal(h.r, xir, h.rlim, self.rp)
        assert np.all(abs(wprp - wprp_anl) / wprp_anl < 0.01)
