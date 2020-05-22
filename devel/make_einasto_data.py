"""
Module to do the Fourier Transform of the Einasto profile numerically and cache results for later use.

We save a binary file to the data directory to be read in, but also some ASCII files for long-term use.
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as spline


def f(x, a=0.18):
    return np.exp((-2 / a) * (x ** a - 1))


def _p(K, c):
    minsteps = 1000

    res = np.zeros((len(K), len(c)))

    for ik, kappa in enumerate(K):
        smallest_period = np.pi / kappa
        dx = smallest_period / 10

        nsteps = max(int(np.ceil(c.max() / dx)), minsteps)

        x, dx = np.linspace(0, c.max(), nsteps, retstep=True)

        spl = spline(x, x * f(x) * np.sin(kappa * x) / kappa)
        intg = spl.antiderivative()
        res[ik, :] = intg(c) - intg(0)

    return np.clip(res, 0, None)


K = np.logspace(-5, 4, 1000)
c = np.logspace(0, 2, 1000)

pk = _p(K, c)

np.savez("../src/halomod/data/uKc_einasto.npz", pk=pk, K=K, c=c)
np.savetxt("c_einasto.dat", c)
np.savetxt("K_einasto.dat", K)
np.savetxt("pkc_einasto.dat", pk)
