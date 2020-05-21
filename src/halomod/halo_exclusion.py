"""
Module defining halo model components for halo exclusion.
"""
import numpy as np
from hmf import Component
from cached_property import cached_property
from scipy import integrate as intg
import warnings

try:
    from numba import jit

    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    warnings.warn(
        "Warning: Some Halo-Exclusion models have significant speedup when using Numba"
    )


# ===============================================================================
# UTILITIES
# ===============================================================================
def outer(a, b):
    return np.outer(a, b).reshape(a.shape + b.shape)


def dbltrapz(X, dx, dy=None):
    """
    Double-integral over the last two dimensions of X using trapezoidal rule
    """
    dy = dy or dx
    out = X.copy()
    out[..., 1:-1, :] *= 2
    out[..., :, 1:-1] *= 2
    return dx * dy * np.sum(out, axis=(-2, -1)) / 4.0


def dblsimps(X, dx, dy=None):
    """
    Double-integral over the last two dimensions of X.
    """
    if dy is None:
        dy = dx

    if X.shape[-2] % 2 == 0:
        X = X[..., :-1, :]
    if X.shape[-1] % 2 == 0:
        X = X[..., :-1]

    (nx, ny) = X.shape[-2:]

    W = makeW(nx, ny)

    return dx * dy * np.sum(W * X, axis=(-2, -1)) / 9.0


def makeW(nx, ny):
    W = np.ones((nx, ny))
    W[1 : nx - 1 : 2, :] *= 4
    W[:, 1 : ny - 1 : 2] *= 4
    W[2 : nx - 1 : 2, :] *= 2
    W[:, 2 : ny - 1 : 2] *= 2
    return W


if USE_NUMBA:

    @jit(nopython=True)
    def dblsimps_(X, dx, dy):
        """
        Double-integral of X FOR SYMMETRIC FUNCTIONS
        """
        nx = X.shape[0]
        ny = X.shape[1]

        # Must be odd number
        if nx % 2 == 0:
            nx -= 1
        if ny % 2 == 0:
            ny -= 1

        W = makeW_(nx, ny)  # only upper

        tot = 0.0
        for ix in range(nx):
            tot += W[ix, ix] * X[ix, ix]
            for iy in range(ix + 1, ny):
                tot += 2 * W[ix, iy] * X[ix, iy]

        return dx * dy * tot / 9.0

    @jit(nopython=True)
    def makeW_(nx, ny):
        W = np.ones((nx, ny))
        for ix in range(1, nx - 1, 2):
            for iy in range(ny):
                W[ix, iy] *= 4
                W[iy, ix] *= 4

        for ix in range(2, nx - 1, 2):
            for iy in range(ny):
                W[ix, iy] *= 2
                W[iy, ix] *= 2

        return W

    @jit(nopython=True)
    def makeH_(nx, ny):
        H = np.ones((nx, ny))
        for ix in range(1, nx - 1):
            for iy in range(ny):
                H[ix, iy] *= 2
                H[iy, ix] *= 2

        return H

    @jit(nopython=True)
    def dbltrapz_(X, dx, dy):
        nx = X.shape[0]
        ny = X.shape[1]

        H = makeH_(nx, ny)
        tot = 0.0
        for ix in range(nx):
            tot += H[ix, ix] * X[ix, ix]
            for iy in range(ix + 1, ny):
                tot += 2 * H[ix, iy] * X[ix, iy]

        return dx * dy * tot / 4.0


# ===============================================================================
# Halo-Exclusion Models
# ===============================================================================
class Exclusion(Component):
    """
    Base class for exclusion models.

    All models will need to perform single or double integrals over
    arrays that may have an extra two dimensions. The maximum possible
    size is k*r*m*m, which for normal values of the vectors equates to
    ~ 1000*50*500*500 = 12,500,000,000 values, which in 64-bit reals is
    1e11 bytes = 100GB. We thus limit this to a maximum of either k*r*m
    or r*m*m, both of which should be less than a GB of memory.

    It is possibly better to limit it to k*r or m*m, which should be quite
    memory efficient, but then without accelerators (ie. Numba), these
    will be very slow.
    """

    def __init__(self, m, density, Ifunc, bias, r, delta_halo, mean_density):
        self.density = density  # 1d, (m)
        self.m = m  # 1d, (m)
        self.Ifunc = Ifunc  # 2d, (k,m)
        self.bias = bias  # 1d (m) or 2d (r,m)
        self.r = r  # 1d (r)

        self.mean_density = mean_density
        self.delta_halo = delta_halo
        self.dlnx = np.log(m[1] / m[0])

    def raw_integrand(self):
        """
        Returns either a 2d (k,m) or 3d (r,k,m) array with the general integrand.
        """
        if len(self.bias.shape) == 1:
            return self.Ifunc * self.bias * self.m  # *m since integrating in logspace
        else:
            return np.einsum("ij,kj->kij", self.Ifunc * self.m, self.bias)

    def integrate(self):
        """
        This should pass back whatever is multiplied by P_m(k) to get the two-halo
        term. Often this will be a square of an integral, sometimes a Double-integral.
        """
        pass


class NoExclusion(Exclusion):
    def integrate(self):
        return intg.simps(self.raw_integrand(), dx=self.dlnx) ** 2


class Sphere(Exclusion):
    def raw_integrand(self):
        if len(self.bias.shape) == 1:
            return outer(
                np.ones_like(self.r), self.Ifunc * self.bias * self.m
            )  # *m since integrating in logspace
        else:
            return np.einsum("ij,kj->kij", self.Ifunc * self.m, self.bias)

    @cached_property
    def density_mod(self):
        """
        Return the modified density, under new limits
        """
        density = np.outer(np.ones_like(self.r), self.density * self.m)
        density[self.mask] = 0
        return intg.simps(density, dx=self.dlnx)

    @cached_property
    def mask(self):
        "Elements that should be set to 0"
        return (np.outer(self.m, np.ones_like(self.r)) > self.mlim()).T

    def mlim(self):
        return 4 * np.pi * (self.r / 2) ** 3 * self.mean_density * self.delta_halo / 3

    def integrate(self):
        integ = self.raw_integrand()  # r,k,m
        integ.transpose((1, 0, 2))[:, self.mask] = 0
        return intg.simps(integ, dx=self.dlnx) ** 2


class DblSphere(Sphere):
    @property
    def rvir(self):
        return (3 * self.m / (4 * np.pi * self.delta_halo * self.mean_density)) ** (
            1.0 / 3.0
        )

    @cached_property
    def mask(self):
        "Elements that should be set to 0 (r,m,m)"
        rvir = self.rvir
        return (outer(np.add.outer(rvir, rvir), np.ones_like(self.r)) > self.r).T

    @cached_property
    def density_mod(self):
        out = np.zeros_like(self.r)
        for i, r in enumerate(self.r):
            integrand = np.outer(self.density * self.m, np.ones_like(self.density))
            integrand[self.mask] = 0
            out[i] = dblsimps(integrand, self.dlnx)
        return np.sqrt(out)

    def integrate(self):
        integ = self.raw_integrand()  # (r,k,m)
        return integrate_dblsphere(integ, self.mask, self.dlnx)


def integrate_dblsphere(integ, mask, dx):
    out = np.zeros_like(integ[:, :, 0])
    integrand = np.zeros_like(mask)
    for ik in range(integ.shape[1]):
        for ir in range(mask.shape[0]):
            integrand[ir] = np.outer(integ[ir, ik, :], integ[ir, ik, :])
        integrand[mask] = 0
        out[:, ik] = dblsimps(integrand, dx)
    return out


if USE_NUMBA:

    @jit(nopython=True)
    def integrate_dblsphere_(integ, mask, dx):
        nr = integ.shape[0]
        nk = integ.shape[1]
        nm = mask.shape[1]

        out = np.zeros((nr, nk))
        integrand = np.zeros((nm, nm))

        for ir in range(nr):
            for ik in range(nk):
                for im in range(nm):
                    for jm in range(im, nm):
                        if mask[ir, im, jm]:
                            integrand[im, jm] = 0
                        else:
                            integrand[im, jm] = integ[ir, ik, im] * integ[ir, ik, jm]
                out[ir, ik] = dblsimps_(integrand, dx, dx)
        return out

    class DblSphere_(DblSphere):
        def integrate(self):
            integ = self.raw_integrand()  # (r,k,m)
            return integrate_dblsphere_(integ, self.mask, self.dlnx)


class DblEllipsoid(DblSphere):
    @cached_property
    def mask(self):
        "Unecessary for this approach"
        return None

    @cached_property
    def prob(self):
        rvir = self.rvir
        x = outer(self.r, 1 / np.add.outer(rvir, rvir))
        x = (x - 0.8) / 0.29  # this is y but we re-use the memory
        np.clip(x, 0, 1, x)
        return 3 * x ** 2 - 2 * x ** 3

    @cached_property
    def density_mod(self):
        integrand = self.prob * outer(
            np.ones_like(self.r), np.outer(self.density * self.m, self.density * self.m)
        )
        a = np.sqrt(dbltrapz(integrand, self.dlnx))

        return a

    def integrate(self):
        integ = self.raw_integrand()  # (r,k,m)
        out = np.zeros_like(integ[:, :, 0])

        integrand = np.zeros_like(self.prob)
        for ik in range(integ.shape[1]):

            for ir in range(len(self.r)):
                integrand[ir] = self.prob[ir] * np.outer(
                    integ[ir, ik, :], integ[ir, ik, :]
                )
            out[:, ik] = dbltrapz(integrand, self.dlnx)
        return out


if USE_NUMBA:

    class DblEllipsoid_(DblEllipsoid):
        @cached_property
        def density_mod(self):
            return density_mod_(
                self.r,
                self.rvir,
                np.outer(self.density * self.m, self.density * self.m),
                self.dlnx,
            )

        @cached_property
        def prob(self):
            return prob_inner_(self.r, self.rvir)

        def integrate(self):
            return integrate_dblell(self.raw_integrand(), self.prob, self.dlnx)

    @jit(nopython=True)
    def integrate_dblell(integ, prob, dx):
        nr = integ.shape[0]
        nk = integ.shape[1]
        nm = prob.shape[1]

        out = np.zeros((nr, nk))
        integrand = np.zeros((nm, nm))

        for ir in range(nr):
            for ik in range(nk):
                for im in range(nm):
                    for jm in range(im, nm):
                        integrand[im, jm] = (
                            integ[ir, ik, im] * integ[ir, ik, jm] * prob[ir, im, jm]
                        )
                out[ir, ik] = dbltrapz_(integrand, dx, dx)
        return out

    @jit(nopython=True)
    def density_mod_(r, rvir, densitymat, dx):
        d = np.zeros(len(r))
        for ir, rr in enumerate(r):
            integrand = prob_inner_r_(rr, rvir) * densitymat
            d[ir] = dbltrapz_(integrand, dx, dx)
        return np.sqrt(d)

    @jit(nopython=True)
    def prob_inner_(r, rvir):
        """
        Jit-compiled version of calculating prob, taking advantage of symmetry.
        """
        nrv = len(rvir)
        out = np.empty((len(r), nrv, nrv))
        for ir, rr in enumerate(r):
            for irv, rv1 in enumerate(rvir):
                for jrv in range(irv, nrv):
                    rv2 = rvir[jrv]
                    x = (rr / (rv1 + rv2) - 0.8) / 0.29
                    if x <= 0:
                        out[ir, irv, jrv] = 0
                    elif x >= 1:
                        out[ir, irv, jrv] = 1
                    else:
                        out[ir, irv, jrv] = 3 * x ** 2 - 2 * x ** 3
        return out

    @jit(nopython=True)
    def prob_inner_r_(r, rvir):
        nrv = len(rvir)
        out = np.empty((nrv, nrv))
        for irv, rv1 in enumerate(rvir):
            for jrv in range(irv, nrv):
                rv2 = rvir[jrv]
                x = (r / (rv1 + rv2) - 0.8) / 0.29
                if x <= 0:
                    out[irv, jrv] = 0
                elif x >= 1:
                    out[irv, jrv] = 1
                else:
                    out[irv, jrv] = 3 * x ** 2 - 2 * x ** 3
        return out


class NgMatched(DblEllipsoid):
    @cached_property
    def mask(self):
        integrand = self.density * self.m
        # cumint = cumsimps(integrand,dx = self.dlnx)
        cumint = intg.cumtrapz(integrand, dx=self.dlnx, initial=0)  # len m
        cumint = np.outer(np.ones_like(self.r), cumint)  # r,m
        return np.where(
            cumint > 1.0001 * np.outer(self.density_mod, np.ones_like(self.m)),
            np.ones_like(cumint, dtype=bool),
            np.zeros_like(cumint, dtype=bool),
        )

    def integrate(self):
        integ = self.raw_integrand()  # r,k,m
        integ.transpose((1, 0, 2))[:, self.mask] = 0
        return intg.simps(integ, dx=self.dlnx) ** 2


if USE_NUMBA:

    class NgMatched_(DblEllipsoid_):
        @cached_property
        def mask(self):
            integrand = self.density * self.m
            # cumint = cumsimps(integrand,dx = self.dlnx)
            cumint = intg.cumtrapz(integrand, dx=self.dlnx, initial=0)  # len m
            cumint = np.outer(np.ones_like(self.r), cumint)  # r,m
            return np.where(
                cumint > 1.0001 * np.outer(self.density_mod, np.ones_like(self.m)),
                np.ones_like(cumint, dtype=bool),
                np.zeros_like(cumint, dtype=bool),
            )

        def integrate(self):
            integ = self.raw_integrand()  # r,k,m
            integ.transpose((1, 0, 2))[:, self.mask] = 0
            return intg.simps(integ, dx=self.dlnx) ** 2


def cumsimps(func, dx):
    """
    A very simplistic cumulative simpsons rule integrator. func is an array,
    h is the equal spacing. It is somewhat inaccurate in the first few bins, since
    we just truncate the integral, regardless of whether it is odd or even numbered.

    Examples
    --------
    >>> x = np.linspace(0,1,1001)
    >>> y = np.sin(x)
    >>> print cumsimps(y,0.001)/(1-np.cos(x))
    """
    f1 = func.copy()
    f1[1:-1] *= 2
    f1[1:-1:2] *= 2
    rm = func.copy()
    rm[1:-1:2] *= 3
    cs = np.cumsum(f1)
    cs -= rm
    return cs * dx / 3
