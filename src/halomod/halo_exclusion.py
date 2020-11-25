"""
Module defining halo model components for halo exclusion.
"""
import numpy as np
from hmf import Component
from cached_property import cached_property
from scipy import integrate as intg
import warnings
from hmf._internals import pluggable

try:
    from numba import jit

    USE_NUMBA = True
except ImportError:  # pragma: no cover
    USE_NUMBA = False
    warnings.warn(
        "Warning: Some Halo-Exclusion models have significant speedup when using Numba"
    )


# ===============================================================================
# UTILITIES
# ===============================================================================
def outer(a, b):
    r"""
    Calculate the outer product of two vectors.
    """
    return np.outer(a, b).reshape(a.shape + b.shape)


def dbltrapz(X, dx, dy=None):
    """
    Double-integral over the last two dimensions of X using trapezoidal rule.
    """
    dy = dy or dx
    out = X.copy()
    out[..., 1:-1, :] *= 2
    out[..., :, 1:-1] *= 2
    return dx * dy * np.sum(out, axis=(-2, -1)) / 4.0


def makeW(nx, ny):
    r"""
    Return a window matrix for double-intergral.
    """
    W = np.ones((nx, ny))
    W[1 : nx - 1 : 2, :] *= 4
    W[:, 1 : ny - 1 : 2] *= 4
    W[2 : nx - 1 : 2, :] *= 2
    W[:, 2 : ny - 1 : 2] *= 2
    return W


if USE_NUMBA:

    @jit(nopython=True)
    def dblsimps_(X, dx, dy):  # pragma: no cover
        """
        Double-integral of X **FOR SYMMETRIC FUNCTIONS**.
        """
        nx = X.shape[-2]
        ny = X.shape[-1]

        W = makeW_(nx, ny)  # only upper

        tot = np.zeros_like(X[..., 0, 0])
        for ix in range(nx):
            tot += W[ix, ix] * X[..., ix, ix]
            for iy in range(ix + 1, ny):
                tot += 2 * W[ix, iy] * X[..., ix, iy]

        return dx * dy * tot / 9.0

    @jit(nopython=True)
    def makeW_(nx, ny):  # pragma: no cover
        r"""
        Return a window matrix for symmetric double-intergral.
        """
        W = np.ones((nx, ny))
        if nx % 2 == 0:
            for ix in range(1, nx - 2, 2):
                W[ix, -1] *= 4
                W[-1, ix] *= 4
                for iy in range(ny - 1):
                    W[ix, iy] *= 4
                    W[iy, ix] *= 4

            for ix in range(2, nx - 2, 2):
                W[ix, -1] *= 2
                W[-1, ix] *= 2
                for iy in range(ny - 1):
                    W[ix, iy] *= 2
                    W[iy, ix] *= 2

            for ix in range(nx):
                W[ix, -2] *= 2.5
                W[ix, -1] *= 1.5
                W[-2, ix] *= 2.5
                W[-1, ix] *= 1.5
        else:
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
    def makeH_(nx, ny):  # pragma: no cover
        """Return the window matrix for trapezoidal intergral."""
        H = np.ones((nx, ny))
        for ix in range(1, nx - 1):
            for iy in range(ny):
                H[ix, iy] *= 2
                H[iy, ix] *= 2

        return H

    @jit(nopython=True)
    def dbltrapz_(X, dx, dy):  # pragma: no cover
        """Double-integral of X for the trapezoidal method."""
        nx = X.shape[-2]
        ny = X.shape[-1]

        H = makeH_(nx, ny)
        tot = np.zeros_like(X[..., 0, 0])
        for ix in range(nx):
            tot += H[ix, ix] * X[ix, ix]
            for iy in range(ix + 1, ny):
                tot += 2 * H[ix, iy] * X[ix, iy]

        return dx * dy * tot / 4.0


# ===============================================================================
# Halo-Exclusion Models
# ===============================================================================
@pluggable
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

    def raw_integrand(self) -> np.ndarray:
        """
        Return either a 2d (k,m) or 3d (r,k,m) array with the general integrand.
        """
        if self.bias.ndim == 1:
            return self.Ifunc * self.bias * self.m  # *m since integrating in logspace
        else:
            return np.einsum("ij,kj->kij", self.Ifunc * self.m, self.bias)

    def integrate(self):
        """
        Integrate the :meth:`raw_integrand` over mass.

        This should pass back whatever is multiplied by P_m(k) to get the two-halo
        term. Often this will be a square of an integral, sometimes a Double-integral.
        """
        pass


class NoExclusion(Exclusion):
    r"""A model where there's no halo exclusion."""

    def integrate(self):
        """Integrate the :meth:`raw_integrand` over mass."""
        return intg.simps(self.raw_integrand(), dx=self.dlnx) ** 2


class Sphere(Exclusion):
    r"""Spherical halo exclusion model.

    Only halo pairs where the virial radius of
    either halo is smaller than half of the seperation, i.e.:

    .. math:: R_{\rm vir} \le r/2

    will be accounted for.
    """

    def raw_integrand(self):
        """
        Return either a 2d (k,m) or 3d (r,k,m) array with the general integrand.
        """
        if self.bias.ndim == 1:
            # *m since integrating in logspace
            return outer(np.ones_like(self.r), self.Ifunc * self.bias * self.m)
        else:
            return np.einsum("ij,kj->kij", self.Ifunc * self.m, self.bias)

    @cached_property
    def density_mod(self):
        """The modified density, under new limits."""
        density = np.outer(np.ones_like(self.r), self.density * self.m)
        density[self.mask] = 0
        return intg.simps(density, dx=self.dlnx, even="first")

    @cached_property
    def mask(self):
        """Elements that should be set to zero."""
        return (np.outer(self.m, np.ones_like(self.r)) > self.mlim).T

    @property
    def mlim(self):
        """The mass threshold for the mask."""
        return 4 * np.pi * (self.r / 2) ** 3 * self.mean_density * self.delta_halo / 3

    def integrate(self):
        """
        Integrate the :meth:`raw_integrand` over mass.
        """
        integ = self.raw_integrand()  # r,k,m
        integ.transpose((1, 0, 2))[:, self.mask] = 0
        return intg.simps(integ, dx=self.dlnx, even="first") ** 2


class DblSphere(Sphere):
    r"""Double Sphere model of halo exclusion.

    Only halo pairs for which the sum of virial radii
    is smaller than the separation, i.e.:

    .. math:: R_{\rm vir,1}+R_{\rm vir,2} \le r

    will be accounted for.
    """

    @property
    def r_halo(self):
        """The virial radius of the halo"""
        return (3 * self.m / (4 * np.pi * self.delta_halo * self.mean_density)) ** (
            1.0 / 3.0
        )

    @cached_property
    def mask(self):
        """Elements that should be set to zero (r,m,m)."""
        rvir = self.r_halo
        return (outer(np.add.outer(rvir, rvir), np.ones_like(self.r)) > self.r).T

    @cached_property
    def density_mod(self):
        """The modified density, under new limits."""
        out = np.zeros_like(self.r)
        for i, r in enumerate(self.r):
            integrand = np.outer(self.density * self.m, np.ones_like(self.density))
            integrand[self.mask[i]] = 0
            out[i] = intg.simps(
                intg.simps(integrand, dx=self.dlnx, even="first"),
                dx=self.dlnx,
                even="first",
            )
        return np.sqrt(out)

    def integrate(self):
        """
        Integrate the :meth:`raw_integrand` over mass.
        """
        integ = self.raw_integrand()  # (r,k,m)
        return integrate_dblsphere(integ, self.mask, self.dlnx)


def integrate_dblsphere(integ, mask, dx):
    """
    Integration function for double sphere model.
    """
    out = np.zeros_like(integ[:, :, 0])
    integrand = np.zeros_like(mask, dtype=float)
    for ik in range(integ.shape[1]):
        for ir in range(mask.shape[0]):
            integrand[ir] = np.outer(integ[ir, ik, :], integ[ir, ik, :])
        integrand[mask] = 0
        out[:, ik] = intg.simps(
            intg.simps(integrand, dx=dx, even="first"), dx=dx, even="first"
        )
    return out


if USE_NUMBA:

    @jit(nopython=True)
    def integrate_dblsphere_(integ, mask, dx):  # pragma: no cover
        r"""
        The same as :func:`integrate_dblsphere`, but uses NUMBA to speed up.
        """
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

    class DblSphere_(DblSphere):  # pragma: no cover
        r"""
        The same as :class:`DblSphere`. But uses NUMBA to speed up the integration.
        """

        def integrate(self):
            """Integrate the :meth:`raw_integrand` over mass."""
            integ = self.raw_integrand()  # (r,k,m)
            return integrate_dblsphere_(integ, self.mask, self.dlnx)


class DblEllipsoid(DblSphere):
    r"""
    Double Ellipsoid model of halo exclusion.

    Assuming a lognormal distribution
    of ellipticities for halos, the probability of halo pairs **not** excluded
    is:

    .. math:: P(y) = 3 y^2 - 2 y^3 ,\; y = (x-0.8)/0.29,\; x = r/(R_{\rm vir,1}+R_{\rm vir,2})

    taken from [1]_.



    References
    ----------
    .. [1]  Tinker, J. et al., " On the Mass-to-Light Ratio of Large-Scale Structure",
            https://ui.adsabs.harvard.edu/abs/2005ApJ...631...41T.
    """

    @cached_property
    def mask(self):
        "Unecessary for this approach."
        return None

    @cached_property
    def prob(self):
        """
        The probablity distribution used in calculating double integral.
        """
        rvir = self.r_halo
        x = outer(self.r, 1 / np.add.outer(rvir, rvir))
        x = (x - 0.8) / 0.29  # this is y but we re-use the memory
        np.clip(x, 0, 1, x)
        return 3 * x ** 2 - 2 * x ** 3

    @cached_property
    def density_mod(self):
        """The modified density, under new limits."""
        integrand = self.prob * outer(
            np.ones_like(self.r), np.outer(self.density * self.m, self.density * self.m)
        )
        return np.sqrt(dbltrapz(integrand, self.dlnx))

    def integrate(self):
        """
        Integrate the :meth:`raw_integrand` over mass.
        """
        integ = self.raw_integrand()  # (r,k,m)
        out = np.zeros_like(integ[:, :, 0])

        integrand = np.zeros_like(self.prob)
        for ik in range(integ.shape[1]):

            for ir in range(len(self.r)):
                integrand[ir] = self.prob[ir] * np.outer(
                    integ[ir, ik, :], integ[ir, ik, :]
                )
            out[:, ik] = intg.simps(intg.simps(integrand, dx=self.dlnx), dx=self.dlnx)
        return out


if USE_NUMBA:

    class DblEllipsoid_(DblEllipsoid):  # pragma: no cover
        r"""
        The same as :class:`DblEllipsoid`. But uses NUMBA to speed up the integration.
        """

        @cached_property
        def density_mod(self):  # pragma: no cover
            """The modified density, under new limits."""
            return density_mod_(
                self.r,
                self.r_halo,
                np.outer(self.density * self.m, self.density * self.m),
                self.dlnx,
            )

        @cached_property
        def prob(self):  # pragma: no cover
            """
            The probablity distribution used in calculating double integral
            """
            return prob_inner_(self.r, self.r_halo)

        def integrate(self):  # pragma: no cover
            """
            Integrate the :meth:`raw_integrand` over mass.
            """
            return integrate_dblell(self.raw_integrand(), self.prob, self.dlnx)

    @jit(nopython=True)
    def integrate_dblell(integ, prob, dx):  # pragma: no cover
        r"""Double Integration via the trapezoidal method if using NUMBA"""
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
    def density_mod_(r, rvir, densitymat, dx):  # pragma: no cover
        """The modified density, under new limits."""
        d = np.zeros(len(r))
        for ir, rr in enumerate(r):
            integrand = prob_inner_r_(rr, rvir) * densitymat
            d[ir] = dbltrapz_(integrand, dx, dx)
        return np.sqrt(d)

    @jit(nopython=True)
    def prob_inner_(r, rvir):  # pragma: no cover
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
    def prob_inner_r_(r, rvir):  # pragma: no cover
        """
        Jit-compiled version of calculating prob along one r,
        taking advantage of symmetry.
        """
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
    r"""
    A model for double ellipsoid halo exclusion, where a mask is
    defined so that the number density of galaxies is matched.
    """

    @cached_property
    def mask(self):
        """Mask Function for matching density"""
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
        """
        Integrate the :meth:`raw_integrand` over mass.
        """
        integ = self.raw_integrand()  # r,k,m
        integ.transpose((1, 0, 2))[:, self.mask] = 0
        return intg.simps(integ, dx=self.dlnx) ** 2


if USE_NUMBA:

    class NgMatched_(DblEllipsoid_):  # pragma: no cover
        r"""
        The same as :class:`NgMatched`. But uses NUMBA to speed up the integration.
        """

        @cached_property
        def mask(self):
            """Mask Function for matching density"""
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
            """
            Integrate the :meth:`raw_integrand` over mass.
            """
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
