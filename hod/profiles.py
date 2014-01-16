
import numpy as np
import scipy.special as sp

class profiles(object):
    """
    Halo radial density profiles.
    
    This class deals with halo density profiles and their normalised 
    fourier-transform pairs, along with concentration-mass relations.
    
    Each density profile is a function of r,m,z. each fourier pair is a function of
    k,m,z and each mass relation is a function of mass, and also outputs r_s if
    desired.
    
    .. note :: Currently we are only implementing the NFW profile. We could 
            implement other profiles, but it is unclear what the necessary
            concentration-mass relation would be.
    """
    def __init__(self, mean_dens=2.775E11, delta_halo=200.0, profile='nfw',
                 cm_relation='zehavi'):

        self.mean_dens = mean_dens
        self.delta_halo = delta_halo
        self._profile = profile
        self._cm_relation = cm_relation

    def mvir_to_rvir(self, m):
        return (3 * m / (4 * np.pi * self.delta_halo * self.mean_dens)) ** (1. / 3.)


    #===========================================================================
    # THE WRAPPING FUNCTIONS - THE ONLY ONES EVER CALLED
    #===========================================================================
    # It would be 'simpler' to set eg. self.rho = self.rho_nfw etc (for relevant
    # profile parameter), but we CANNOT do this because then the class can't
    # be pickled, which then means we can't go parallel!!
    def rho(self, r, m, z):
        """
        The density at radius r of a halo of mass m and redshift z
        """
        # First treat case in which r AND m are arrays - return a matrix
        if np.iterable(r) and np.iterable(m):
            result = np.zeros((len(r), len(m)))
            for i, rr in enumerate(r):
                if self._profile == "nfw":
                    result[i, :] = self.rho_nfw(rr, m, z)

            return result
        # Now the obvious case in which either one or none is an array
        else:
            if self._profile == 'nfw':
                return self.rho_nfw(r, m, z)


    def u(self, k, m, z):
        """
        The normalised fourier-transform of the density profile 
        """
        if np.iterable(k) and np.iterable(m):
            result = np.zeros((len(k), len(m)))
            for i, kk in enumerate(k):
                if self._profile == "nfw":
                    result[i, :] = self.u_nfw(kk, m, z)

            return result
        else:
            if self._profile == "nfw":
                return self.u_nfw(k, m, z)

    def lam(self, r, m, z):
        """
        The density profile convolved with itself
        """
        if np.iterable(r) and np.iterable(m):
            result = np.zeros((len(r), len(m)))
            for i, rr in enumerate(r):
                if self._profile == "nfw":
                    result[i, :] = self.lam_nfw(rr, m, z)

            return result
        else:
            if self._profile == "nfw":
                return self.lam_nfw(r, m, z)

    def cm_relation(self, m, z, get_rs):
        """
        The concentration-mass relation
        """
        if self._cm_relation == "duffy":
            return self.cm_duffy(m, z, get_rs)
        elif self._cm_relation == "zehavi":
            return self.cm_zehavi(m, z, get_rs)

    #===========================================================================
    # DEFINE NFW FUNCTIONS
    #===========================================================================
    def rho_nfw(self, r, m, z):

        c, r_s = self.cm_relation(m, z, get_rs=True)
        x = r / r_s
        rn = x / c

        if np.iterable(x):
            result = np.zeros_like(x)
            result[rn <= 1] = (self._dc_nfw(c) / (c * r_s) ** 3 / (x * (1 + x) ** 2))[rn <= 1]

            return result
        else:
            if rn <= 1.0:
                return self._dc_nfw(c) / (c * r_s) ** 3 / (x * (1 + x) ** 2)
            else:
                return 0.0

    def u_nfw(self, k, m, z):
        c, r_s = self.cm_relation(m, z, get_rs=True)

        K = k * r_s

        asi, ac = sp.sici((1 + c) * K)
        bs, bc = sp.sici(K)

        return (np.sin(K) * (asi - bs) - np.sin(c * K) / ((1 + c) * K) + np.cos(K) * (ac - bc)) / (np.log(1 + c) - c / (1 + c))

    def lam_nfw(self, r, m, z):
        c, r_s = self.cm_relation(m, z, get_rs=True)
        x = r / r_s  # x is such that nfw = 1/x*(1+x)^2

        if not np.iterable(x):
            x = np.array([x])
        if not np.iterable(c):
            # Could have iterable r rather than c which this fixes
            c = np.repeat(c, len(x))

        r = x / c  # r is here scaled by the virial radius

        result = np.zeros_like(x)

        if r.min() > 2:
            return result  # Stays as zero

        fa = self._dc_nfw(c) * 4 * np.pi / c ** 3
        f1 = (fa * m) ** 2 / (4 * np.pi * r_s ** 3)

        # GET LOW VALUES
        if r.min() < 1:
            mask = r <= 1
            x_lo = x[mask]
            c_lo = c[mask]
            a_lo = 1.0 / c_lo

            f2_lo = -4 * (1 + a_lo) + 2 * a_lo * x_lo * (1 + 2 * a_lo) + (a_lo * x_lo) ** 2
            f2_lo /= 2 * (x_lo * (1 + a_lo)) ** 2 * (2 + x_lo)
            f3_lo = np.log((1 + a_lo - a_lo * x_lo) * (1 + x_lo) / (1 + a_lo)) / x_lo ** 3
            f4 = np.log(1 + x_lo) / (x_lo * (2 + x_lo) ** 2)


            result[mask] = f1[mask] * (f2_lo + f3_lo + f4)

        if r.min() < 2 and r.max() > 1:
            mask = np.logical_and(r > 1, r <= 2)
            x_hi = x[mask]
            c_hi = c[mask]
            a_hi = 1.0 / c_hi

            f2_hi = np.log((1 + a_hi) / (a_hi + a_hi * x_hi - 1)) / (x_hi * (2 + x_hi) ** 2)
            f3_hi = (x_hi * a_hi ** 2 - 2 * a_hi) / (2 * x_hi * (1 + a_hi) ** 2 * (2 + x_hi))

            result[mask] = f1[mask] * (f2_hi + f3_hi)

        return result

    def _dc_nfw(self, c):
        return c ** 3 / (4 * np.pi) / (np.log(1 + c) - c / (1 + c))

    #===========================================================================
    # CONCENTRATION-MASS RELATIONS
    #===========================================================================
    def cm_duffy(self, m, z, get_rs=True):
        c = 6.71 * (m / (2.0 * 10 ** 12)) ** -0.091 * (1 + z) ** -0.44

        rvir = self.mvir_to_rvir(m)

        if get_rs:
            return c, rvir / c
        else:
            return c

    def cm_zehavi(self, m, z, get_rs=True):
        c = ((m / 1.5E13) ** -0.13) * 9.0 / (1 + z)
        rvir = self.mvir_to_rvir(m)

        if get_rs:
            return c, rvir / c
        else:
            return c



