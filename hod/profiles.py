"""
A few halo density profiles and their normalised  fourier-transform pairs, along with concentration-mass relations.

Each density profile is a function of r,m,z. each fourier pair is a function of k,m,z and each mass relation is a function of mass, and outputs r_s as well if you want it to.
"""
import numpy as np
import scipy.special as sp

class profiles(object):

    def __init__(self, mean_dens, delta_halo, profile='nfw', cm_relation='zehavi'):

        self.mean_dens = mean_dens
        self.delta_halo = delta_halo

        if profile == 'nfw':
            self.rho = self.rho_nfw
            self.u = self.u_nfw
            self.lam = self.lam_nfw

        if cm_relation == 'duffy':
            self.cm_relation = self.cm_duffy
        elif cm_relation == 'zehavi':
            self.cm_relation = self.cm_zehavi

    def mvir_to_rvir(self, m):

        return (3 * m / (4 * np.pi * self.delta_halo * self.mean_dens)) ** (1. / 3.)


    def rho_nfw(self, r, m, z):

        c, r_s = self.cm_relation(m, z, get_rs=True)

        x = r / r_s
#        if r <= 0.08 or r >= 5.0 and r <= 5.5:
#            for i, M in enumerate(m):
#                if (M < 2 * 10 ** 16 and M > 1.8E16) or (M < 1.5E12 and M > 1.0E12):
#                    print r, M , r_s[i], c[i], x[i], (self._dc_nfw(c) * 4 * np.pi / c ** 3)[i]

        r = x / c

        if np.iterable(x):
            result = np.zeros_like(x)
            try:
                result[r <= 1] = self._dc_nfw(c) / (c * r_s) ** 3 / (x[r <= 1] * (1 + x[r <= 1]) ** 2)
                return result
            except:
                result[r <= 1] = self._dc_nfw(c)[r <= 1] / ((c * r_s) ** 3)[r <= 1] / (x[r <= 1] * (1 + x[r <= 1]) ** 2)
                return result
        else:
            if r <= 1.0:
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
            #Could have iterable r rather than c which this fixes
            c = np.repeat(c, len(x))

        r = x / c  #r is here scaled by the virial radius

        result = np.zeros_like(x)

        if r.min() > 2:
            return result  #Stays as zero

        fa = self._dc_nfw(c) * 4 * np.pi / c ** 3
        f1 = (fa * m) ** 2 / (4 * np.pi * r_s ** 3)

        #GET LOW VALUES
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

#            for i, R in enumerate(r[mask]):
#                if (R <= 1.05 or R >= 1.95) and (m[mask][i] <= 1.5E12 or m[mask][i] >= 1.8E16):
#                    print R, m[mask][i], c_hi[i], f1[mask][i], f2_hi[i], f3_hi[i]  #, f4[i]

            result[mask] = f1[mask] * (f2_hi + f3_hi)

        return result

    def _dc_nfw(self, c):
        #We tentatively follow charles here...
        return c ** 3 / (4 * np.pi) / (np.log(1 + c) - c / (1 + c))

    def cm_duffy(self, m, z, get_rs=True):
        c = 6.71 * (m / (2.0 * 10 ** 12)) ** -0.091 * (1 + z) ** -0.44

        rvir = self.mvir_to_rvir(m)

        if get_rs:
            return c, rvir / c
        else:
            return c

    def cm_zehavi(self, m, z, get_rs=True):
        c = ((0.7 * m / 1.5E13) ** -0.13) * 9.0 / (1 + z)
        rvir = self.mvir_to_rvir(m)

        if get_rs:
            return c, rvir / c
        else:
            return c



