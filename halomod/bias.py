'''
Created on Oct 1, 2013

@author: Steven
'''
import numpy as np

class Bias(object):
    '''
    Contains all bias functions
    '''


    def __init__(self, hmf, bias_model="ST"):
        '''
        Picks a bias function for you :)
        '''
        self.hmf = hmf
        self.dc = hmf.cosmo.delta_c
        self.nu = (hmf.cosmo.delta_c / hmf.sigma) ** 2

        if bias_model == "ST":
            self.bias = self._bias_st()
        if bias_model == 'seljak':
            self.bias = self._bias_seljak()
        elif bias_model == "ma":
            self.bias = self._bias_ma()
        elif bias_model == "seljak_warren":
            self.bias = self._bias_sw()
        elif bias_model == "Tinker05":
            self.bias = self._bias_tinker05()
        elif bias_model == "tinker10":
            self.bias = self._bias_tinker10()

    def _bias_st(self):
        """
        Sheth-Tormen bias
        """
        q = 0.707
        p = 0.3

        return 1 + (q * self.nu - 1) / self.dc + (2 * p / self.dc) / (1 + (q * self.nu) ** p)

    def _bias_seljak(self):
        """
        Large scale bias b(M)
        """
        return 1 + (self.nu - 1) / self.dc + 0.6 / (self.dc * (1 + (0.707 * self.nu) ** 0.3))

    def _bias_ma(self):
        return (1 + (self.nu - 1) / self.dc) * (1 / (2 * self.nu ** 2) + 1) ** (0.06 - 0.02 * self.hmf.cosmo.n)

    def _bias_sw(self):
        raise ValueError("Seljak-Warren Not Implemented Yet")
#        siginv = spline(self.sigma, self.M)
#        mstar = siginv(dc)
#        x = self.M / mstar
#        bias_ls = 0.53 + 0.39 * x ** 0.45 + 0.13 / (40 * x + 1) + 5E-4 * x ** 1.5
#
#        #TODO: what the monkeys is alpha_s??
#        return bias_ls + np.log10(x) * (0.4 * (self.cosmo_params['omegam'] - 0.3 + self.cosmo_params['n'] - 1) + 0.3 * (self.cosmo_params['sigma_8'] - 0.9 + self.cosmo_params['H0'] / 100 - 0.7) + 0.8)

    def _bias_tinker05(self):
        a = 0.707
        sa = np.sqrt(a)
        b = 0.35
        c = 0.8

        return 1 + 1 / (sa * self.dc) * (sa * (a * self.nu) + sa * b * (a * self.nu) ** (1 - c) - (a * self.nu) ** c / ((a * self.nu) ** c + b * (1 - c) * (1 - c / 2)))

    def _bias_tinker10(self):
        nu = np.sqrt(self.nu)
        y = np.log10(self.hmf.delta_halo)
        A = 1.0 + 0.24 * y * np.exp(-(4 / y) ** 4)
        a = 0.44 * y - 0.88
        B = 0.183
        b = 1.5
        C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y) ** 4)
        c = 2.4

#         print y, A, a, B, b, C, c
        return 1 - A * nu ** a / (nu ** a + self.dc ** a) + B * nu ** b + C * nu ** c

    def bias_scale(self, xi_dm):
        """
        Scale-dependent bias
        """
        return np.sqrt(self.bias ** 2 * (1 + 1.17 * xi_dm) ** 1.49 / (1 + 0.69 * xi_dm) ** 2.09)
