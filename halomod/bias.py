'''
Created on Oct 1, 2013

@author: Steven
'''
import numpy as np
import sys

_allmodels = ["ST", "seljak", 'ma', 'tinker05', 'tinker10']

def get_bias(model):
    """
    A function that chooses the correct bias function and returns it
    """
    try:
        return getattr(sys.modules[__name__], model)
    except AttributeError:
        raise
        raise AttributeError(str(model) + "  is not a valid bias model")

def ST(hmod, **params):
    """
    Sheth-Tormen bias
    """

    q = params.get("q", 0.707)
    p = params.get("p", 0.3)

    return 1 + (q * hmod.nu - 1) / hmod.delta_c + (2 * p / hmod.delta_c) / (1 + (q * hmod.nu) ** p)

def seljak(hmod):
    """
    Large scale bias b(M)
    """
    return 1 + (hmod.nu - 1) / hmod.delta_c + 0.6 / (hmod.delta_c * (1 + (0.707 * hmod.nu) ** 0.3))

def ma(hmod):
    return (1 + (hmod.nu - 1) / hmod.delta_c) * (1 / (2 * hmod.nu ** 2) + 1) ** (0.06 - 0.02 * hmod.n)

# def _bias_sw(hmod):
#     raise ValueError("Seljak-Warren Not Implemented Yet")
#        siginv = spline(hmod.sigma, hmod.M)
#        mstar = siginv(dc)
#        x = hmod.M / mstar
#        bias_ls = 0.53 + 0.39 * x ** 0.45 + 0.13 / (40 * x + 1) + 5E-4 * x ** 1.5
#
#        #TODO: what the monkeys is alpha_s??
#        return bias_ls + np.log10(x) * (0.4 * (hmod.cosmo_params['omegam'] - 0.3 + hmod.cosmo_params['n'] - 1) + 0.3 * (hmod.cosmo_params['sigma_8'] - 0.9 + hmod.cosmo_params['H0'] / 100 - 0.7) + 0.8)

def tinker05(hmod):
    a = 0.707
    sa = np.sqrt(a)
    b = 0.35
    c = 0.8
    return 1 + 1 / (sa * hmod.delta_c) * (sa * (a * hmod.nu) + sa * b * (a * hmod.nu) ** (1 - c) - (a * hmod.nu) ** c / ((a * hmod.nu) ** c + b * (1 - c) * (1 - c / 2)))

def tinker10(hmod):
    nu = np.sqrt(hmod.nu)
    y = np.log10(hmod.delta_halo)
    A = 1.0 + 0.24 * y * np.exp(-(4 / y) ** 4)
    a = 0.44 * y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y) ** 4)
    c = 2.4

#         print y, A, a, B, b, C, c
    return 1 - A * nu ** a / (nu ** a + hmod.delta_c ** a) + B * nu ** b + C * nu ** c

def bias_scale(hmod, xi_dm):
    """
    Scale-dependent bias, Tinker 2005
    """
    return np.sqrt(hmod.bias ** 2 * (1 + 1.17 * xi_dm) ** 1.49 / (1 + 0.69 * xi_dm) ** 2.09)
