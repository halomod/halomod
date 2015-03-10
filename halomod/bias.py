'''
Created on Oct 1, 2013

@author: Steven
'''
import numpy as np
import sys
from hmf._framework import Model
_allmodels = ["ST", "seljak", 'ma', 'tinker05', 'tinker10']

class Bias(Model):
    def __init__(self, nu, delta_c, delta_halo, n, **model_parameters):
        self.nu = nu
        self.n = n
        self.delta_c = delta_c
        self.delta_halo = delta_halo
        super(Bias, self).__init__(**model_parameters)

    def bias(self):
        pass

class ST(Bias):
    """
    Sheth-Tormen bias
    """
    _defaults = {"q":0.707, "p":0.3}

    def bias(self):
        p = self.params['p']
        q = self.params['q']
        return 1 + (q * self.nu - 1) / self.delta_c + (2 * p / self.delta_c) / (1 + (q * self.nu) ** p)

class Seljak(ST):
    """
    Seljak bias
    """
    def bias(self):
        p = self.params['p']
        q = self.params['q']
        return 1 + (self.nu - 1) / self.delta_c + (2 * p / self.delta_c) / (1 + (q * self.nu) ** p)

class Ma(Bias):
    """
    Ma bias
    """
    _defaults = {"a":0.06, "b":0.02}

    def bias(self):
        a = self.params['a']
        b = self.params['b']
        return (1 + (self.nu - 1) / self.delta_c) * (1 / (2 * self.nu ** 2) + 1) ** (a - b * self.n)

class Tinker05(Bias):
    _defaults = {"a":0.707, "b":0.35, "c":0.8}

    def bias(self):
        a = self.params['a']
        sa = np.sqrt(a)
        b = self.params['b']
        c = self.params['c']
        return 1 + 1 / (sa * self.delta_c) * (sa * (a * self.nu) + sa * b * (a * self.nu) ** (1 - c) - (a * self.nu) ** c / ((a * self.nu) ** c + b * (1 - c) * (1 - c / 2)))

class Tinker10(Bias):
    _defaults = {"B":0.183, "b":1.5, "c":2.4}

    def bias(self):
        y = np.log10(self.delta_halo)
        A = 1.0 + 0.24 * y * np.exp(-(4 / y) ** 4)
        a = 0.44 * y - 0.88
        C = 0.019 + 0.107 * y + 0.19 * np.exp(-(4 / y) ** 4)
        nu = np.sqrt(self.nu)
        B = self.params['B']
        c = self.params['c']
        b = self.params['b']
        return 1 - A * nu ** a / (nu ** a + self.delta_c ** a) + B * nu ** b + C * nu ** c


class ScaleDepBias(Model):
    def __init__(self, xi_dm, **model_parameters):
        self.xi_dm = xi_dm
        super(ScaleDepBias, self).__init__(**model_parameters)

    def bias_scale(self):
        pass

class Tinker_SD05(ScaleDepBias):
    """
    Scale-dependent bias, Tinker 2005
    """
    _defaults = {"a":1.17, "b":1.49, "c":0.69, "d":2.09}

    def bias_scale(self):
        a = self.params['a']
        b = self.params['b']
        c = self.params['c']
        d = self.params['d']
        return np.sqrt((1 + a * self.xi_dm) ** b / (1 + c * self.xi_dm) ** d)
