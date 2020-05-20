"""
Module for defining halo bias models.
"""

import numpy as np
from hmf._framework import Component
from scipy.interpolate import InterpolatedUnivariateSpline as spline


class Bias(Component):
    """
    Base class for all Bias models.

    At this point, we only implement first-order, local, deterministic bias.

    Most models are specified in terms of the peak-height parameter,
    though it is possible to specify them in terms of mass, and include
    cosmological parameters.

    All subclasses must implement the method ``bias``, which returns the Bias
    function at `nu` or `m`.
    """

    _defaults = {}

    def __init__(
        self,
        nu,
        delta_c=1.686,
        m=None,
        mstar=None,
        delta_halo=None,
        n=None,
        Om0=None,
        sigma_8=None,
        h=None,
        **model_parameters
    ):
        """
        Initaliser for the class.

        Parameters
        ----------
        nu : nd-array
            Peak-height, delta_c/sigma
        delta_c : float, optional
            Critical overdensity for collapse.
        m : array
            Vector of halo masses corresponding to `nu`
        mstar : float, optional
            Nonlinear mass, sigma(mstar) = delta_c
        delta_halo : float, optional
        n
        Om0
        sigma_8
        h
        model_parameters
        """
        self.nu = nu
        self.n = n
        self.delta_c = delta_c
        self.delta_halo = delta_halo
        self.m = m
        self.mstar = mstar

        self.n = n
        self.h = h
        self.Om0 = Om0
        self.sigma_8 = sigma_8

        super(Bias, self).__init__(**model_parameters)

    def bias(self):
        return np.ones_like(self.nu)


class UnityBias(Bias):
    """A toy model which is exactly unity for all mass"""

    def bias(self):
        return np.ones_like(self.nu)


class Mo96(Bias):
    """
    Peak-background split bias corresponding to PS HMF.

    Taken from Mo and White (1996)
    """

    def bias(self):
        return 1 + (self.nu - 1) / self.delta_c


class Jing98(Bias):
    """
    Empirical bias of Jing (1998): http://adsabs.harvard.edu/abs/1998ApJ...503L...9J
    """

    _defaults = {"a": 0.5, "b": 0.06, "c": 0.02}

    def bias(self):
        nu = (self.m / self.mstar) ** (self.n + 3) / 6
        a = self.params["a"]
        b = self.params["b"]
        c = self.params["c"]
        return (a / nu ** 4 + 1) ** (b - c * self.n) * (
            1 + (nu ** 2 - 1) / self.delta_c
        )


class ST99(Bias):
    """
    Peak-background split bias corresponding to ST99 HMF.

    Taken from Sheth & Tormen (1999).
    """

    _defaults = {"q": 0.707, "p": 0.3}

    def bias(self):
        p = self.params["p"]
        q = self.params["q"]
        return (
            1
            + (q * self.nu - 1) / self.delta_c
            + (2 * p / self.delta_c) / (1 + (q * self.nu) ** p)
        )


class SMT01(Bias):
    """
    Extended Press-Schechter-derived bias function corresponding to SMT01 HMF

    Taken from Sheth, Mo & Tormen (2001)
    """

    _defaults = {"a": 0.707, "b": 0.5, "c": 0.6}

    def bias(self):
        nu = self.nu
        a = self.params["a"]
        sa = np.sqrt(a)
        b = self.params["b"]
        c = self.params["c"]
        return 1 + (
            sa * (a * nu)
            + sa * b * (a * nu) ** (1 - c)
            - (a * nu) ** c / ((a * nu) ** c + b * (1 - c) * (1 - c / 2))
        ) / (self.delta_c * sa)


class Seljak04(Bias):
    """
    Empirical bias relation from Seljak & Warren (2004), without cosmological dependence
    """

    _defaults = {
        "a": 0.53,
        "b": 0.39,
        "c": 0.45,
        "d": 0.13,
        "e": 40,
        "f": 5e-4,
        "g": 1.5,
    }

    def bias(self):
        a = self.params["a"]
        b = self.params["b"]
        c = self.params["c"]
        d = self.params["d"]
        e = self.params["e"]
        f = self.params["f"]
        g = self.params["g"]
        x = self.m / self.mstar
        return a + b * x ** c + d / (e * x + 1) + f * x ** g


class Seljak04_Cosmo(Seljak04):
    """
    Empirical bias relation from Seljak & Warren (2004), with cosmological dependence.

    Doesn't include the running of the spectral index alpha_s.
    """

    _defaults = {
        "a": 0.53,
        "b": 0.39,
        "c": 0.45,
        "d": 0.13,
        "e": 40,
        "f": 5e-4,
        "g": 1.5,
        "a1": 0.4,
        "a2": 0.3,
    }

    def bias(self):
        b = super(Seljak04, self).bias()
        a1 = self.params["a1"]
        a2 = self.params["a2"]
        return b + np.log10(self.m / self.mstar) * (
            a1 * (self.Om0 - 0.3 + self.n - 1)
            + a2 * (self.sigma_8 - 0.9 + self.h - 0.7)
        )


class Tinker05(SMT01):
    """
    Empirical bias, same as SMT01 but modified parameters.
    """

    _defaults = {"a": 0.707, "b": 0.35, "c": 0.8}


class Mandelbaum05(ST99):
    """
    Empirical bias, same as ST99 but changed parameters
    """

    _defaults = {"q": 0.73, "p": 0.15}


class Pillepich10(Bias):
    """
    Empirical bias of Pillepich et. al. (2010) [Gaussian case]

    Re-parameterised in terms of nu.
    """

    _defaults = {"B0": 0.647, "B1": -0.320, "B2": 0.568}

    def bias(self):
        nu = self.nu
        B0 = self.params["B0"]
        B1 = self.params["B1"]
        B2 = self.params["B2"]
        return B0 + B1 * np.sqrt(nu) + B2 * nu


class Manera10(ST99):
    """
    Peak-background split bias from Manera et al. (2010)
    """

    _defaults = {"q": 0.709, "p": 0.248}


class Tinker10(Bias):
    """
    Empirical bias of Tinker et al. (2010)
    """

    _defaults = {"B": 0.183, "b": 1.5, "c": 2.4}

    def bias(self):
        y = np.log10(self.delta_halo)
        A = 1.0 + 0.24 * y * np.exp(-((4 / y) ** 4))
        a = 0.44 * y - 0.88
        C = 0.019 + 0.107 * y + 0.19 * np.exp(-((4 / y) ** 4))
        nu = np.sqrt(self.nu)
        B = self.params["B"]
        c = self.params["c"]
        b = self.params["b"]
        return (
            1 - A * nu ** a / (nu ** a + self.delta_c ** a) + B * nu ** b + C * nu ** c
        )


class Tinker10_PBsplit(Bias):
    _defaults = {  # --- alpha
        "alpha_200": 0.368,
        "alpha_300": 0.363,
        "alpha_400": 0.385,
        "alpha_600": 0.389,
        "alpha_800": 0.393,
        "alpha_1200": 0.365,
        "alpha_1600": 0.379,
        "alpha_2400": 0.355,
        "alpha_3200": 0.327,
        # --- beta
        "beta_200": 0.589,
        "beta_300": 0.585,
        "beta_400": 0.544,
        "beta_600": 0.543,
        "beta_800": 0.564,
        "beta_1200": 0.623,
        "beta_1600": 0.637,
        "beta_2400": 0.673,
        "beta_3200": 0.702,
        # --- gamma
        "gamma_200": 0.864,
        "gamma_300": 0.922,
        "gamma_400": 0.987,
        "gamma_600": 1.09,
        "gamma_800": 1.2,
        "gamma_1200": 1.34,
        "gamma_1600": 1.5,
        "gamma_2400": 1.68,
        "gamma_3200": 1.81,
        # --- phi
        "phi_200": -0.729,
        "phi_300": -0.789,
        "phi_400": -0.910,
        "phi_600": -1.05,
        "phi_800": -1.2,
        "phi_1200": -1.26,
        "phi_1600": -1.45,
        "phi_2400": -1.5,
        "phi_3200": -1.49,
        # -- eta
        "eta_200": -0.243,
        "eta_300": -0.261,
        "eta_400": -0.261,
        "eta_600": -0.273,
        "eta_800": -0.278,
        "eta_1200": -0.301,
        "eta_1600": -0.301,
        "eta_2400": -0.319,
        "eta_3200": -0.336,
        # --others
        "beta_exp": 0.2,
        "phi_exp": -0.08,
        "eta_exp": 0.27,
        "gamma_exp": -0.01,
        "max_z": 3,
    }

    delta_virs = np.array([200, 300, 400, 600, 800, 1200, 1600, 2400, 3200])

    def bias(self):
        if self.delta_halo not in self.delta_virs:
            beta_array = np.array([self.params["beta_%s" % d] for d in self.delta_virs])
            gamma_array = np.array(
                [self.params["gamma_%s" % d] for d in self.delta_virs]
            )
            phi_array = np.array([self.params["phi_%s" % d] for d in self.delta_virs])
            eta_array = np.array([self.params["eta_%s" % d] for d in self.delta_virs])

            beta_func = spline(self.delta_virs, beta_array)
            gamma_func = spline(self.delta_virs, gamma_array)
            phi_func = spline(self.delta_virs, phi_array)
            eta_func = spline(self.delta_virs, eta_array)

            beta_0 = beta_func(self.delta_halo)
            gamma_0 = gamma_func(self.delta_halo)
            phi_0 = phi_func(self.delta_halo)
            eta_0 = eta_func(self.delta_halo)
        else:
            beta_0 = self.params["beta_%s" % (int(self.delta_halo))]
            gamma_0 = self.params["gamma_%s" % (int(self.delta_halo))]
            phi_0 = self.params["phi_%s" % (int(self.delta_halo))]
            eta_0 = self.params["eta_%s" % (int(self.delta_halo))]

        beta = (
            beta_0 * (1 + min(self.z, self.params["max_z"])) ** self.params["beta_exp"]
        )
        phi = phi_0 * (1 + min(self.z, self.params["max_z"])) ** self.params["phi_exp"]
        eta = eta_0 * (1 + min(self.z, self.params["max_z"])) ** self.params["eta_exp"]
        gamma = (
            gamma_0
            * (1 + min(self.z, self.params["max_z"])) ** self.params["gamma_exp"]
        )

        return (
            1
            + (gamma * self.nu - (1 + 2 * eta)) / self.delta_c
            + 2 * phi / self.delta_c / (1 + (beta ** 2 * self.nu) ** phi)
        )


class ScaleDepBias(Component):
    """
    Base class for scale-dependent bias models.
    """

    def __init__(self, xi_dm, **model_parameters):
        self.xi_dm = xi_dm
        super(ScaleDepBias, self).__init__(**model_parameters)

    def bias_scale(self):
        pass


class Tinker_SD05(ScaleDepBias):
    """
    Scale-dependent bias, Tinker 2005
    """

    _defaults = {"a": 1.17, "b": 1.49, "c": 0.69, "d": 2.09}

    def bias_scale(self):
        a = self.params["a"]
        b = self.params["b"]
        c = self.params["c"]
        d = self.params["d"]
        return np.sqrt((1 + a * self.xi_dm) ** b / (1 + c * self.xi_dm) ** d)
