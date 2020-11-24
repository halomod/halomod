r"""Module defining halo bias models.

The halo bias is defined as the ratio of the power spectrum of halo (centres) for halos
of a given mass, to the linear matter power spectrum. In particular, it is assumed for
the models defined here that the power spectrum of halo centres is merely a scalar multiple
of the linear matter power spectrum. That is, we implement first-order, local,
deterministic bias.

Bias models are defined as :class:`~hmf.Component` instances -- that is,
they are flexible models that the user can subclass and use in the halo model framework.
See :class:`Bias` for instructions on how to use ``Bias`` models. The following notes
will mostly describe how to subclass :class:`Bias` to define your own model.

Also provided are several models from the literature.

In addition, it defines a factory function :func:`make_colossus_bias`
which helps with integration with the ``colossus`` cosmology code. With this function,
the user is able to easily create a ``halomod``-compatible ``Component`` model that
transparently uses ``colossus`` the background to do the actual computation of the
halo bias. This means it is easy to use any of the updated models
from ``colossus`` in a native way.

Most models are specified in terms of the peak-height parameter,
though it is possible to specify them in terms of mass, and include
cosmological parameters.

To define your own bias model, subclass either :class:`Bias` or any of the in-built
models provided here. The only method required to be implemented is ``bias()``, which
takes no parameters. It must return the local first-order bias as an array of the
same shape as ``m`` (you also have access to the peak-height ``nu`` as an instance
variable). See documentation for :class:`Bias` for more information on the instance
variables available in the definition.

As with all ``Component`` subclasses, arbitrary user-specified variables can be received
by defining them in the `_defaults` class-level dictionary.

The module also defines a :class:`ScaleDependentBias`, which corrects the bias function
on different length scales.

Examples
--------
Define your own bias model that is unity for all masses (in fact this one is already built-in)::

    >>> class UnityBias(Bias):
    >>>    def bias(self):
    >>>        return np.ones_like(self.m)

Use this bias model in a halo model::

    >>> from halomod import HaloModel
    >>> hm = HaloModel(bias_model=UnityBias)
    >>> assert np.all(hm.bias == 1)

Constructing and using a colossus-based halo bias::

    >>> from halomod import HaloModel
    >>> from halomod.bias import make_colossus_bias
    >>> comparat = bias.make_colossus_bias(model="comparat17")
    >>> hm = HaloModel(bias_model=comparat)
"""
from typing import Optional

import numpy as np
from hmf import Component
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from hmf.cosmology.cosmo import astropy_to_colossus
from colossus.lss.bias import haloBiasFromNu
from astropy.cosmology import FLRW, Planck15
from hmf.halos.mass_definitions import SOMean
from hmf._internals import pluggable


@pluggable
class Bias(Component):
    r"""
    The base Bias component.

    This class should not be instantiated directly! Use a subclass that implements
    a specific bias model. The parameters listed below are the input parameters
    for *all* bias models. Extra model-specific parameters can be given -- these
    are documented in their respective class docstring.

    Parameters
    ----------
    nu : array-like
        Peak-height, ``delta^2_c/sigma^2``.
    delta_c : float, optional
        Critical over-density for collapse. Not all bias components require this
        parameter.
    m : array, optional
        Vector of halo masses corresponding to `nu`. Not all bias components require
        this parameter.
    mstar : float, optional
        Nonlinear mass, defined by the relation ``sigma(mstar) = delta_c``, with
        ``sigma`` the mass variance in spheres corresponding to virial radii of halos
        of mass ``mstar``.
    delta_halo : float, optional
        The over-density of halos with respect to the mean background matter density.
    n : float, optional
        The spectral index of the linear matter power spectrum..
    Om0 : float, optional
        The matter density, as a fraction of critical density, in the current universe.
    sigma_8 : float, optional
        The square root of the mass in spheres of radius 8 Mpc/h in the
        present day (normalizes the power spectrum).
    h : float, optional
        Hubble parameter in units of 100 km/s/Mpc.

    """

    _models = {}
    _defaults = {}

    def __init__(
        self,
        nu: np.ndarray,
        delta_c: float = 1.686,
        m: Optional[np.ndarray] = None,
        mstar: Optional[float] = None,
        delta_halo: Optional[float] = 200,
        n: Optional[float] = 1,
        sigma_8: Optional[float] = 0.8,
        cosmo: FLRW = Planck15,
        n_eff: [None, np.ndarray] = None,
        z: float = 0.0,
        **model_parameters
    ):
        self.nu = nu
        self.n = n
        self.delta_c = delta_c
        self.delta_halo = delta_halo
        self.m = m
        self.mstar = mstar
        self.z = z
        self.cosmo = cosmo
        self.h = cosmo.h
        self.Om0 = cosmo.Om0
        self.sigma_8 = sigma_8
        self.n_eff = n_eff

        super(Bias, self).__init__(**model_parameters)

    def bias(self) -> np.ndarray:
        """Calculate the first-order, linear, deterministic halo bias.

        Returns
        -------
        b : array-like
            The bias as a function of mass, as an array of values corresponding to the
            instance attributes `m` and/or `nu`.

        Examples
        --------
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from halomod.bias import Mo96
        >>> peak_height = np.linspace(0.1, 2, 100)
        >>> bias = Mo96(nu=peak_height)
        >>> plt.plot(peak_height, bias.bias())
        """
        return np.ones_like(self.nu)


class UnityBias(Bias):
    """A toy bias model which is exactly unity for all mass.

    See documentation for :class:`Bias` for information on input parameters. This
    model has no free parameters.
    """

    def bias(self):
        return np.ones_like(self.nu)


class Mo96(Bias):
    r"""
    Peak-background split bias corresponding to PS HMF.

    See documentation for :class:`Bias` for information on input parameters. This
    model has no free parameters.

    Notes
    -----
    This bias form can be explicitly derived by assuming a Press-Schechter form
    for the HMF, as shown for example in [1]_. The form is

    .. math:: 1 + \frac{(\nu - 1)}{\delta_c}

    References
    ----------
    .. [1] Mo, H. J. and White, S. D. M., "An analytic model for the spatial clustering
           of dark matter haloes", https://ui.adsabs.harvard.edu/abs/1996MNRAS.282..347M,
           1996
    """

    def bias(self):
        return 1 + (self.nu - 1) / self.delta_c


class Jing98(Bias):
    r"""
    Empirical bias of Jing (1998).

    See documentation for :class:`Bias` for information on input parameters. This
    model has no free parameters.

    Notes
    -----
    This is an empirical form proposed in [1]_, with the formula

    .. math:: (a/\nu^4 + 1)^{b - c n} \left(1 + \frac{\nu^2 - 1}{\delta_c}\right)

    The parameters ``a``, ``b`` and ``c`` are free parameters, with values fitted in
    [1]_ of ``(0.5, 0.06, 0.02)``, which are the defaults here.

    Other Parameters
    ----------------
    a,b,c : float
        The fitting parameters.

    References
    ----------
    .. [1] Jing, Y. P., "Accurate Fitting Formula for the Two-Point Correlation Function
           of Dark Matter Halos", http://adsabs.harvard.edu/abs/1998ApJ...503L...9J, 1998.
    """

    _defaults = {"a": 0.5, "b": 0.06, "c": 0.02}

    def bias(self):
        nu = self.nu

        a = self.params["a"]
        b = self.params["b"]
        c = self.params["c"]
        return (a / nu ** 2 + 1) ** (b - c * self.n_eff) * (1 + (nu - 1) / self.delta_c)


class ST99(Bias):
    r"""
    Peak-background split bias corresponding to ST99 HMF.

    See documentation for :class:`Bias` for information on input parameters. This
    model has no free parameters.

    Notes
    -----
    This bias form can be explicitly derived by assuming a Sheth-Tormen form
    for the HMF, as shown for example in [1]_. The form is

    .. math:: 1 + \frac{q\nu - 1}{\delta_c} + \frac{2p}{\delta_c ( 1 + q^p \nu^p)}

    with ``p`` and ``q`` having default values of ``(0.707, 0.3)``. They are free
    in this implementation for the user to modify.

    Other Parameters
    ----------------
    p,q : float, optional
        The free parameters of the form.

    References
    ----------
    .. [1] Sheth, R. K.. and Tormen, G., "Large-scale bias and the peak background
           split", https://ui.adsabs.harvard.edu/abs/1999MNRAS.308..119S, 1999
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
    r"""
    Extended Press-Schechter-derived bias function corresponding to SMT01 HMF.

    See documentation for :class:`Bias` for information on input parameters. This
    model has no free parameters.

    Notes
    -----
    This bias form can be explicitly derived by assuming a Sheth-Tormen form
    for the HMF and allowing for ellipsoidal collapse, as shown for example in [1]_.
    The form is

    .. math:: 1 + \frac{1}{\delta_c \sqrt{a}} \left(\sqrt{a} a \nu + \sqrt{a} b (a\nu)^{1-c}  - \frac{(a\nu)^c}{(a\nu)^c + b(1-c)(1 - c/2)}\right)

    with ``a``, ``b`` and ``c`` having default values of ``(0.707, 0.5, 0.6)``.
    They are free in this implementation for the user to modify.

    Other Parameters
    ----------------
    a,b,c : float, optional
        The free parameters of the form.

    References
    ----------
    .. [1] Sheth, R. K. and Tormen G., "Ellipsoidal collapse and an improved model for
           the number and spatial distribution of dark matter haloes",
           https://ui.adsabs.harvard.edu/abs/2001MNRAS.323....1S, 2001
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
    r"""
    Empirical bias relation from Seljak & Warren (2004), without cosmological dependence.

    See documentation for :class:`Bias` for information on input parameters. This
    model has no free parameters.

    Notes
    -----
    This the form from [1]_ *without* cosmological dependence. The form is

    .. math:: a + bx^c + \frac{d}{ex+1} + fx^g

    with :math:`x = m/m_\star` (and :math:`m_star` the nonlinear mass -- see :class:`Bias`
    for details). The other parameters are all fitted, with values given [1]_ as
    ``(a,b,c,d,e,f,g) = (0.53, 0.39, 0.45, 0.13, 40, 5e-4, 1.5)``.

    Other Parameters
    ----------------
    a,b,c,d,e,f,g : float, optional
        The fitted parameters.

    References
    ----------
    .. [1] Seljak, U. and Warren M. S., "Large-scale bias and stochasticity of haloes
           and dark matter", https://ui.adsabs.harvard.edu/abs/2004MNRAS.355..129S, 2004.
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


class Seljak04Cosmo(Seljak04):
    r"""
    Empirical bias relation from Seljak & Warren (2004), with cosmological dependence.

    See documentation for :class:`Bias` for information on input parameters. This
    model has no free parameters.

    Notes
    -----
    This the form from [1]_ *with* cosmological dependence -- except we do not include
    the running of the spectral index. The form is

    .. math:: b_{\rm no cosmo} + \log_10(x) \left[a_1 (\Omega_{m,0} - 0.3 + n - 1) + a_2(\sigma_8 - 0.9 + h-0.7)\right]

    with :math:`x = m/m_\star` (and :math:`m_{\star}` the nonlinear mass -- see :class:`Bias`
    for details). The non-cosmologically-dependent bias is that given by :class:`Seljak04`.
    ``a1`` and ``a2`` are fitted, with values given in [1]_ as
    ``(a1,a2) = (0.4, 0.3)``.

    Other Parameters
    ----------------
    a,b,c,d,e,f,g : float, optional
        The fitted parameters for :class:`Seljak04`.
    a1,a2 : float, optional
        Fitted parameters for the cosmological dependence.

    References
    ----------
    .. [1] Seljak, U. and Warren M. S., "Large-scale bias and stochasticity of haloes
           and dark matter", https://ui.adsabs.harvard.edu/abs/2004MNRAS.355..129S, 2004.
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
        b = super().bias()
        a1 = self.params["a1"]
        a2 = self.params["a2"]
        x = np.log10(self.m / self.mstar)
        x[x < -1] = -1
        return b + x * (
            a1 * (self.Om0 - 0.3 + self.n - 1)
            + a2 * (self.sigma_8 - 0.9 + self.h - 0.7)
        )


class Tinker05(SMT01):
    r"""
    Empirical bias from Tinker et al (2005).

    See documentation for :class:`Bias` for information on input parameters. This
    model has no free parameters.

    Notes
    -----
    This form is the same as that of :class:`SMT01`, however the parameters were re-fit
    to simulations in [1]_. Here the default parameters are ``(a,b,c) = (0.707, 0.35, 0.8)``.

    References
    ----------
    .. [1] Tinker J. et al., "On the Mass-to-Light Ratio of Large-Scale Structure",
           https://ui.adsabs.harvard.edu/abs/2005ApJ...631...41T, 2005
    """

    _defaults = {"a": 0.707, "b": 0.35, "c": 0.8}


class Mandelbaum05(ST99):
    r"""
    Empirical bias of Mandelbaum (2005).

    See documentation for :class:`Bias` for information on input parameters. This
    model has no free parameters.

    Notes
    -----
    This form is the same as that of :class:`SMT99`, however the parameters were re-fit
    to simulations from [1]_ in [2]_. Here the default parameters are ``(q,p) = (0.73, 0.15)``.

    References
    ----------
    .. [1] Seljak, U. and Warren M. S., "Large-scale bias and stochasticity of haloes
           and dark matter", https://ui.adsabs.harvard.edu/abs/2004MNRAS.355..129S, 2004.
    .. [2] Mandelbaum, R. et al., "Galaxy-galaxy lensing: dissipationless simulations
           versus the halo model", https://ui.adsabs.harvard.edu/abs/2005MNRAS.362.1451M, 2005.
    """

    _defaults = {"q": 0.73, "p": 0.15}


class Pillepich10(Bias):
    r"""
    Empirical bias of Pillepich et al (2010).

    See documentation for :class:`Bias` for information on input parameters. This
    model has no free parameters.

    Notes
    -----
    This is the fit from [1]_, but it is the Gaussian case. The form is

    .. math:: B_0 + B_1 \sqrt{\nu} + B_2 \nu

    with :math:`\nu` the peak-height parameter. The values of the parameters fitted
    to simulation are given as ``(B0, B1, B2) = (0.647, -0.32, 0.568)``. They are left
    free to the user.

    Other Parameters
    ----------------
    B1, B2, B3 : float, optional
        The fitted parameters.

    References
    ----------
    .. [1]  Pillepich, A., Porciani, C. and Hahn, O., "Halo mass function and
            scale-dependent bias from N-body simulations with non-Gaussian initial
            conditions", https://ui.adsabs.harvard.edu/abs/2010MNRAS.402..191P, 2010
    """

    _defaults = {"B0": 0.647, "B1": -0.320, "B2": 0.568}

    def bias(self):
        nu = self.nu
        B0 = self.params["B0"]
        B1 = self.params["B1"]
        B2 = self.params["B2"]
        return B0 + B1 * np.sqrt(nu) + B2 * nu


class Manera10(ST99):
    r"""
    Peak-background split bias from Manera et al. (2010) [1]_.

    See documentation for :class:`Bias` for information on input parameters. This
    model has no free parameters.

    Other Parameters
    ----------------
    q, p : float, optional
        The fitted parameters.

    Notes
    -----
    .. note:: This form from [1]_ has the same form as :class:`ST99`, but has refitted the parameters with ``(q, p) = (0.709, 0.2)``.

    References
    ----------
    .. [1]  Manera M., Sheth,R. K. and Scoccimarro R., "Large-scale bias and the
            inaccuracy of the peak-background split ",
            https://ui.adsabs.harvard.edu/abs/2010MNRAS.402..589M, 2010
    """

    _defaults = {"q": 0.709, "p": 0.248}


class Tinker10(Bias):
    r"""
    Empirical bias of Tinker et al (2010).

    See documentation for :class:`Bias` for information on input parameters. This
    model has no free parameters.

    Notes
    -----
    This is an empirical form that does not obey the peak-background split consistency
    formalism, but fits well to simulations. It is dependent on the spherical halo
    definition. The form from [1]_ is

    .. math:: 1 - A\frac{\nu^a}{\nu^a + \delta_c^a} + B \nu^b + C \nu^c

    with

    .. math:: A = 1 + 0.24 y e^{-(4/y)^4},

    and

    .. math:: a = 0.44y - 0.88

    and

    .. math:: C = 0.019 + 0.107y + 0.19 e^{-(4/y)^4}

    and :math:`y=\log_{10} \Delta_{\rm halo}`.

    The fitted parameters are ``(B,b,c) = (0.183, 1.5, 2.4)``.

    Other Parameters
    ----------------
    B,b,c : float, optional
        The fitted parameters.

    References
    ----------
    .. [1] Tinker, J. L. et al., "The Large-scale Bias of Dark Matter Halos:
           Numerical Calibration and Model Tests",
           https://ui.adsabs.harvard.edu/abs/2010ApJ...724..878T, 2010

    See Also
    --------
    :class:`Tinker10PBsplit`
        Bias from the same study but with the constraint of the peak-background
        split formalism.
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


class Tinker10PBSplit(Bias):
    r"""
    Empirical bias of Tinker et al (2010).

    See documentation for :class:`Bias` for information on input parameters. This
    model has no free parameters.

    Notes
    -----
    This is form from [1]_ obeys the peak-background split consistency
    formalism, which offers some advantages, but also fits well to simulations.
    It is dependent on the spherical halo definition. See the reference for details
    on the form.

    Other Parameters
    ----------------
    alpha, beta,gamma,phi, eta : float, optional
        The fitted parameters. Each of these are available to specify at a certain
        overdensity. So for example ``alpha_200`` specifies the ``alpha`` parameter
        at a spherical halo overdensity of 200. All default values are taken from
        Tinker 2010.
    beta_exp, phi_exp, eta_exp, gamma_exp: float, optional
        The value of ``beta``, ``phi`` etc., are functions of redshift via the relation
        ``beta = beta0 (1 + z)^beta_exp`` (and likewise for the other parameters).
    max_z : float, optional
        The maximum redshift for which the redshift evolution holds. Above this
        redshift, the relation flattens. Default 3.

    References
    ----------
    .. [1] Tinker, J. L. et al., "The Large-scale Bias of Dark Matter Halos:
           Numerical Calibration and Model Tests",
           https://ui.adsabs.harvard.edu/abs/2010ApJ...724..878T, 2010

    See Also
    --------
    :class:`Tinker10`
        Bias from the same study but without the constraint of the peak-background
        split formalism.
    """
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


@pluggable
class ScaleDepBias(Component):
    r"""Base class for scale-dependent bias models.

    Parameters
    ----------
    xi_dm : np.ndarray
        The dark matter correlation function defined at some real-space scales, r.
    """

    def __init__(self, xi_dm: np.ndarray, **model_parameters):
        self.xi_dm = xi_dm
        super(ScaleDepBias, self).__init__(**model_parameters)

    def bias_scale(self) -> np.ndarray:
        """Return the scale dependent bias as a function of r.

        The scale-dependent bias is a function of the dark matter correlation
        function, and the length of the returned array should be the same size as the
        instance :attr:`xi_dm`.
        """
        pass


class TinkerSD05(ScaleDepBias):
    r"""Scale-dependent bias from Tinker 2005.

    Notes
    -----
    Defined in [1]_ as

    .. math:: \sqrt{\frac{(1 + a\xi)^b}{(1 + c\xi)^d}}

    with the fitted parameters ``(a,b,c,d)=(1.17, 1.49, 0.69, 2.09)``.

    References
    ----------
    .. [1] Tinker J. et al., "On the Mass-to-Light Ratio of Large-Scale Structure",
           https://ui.adsabs.harvard.edu/abs/2005ApJ...631...41T, 2005
    """

    _defaults = {"a": 1.17, "b": 1.49, "c": 0.69, "d": 2.09}

    def bias_scale(self):
        a = self.params["a"]
        b = self.params["b"]
        c = self.params["c"]
        d = self.params["d"]
        return np.sqrt((1 + a * self.xi_dm) ** b / (1 + c * self.xi_dm) ** d)


def make_colossus_bias(model="comparat17", mdef=SOMean(), **defaults):
    r"""
    A factory function which helps with integration with the ``colossus`` cosmology code.
    See :mod:`~halomod.bias` for an example of how to use it.

    Notice that it returns a *class* :class:`CustomColossusBias` not an instance.
    """

    class CustomColossusBias(Bias):
        _model_name = model
        _defaults = defaults
        _mdef = mdef

        def __init__(self, *args, **kwargs):
            super(CustomColossusBias, self).__init__(*args, **kwargs)
            astropy_to_colossus(self.cosmo, sigma8=self.sigma_8, ns=self.n)

        def bias(self):
            return haloBiasFromNu(
                nu=np.sqrt(self.nu),
                z=self.z,
                mdef=self._mdef.colossus_name,
                model=self._model_name,
                **self.params
            )

    CustomColossusBias.__name__ = model.capitalize()
    CustomColossusBias.__qualname__ = model.capitalize()

    return CustomColossusBias
