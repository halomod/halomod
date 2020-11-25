"""
Module defining oncentration-mass relations.

This module defines a base :class:`CMRelation` component class, and a number of specific
concentration-mass relations. In addition, it defines a factory function :func:`make_colossus_cm`
which helps with integration with the ``colossus`` cosmology code. With this function,
the user is able to easily create a ``halomod``-compatible ``Component`` model that
transparently uses ``colossus`` the background to do the actual computation of the
concentration mass relation. This means it is easy to use any of the updated models
from ``colossus`` in a native way.

Examples
--------
A simple example of using a native concentration-mass relation::

>>> from halomod.concentration import Duffy08
>>> duffy = Duffy08()
>>> m = np.logspace(10, 15, 100)
>>> plt.plot(m, duffy.cm(m, z=0))

You can also specify a different concentration-mass relation for tracer
if you're working with :class:`~halomod.halo_model.TracerHaloModel` ::
    >>> from halomod import HaloModel
    >>> hm = HaloModel(halo_concentration_model='Ludlow16',
    >>>                tracer_concentration_model='Duffy08')

Constructing and using a colossus-based relation::

>>> from halomod.concentration import make_colossus_cm
>>> diemer = make_colossus_cm(model='diemer15', statistic='median')()
>>> plt.plot(m, diemer.cm(m, z=1))

Note the extra function call on the second line here -- :func:`make_colossus_cm` returns
a *class*, not an instance. Under the hood, any parameters passed to the function other
than ``model`` are set as "defaults", and can be modified like standard model params.
For instance, using such a model in a broader :class:`~HaloModel` framework::

>>> diemer19_cls = make_colossus_cm(model='diemer19', ps_args={})
>>> hm = HaloModel(
>>>     halo_concentration_model=diemer19_cls,
>>>     halo_concentration_params={'ps_args': {'model': 'eisenstein98'}}
>>> )
>>> hm.update(halo_concentration_params = {"ps_args": {"model": 'sugiyama95'}})

Note that while ``statistic`` is a valid argument to the `diemer19` model in COLOSSUS,
we have constructed it without access to that argument (and so it recieves its default
value of "median"). This means we *cannot* update it via the ``HaloModel`` interface.
"""
import warnings
from typing import Optional

import numpy as np
from hmf import Component
from scipy import special as sp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from astropy.cosmology import Planck15

from colossus.halo import concentration
from hmf.cosmology.cosmo import astropy_to_colossus

from hmf.density_field.filters import Filter
from hmf.cosmology.growth_factor import GrowthFactor
from .profiles import Profile, NFW
from hmf.cosmology.cosmo import Cosmology
from hmf.halos.mass_definitions import (
    MassDefinition,
    SOMean,
    SOVirial,
    SOCritical,
    from_colossus_name,
)
from hmf._internals import pluggable


@pluggable
class CMRelation(Component):
    r"""
    Base-class for Concentration-Mass relations
    """
    _pdocs = r"""

        Parameters
        ----------
        filter0 : :class:`hmf.filters.Filter` instance
            An instance of a filter function, with the power specified at z=0.
            Required for ``Bullock01``.
        growth : :class:`hmf.growth_factor.GrowthFactor` instance
            Specifies the growth function for the cosmology.
            Required for ``Bullock01``
        delta_c : float, optional
            Critical density for collapse
            Used in ``Bullock01``
        mstar : float, optional
            The nonlinear mass at the desired redshift.
            If not provided, will be calculated if required.
        \*\*model_parameters : unpacked-dictionary
            These parameters are model-specific. For any model, list the available
            parameters (and their defaults) using ``<model>._defaults``

        """
    __doc__ += _pdocs
    _defaults = {}

    native_mdefs = tuple()

    def __init__(
        self,
        cosmo: Cosmology = Cosmology(),
        filter0: Optional[Filter] = None,
        growth: Optional[GrowthFactor] = None,
        delta_c: float = 1.686,
        profile: Optional[Profile] = None,
        mdef: Optional[MassDefinition] = None,
        **model_parameters,
    ):
        # Save instance variables
        self.filter = filter0

        self.growth = GrowthFactor(cosmo=cosmo.cosmo) if growth is None else growth
        self.delta_c = delta_c

        self.mdef = self.native_mdefs[0] if mdef is None else mdef
        self.profile = NFW(self, mdef=self.mdef) if profile is None else profile

        self.cosmo = cosmo
        self.mean_density0 = cosmo.mean_density0

        # TODO: actually implement conversion of mass definitions.
        if self.mdef not in self.native_mdefs:
            warnings.warn(
                f"Requested mass definition '{mdef}' is not in native definitions for "
                f"the '{self.__class__.__name__}' CMRelation. No mass conversion will be "
                f"performed, so results will be wrong. Using '{self.mdef}'."
            )

        super(CMRelation, self).__init__(**model_parameters)

    def mass_nonlinear(self, z):
        """
        Return the nonlinear mass at z.

        Parameters
        ----------
        z : float
            Redshift. Must not be an array.
        """

        def model(lnr):
            return (
                self.filter.sigma(np.exp(lnr)) * self.growth.growth_factor(z)
                - self.delta_c
            ) ** 2

        res = minimize(model, [1.0,])

        if res.success:
            r = np.exp(res.x[0])
            return self.filter.radius_to_mass(
                r, self.mean_density0
            )  # TODO *(1+z)**3 ????
        else:
            warnings.warn("Minimization failed :(")
            return 0

    def cm(self, m, z=0):
        """
        Return concentration parameter for mass m at z.

        Parameters
        ----------
        z : float
            Redshift. Must not be an array.
        m : float
            Halo Mass.
        """
        pass


def make_colossus_cm(model="diemer15", **defaults):
    r"""
    A factory function which helps with integration with the ``colossus`` cosmology code.
    See :mod:`~halomod.concentration` for an example of how to use it.

    Notice that it returns a *class* :class:`CustomColossusCM` not an instance.
    """

    class CustomColossusCM(CMRelation):
        _model_name = model
        _defaults = defaults
        native_mdefs = tuple(
            from_colossus_name(d) for d in concentration.models[model].mdefs
        )

        def __init__(self, *args, **kwargs):
            super(CustomColossusCM, self).__init__(*args, **kwargs)
            # TODO: may want a more accurate way of passing sigma8 and ns here.
            astropy_to_colossus(self.cosmo.cosmo, sigma8=0.8, ns=1)

        def cm(self, m, z=0):
            return concentration.concentration(
                M=m,
                mdef=self.mdef.colossus_name,
                z=z,
                model=self._model_name,
                range_return=False,
                range_warning=True,
                **self.params,
            )

    CustomColossusCM.__name__ = model.capitalize()
    CustomColossusCM.__qualname__ = model.capitalize()

    return CustomColossusCM


class Bullock01(CMRelation):
    r"""
    Concentration-Mass relation of Bullock et al.(2001) [1]_.

    See documentation for :class:`Bias` for information on input parameters. This
    model has two model parameters.

    Notes
    -----
    The form of the concentration is

    .. math:: c_{\rm vir} = K a/a_c = K (1+z_c)/(1+z)

    The detailed description of the model can be found in [1]_.

    Other Parameters
    ----------------`
    F, K : float
        Default value is ``F=0.01`` and ``K=0.34``

    References
    ----------
    .. [1] Bullock, J.S. et al., " Profiles of dark haloes:
           evolution, scatter and environment ",
           https://ui.adsabs.harvard.edu/abs/1996MNRAS.282..347M.
    """
    _defaults = {"F": 0.01, "K": 3.4}
    native_mdefs = (SOCritical(),)

    def zc(self, m, z=0):
        r = self.filter.mass_to_radius(self.params["F"] * m, self.mean_density0)
        nu = self.filter.nu(r, self.delta_c)
        g = self.growth.growth_factor_fn(inverse=True)
        zc = g(np.sqrt(nu))
        zc[zc < z] = z  # hack?
        return zc

    def cm(self, m, z=0):
        return self.params["K"] * (self.zc(m, z) + 1.0) / (z + 1.0)


class Bullock01Power(CMRelation):
    r"""
    Extended Concentration-Mass relation of Bullock et al.(2001) [1]_.

    See documentation for :class:`Bias` for information on input parameters. This
    model has three model parameters.

    Notes
    -----
    The form of the concentration is

    ..math:: c_{\rm vir} = a/(1+z)^c\big(\frac{m}{m_s}\big)^b

    where a,b,c,ms are model parameters.

    Other Parameters
    ----------------`
    a, b, c: float
        Default value is ``a=9.0``, ``b=-0.13`` and ``c=1.0``.

    ms: float
        Default value is ``None``, where it's set to be the non-linear mass at z.

    References
    ----------
    .. [1] Bullock, J.S. et al., " Profiles of dark haloes:
           evolution, scatter and environment ",
           https://ui.adsabs.harvard.edu/abs/1996MNRAS.282..347M.
    """
    _defaults = {"a": 9.0, "b": -0.13, "c": 1.0, "ms": None}
    native_mdefs = (SOCritical(),)

    def _cm(self, m, ms, a, b, c, z=0):
        return a / (1 + z) ** c * (m / ms) ** b

    def cm(self, m, z=0):
        ms = self.params["ms"] or self.mass_nonlinear(z)
        return self._cm(m, ms, self.params["a"], self.params["b"], self.params["c"], z)


class Maccio07(CMRelation):
    """
    Concentration-Mass relation based on Maccio et al.(2007) [1]_.
    Default value taken from Padmanabhan et al.(2017) [2]_.

    References
    ----------
    .. [1] Maccio, A. V. et al., "Concentration, spin and shape of dark matter haloes:
           scatter and the dependence on mass and environment",
           https://ui.adsabs.harvard.edu/abs/2007MNRAS.378...55M.

    .. [2] Padmanabhan, H. et al., "A halo model for cosmological neutral hydrogen :
           abundances and clustering ",
           https://ui.adsabs.harvard.edu/abs/2017MNRAS.469.2323P/abstract.
    """

    _defaults = {"c_0": 28.65, "gamma": 1.45}
    native_mdefs = (SOMean(),)

    def cm(self, m, z):
        return (
            self.params["c_0"]
            * (m * 10 ** (-11)) ** (-0.109)
            * 4
            / (1 + z) ** self.params["gamma"]
        )


class Duffy08(Bullock01Power):
    r"""
    Concentration-mass relation from Duffy et al.(2008) [1]_.

    It has the same fomulae as :class:`Bullock01Power`,
    but with parameter values refitted.

    See documentation for :class:`Bias` for information on input parameters. This
    model has five model parameters.

    Notes
    -----

    .. note:: Only "NFW" parameters are implemented by default here. Of course, you can
              always pass your own parameters from Table 1 of [1]_.

    Other Parameters
    ----------------
    a, b, c: float
        Default is "NFW" parameters in [1]_.

    ms: float
        Default value is ``2e12``.

    sample : str
        Either "relaxed"(default) or "full". Specifies which set of parameters to take as
        default parameters, from Table 1 of [1]_.

    References
    ----------
    .. [1] Duffy, A. R. et al., "Dark matter halo concentrations in the
           Wilkinson Microwave Anisotropy Probe year 5 cosmology ",
           https://ui.adsabs.harvard.edu/abs/2008MNRAS.390L..64D.
    """

    _defaults = {"a": None, "b": None, "c": None, "ms": 2e12, "sample": "relaxed"}
    native_mdefs = (SOCritical(), SOMean(), SOVirial())

    def cm(self, m, z=0):
        # All the params defined in Table 1 of Duffy 2008
        set_params = {
            "200c": {
                "full": {"a": 5.71, "b": -0.084, "c": 0.47,},
                "relaxed": {"a": 6.71, "b": -0.091, "c": 0.44,},
            },
            "vir": {
                "full": {"a": 7.85, "b": -0.081, "c": 0.71,},
                "relaxed": {"a": 9.23, "b": -0.09, "c": 0.69,},
            },
            "200m": {
                "full": {"a": 10.14, "b": -0.081, "c": 1.01,},
                "relaxed": {"a": 11.93, "b": -0.09, "c": 0.99,},
            },
        }

        parameter_set = set_params.get(self.mdef.colossus_name, set_params["200c"]).get(
            self.params["sample"]
        )
        a = self.params["a"] or parameter_set["a"]
        b = self.params["b"] or parameter_set["b"]
        c = self.params["c"] or parameter_set["c"]
        return self._cm(m, self.params["ms"], a, b, c, z)


class Zehavi11(Bullock01Power):
    r"""
    Concentration-mass relation from Duffy et al.(2008) [1]_.

    It has the same fomulae as :class:`Bullock01Power`,
    but with parameter values refitted.

    See documentation for :class:`Bias` for information on input parameters. This
    model has four model parameters.

    Other Parameters
    ----------------
    a, b, c, ms: float
        Default is ``(11.0,-0.13,1.0,2.26e12)``.

    References
    ----------
    .. [1] Zehavi, I. et al., "Galaxy Clustering in the Completed SDSS Redshift Survey:
           The Dependence on Color and Luminosity",
           https://ui.adsabs.harvard.edu/abs/2011ApJ...736...59Z.
    """
    _defaults = {"a": 11.0, "b": -0.13, "c": 1.0, "ms": 2.26e12}


class Ludlow16(CMRelation):
    r"""
    Analytical Concentration-Mass relation of Ludlow et al.(2016) [1]_.

    See documentation for :class:`Bias` for information on input parameters. This
    model has two model parameters.

    Notes
    -----
    .. note:: The form of the concentration is described by eq(6) and eq(7) in [1]_.

    Other Parameters
    ----------------`
    f, C : float
        Default value is ``f=0.02`` and ``C=650``

    References
    ----------
    .. [1]  Ludlow, A. D. et al., "The mass-concentration-redshift relation
            of cold and warm dark matter haloes ",
            https://ui.adsabs.harvard.edu/abs/2016MNRAS.460.1214L.
    """
    # Note: only defined for NFW for now.
    _defaults = {
        "f": 0.02,  # Fraction of mass assembled at "formation"
        "C": 650,  # Constant scaling
    }
    native_mdefs = (SOCritical(),)

    def delta_halo(self, z=0):
        return self.mdef.halo_overdensity_crit(z, self.cosmo.cosmo)

    def _eq6_zf(self, c, C, z):
        cosmo = self.cosmo.cosmo
        M2 = self.profile._h(1) / self.profile._h(c)
        rho_2 = self.delta_halo(z) * c ** 3 * M2
        rhoc = rho_2 / C
        in_brackets = (
            rhoc * (cosmo.Om0 * (1 + z) ** 3 + cosmo.Ode0) - cosmo.Ode0
        ) / cosmo.Om0
        c = c[in_brackets > 0]
        in_brackets = in_brackets[in_brackets > 0]
        return c, in_brackets ** 0.33333 - 1.0

    def _eq7(self, f, C, m, z):
        cvec = np.logspace(0, 2, 400)

        # Calculate zf for all values in cvec
        cvec, zf = self._eq6_zf(cvec, C, z)

        # Mask out those that are unphysical
        mask = (np.isnan(zf) | np.isinf(zf)) | (zf < 0)
        zf = zf[~mask]
        cvec = cvec[~mask]

        lhs = self.profile._h(1) / self.profile._h(cvec)

        rf = self.filter.mass_to_radius(f * m, self.mean_density0)
        r = self.filter.mass_to_radius(m, self.mean_density0)
        sigf = self.filter.sigma(rf) ** 2
        sigr = self.filter.sigma(r) ** 2

        gf = self.growth.growth_factor_fn()
        num = self.delta_c * (1.0 / gf(zf) - 1.0 / gf(z))
        den = np.sqrt(2 * (sigf - sigr))
        rhs = sp.erfc(np.outer(num, 1.0 / den))

        # indx_mass = 0
        # print('f, rf: ', rf[indx_mass], r[indx_mass])
        # print('sigf: ', sigf[indx_mass])
        # print('sigr: ', sigr[indx_mass])
        #
        # print('lhs: ', lhs)
        # print("rhs: ", rhs[:, indx_mass])
        # print("num: ", num)
        # print("den: ", den[indx_mass])

        if np.isscalar(m):
            rhs = rhs[:, 0]
            spl = interp1d(lhs - rhs, cvec)
            return spl(0.0)
        else:
            out = np.zeros_like(m)
            for i in range(len(m)):
                arg = lhs - rhs[:, i]

                if np.sum(arg <= 0) == 0:
                    out[i] = cvec.min()
                elif np.sum(arg >= 0) == 0:
                    out[i] = cvec.max()
                else:
                    spl = interp1d(arg, cvec)
                    out[i] = spl(0.0)
            return out

    def cm(self, m, z=0):
        return self._eq7(self.params["f"], self.params["C"], m, z)


class Ludlow16Empirical(CMRelation):
    r"""
    Empirical Concentration-Mass relation of Ludlow et al.(2016) [1]_
    for Planck-like cosmology.

    See documentation for :class:`Bias` for information on input parameters. This
    model has eight model parameters.

    Notes
    -----
    .. note:: The form of the concentration is described by eq(C1-C6) in [1]_::

    Other Parameters
    ----------------`
    c0_0, c0_z, beta_0, beta_z, gamma1_0, gamma1_z, gamma2_0, gamma2_z: float
        Default value is ``(3.395,-0.215,0.307,0.54,0.628,-0.047,0.317,-0.893)``.

    References
    ----------
    .. [1]  Ludlow, A. D. et al., "The mass-concentration-redshift relation
            of cold and warm dark matter haloes ",
            https://ui.adsabs.harvard.edu/abs/2016MNRAS.460.1214L.
    """
    _defaults = {
        "c0_0": 3.395,
        "c0_z": -0.215,
        "beta_0": 0.307,
        "beta_z": 0.54,
        "gamma1_0": 0.628,
        "gamma1_z": -0.047,
        "gamma2_0": 0.317,
        "gamma2_z": -0.893,
    }
    native_mdefs = (SOCritical(),)

    def _c0(self, z):
        return self.params["c0_0"] * (1 + z) ** self.params["c0_z"]

    def _beta(self, z):
        return self.params["beta_0"] * (1 + z) ** self.params["beta_z"]

    def _gamma1(self, z):
        return self.params["gamma1_0"] * (1 + z) ** self.params["gamma1_z"]

    def _gamma2(self, z):
        return self.params["gamma2_0"] * (1 + z) ** self.params["gamma2_z"]

    def _nu_0(self, z):
        a = 1.0 / (1 + z)
        return (
            4.135 - 0.564 / a - 0.21 / a ** 2 + 0.0557 / a ** 3 - 0.00348 / a ** 4
        ) / self.growth.growth_factor(z)

    def cm(self, m, z=0):
        warnings.warn(
            "Only use Ludlow16Empirical c(m,z) relation when using Planck-like cosmology"
        )
        # May be better to use real nu, but we'll do what they do in the paper
        # r = self.filter.mass_to_radius(m, self.mean_density0)
        # nu = self.filter.nu(r,self.delta_c)/self.growth.growth_factor(z)
        xi = 1e10 / m
        sig = (
            self.growth.growth_factor(z)
            * 22.26
            * xi ** 0.292
            / (1 + 1.53 * xi ** 0.275 + 3.36 * xi ** 0.198)
        )
        nu = self.delta_c / sig
        return (
            self._c0(z)
            * (nu / self._nu_0(z)) ** (-self._gamma1(z))
            * (1 + (nu / self._nu_0(z)) ** (1.0 / self._beta(z)))
            ** (-self._beta(z) * (self._gamma2(z) - self._gamma1(z)))
        )


class Ludlow2016(Ludlow16):
    "This class is deprecated -- use :class:`Ludlow16` instead."

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "This class is deprecated -- use Ludlow16 instead.",
            category=DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


class Ludlow2016Empirical(Ludlow16Empirical):
    "This class is deprecated -- use :class:`Ludlow16Empirical` instead."

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "This class is deprecated -- use Ludlow16Empirical instead.",
            category=DeprecationWarning,
        )
        super().__init__(*args, **kwargs)
