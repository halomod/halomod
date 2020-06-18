"""
Concentration-mass relations.

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

>>> duffy = Duffy08()
>>> m = np.logspace(10, 15, 100)
>>> plt.plot(m, duffy.cm(m, z=0))

Constructing and using a colossus-based relation::

>>> diemer = make_colossus_cm(model='diemer15', statistic='median')()
>>> plt.plot(m, diemer.cm(m, z=1))

Note the extra function call on the first line here -- :func:`make_colossus_cm` returns
a *class*, not an instance. Under the hood, any parameters passed to the function other
than ``model`` are set as "defaults", and can be modified like standard model params.
For instance, using such a model in a broader :class:`~HaloModel` framework::

>>> diemer19_cls = make_colossus_cm(model='diemer19', ps_args={})
>>> from halomod import HaloModel
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
        pass


def make_colossus_cm(model="diemer15", **defaults):
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

    return CustomColossusCM


class Bullock01(CMRelation):
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
    _defaults = {"a": 9.0, "b": -0.13, "c": 1.0, "ms": None}
    native_mdefs = (SOCritical(),)

    def _cm(self, m, ms, a, b, c, z=0):
        return a / (1 + z) ** c * (m / ms) ** b

    def cm(self, m, z=0):
        ms = self.params["ms"] or self.mass_nonlinear(z)
        return self._cm(m, ms, self.params["a"], self.params["b"], self.params["c"], z)


class Duffy08(Bullock01Power):
    """Concentration-mass relation from Duffy+2008.

    Notes
    -----

    .. note:: Only "NFW" parameters are implemented by default here. Of course, you can
              always pass your own parameters from Table 1 of Duffy+2008.

    Other Parameters
    ----------------
    sample : str
        Either "relaxed" or "full". Specifies which set of parameters to take as
        default parameters, from Table 1 of Duffy 2008.
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
    _defaults = {"a": 11.0, "b": -0.13, "c": 1.0, "ms": 2.26e12}


class Ludlow16(CMRelation):
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
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "This class is deprecated -- use Ludlow16 instead.",
            category=DeprecationWarning,
        )
        super().__init__(*args, **kwargs)


class Ludlow2016Empirical(Ludlow16Empirical):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "This class is deprecated -- use Ludlow16Empirical instead.",
            category=DeprecationWarning,
        )
        super().__init__(*args, **kwargs)
