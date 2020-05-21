"""
Created on 08/12/2014

@author: Steven
"""
import warnings

import numpy as np
from hmf import Component
from scipy import special as sp
from scipy.interpolate import interp1d
from scipy.optimize import minimize

from colossus.halo import concentration


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
        mean_density0 : float
            Mean density of the universe at z=0
            Required for ``Bullock01``
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

    use_cosmo = False

    def __init__(
        self,
        filter0=None,
        mean_density0=None,
        growth=None,
        delta_c=1.686,
        profile=None,
        cosmo=None,
        delta_halo=200.0,
        mdef="200c",
        **model_parameters
    ):
        # Save instance variables
        self.filter = filter0
        self.growth = growth
        self.mean_density0 = mean_density0
        self.delta_c = delta_c
        self.delta_halo = delta_halo

        # TODO: mean_density0 should be removed, as it is in cosmo
        self.profile = profile
        self.cosmo = cosmo
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

    def _cm(self, m, z=0):
        pass

    def cm(self, m, z=0):
        return concentration.concentration(
            m=m,
            mdef=self.mdef,
            z=z,
            model=self.__class__.__name__,
            range_return=False,
            range_warning=True,
            **self.params
        )


# Wrap all COLOSSUS models with the CMRelation class.
# for _mdl in concentration.models:


# class NFW(CMRelation):
#    _defaults = {'f':,"k":}


class Bullock01(CMRelation):
    _defaults = {"F": 0.01, "K": 3.4}

    def zc(self, m, z=0):
        r = self.filter.mass_to_radius(self.params["F"] * m, self.mean_density0)
        nu = self.filter.nu(r, self.delta_c)
        g = self.growth.growth_factor_fn(inverse=True)
        zc = g(np.sqrt(nu))
        zc[zc < z] = z  # hack?
        return zc

    def cm(self, m, z=0):
        return self.params["K"] * (self.zc(m, z) + 1.0) / (z + 1.0)


class Bullock01_Power(CMRelation):
    _defaults = {"a": 9.0, "b": -0.13, "c": 1.0, "ms": None}

    def cm(self, m, z=0):
        ms = self.params["ms"] or self.mass_nonlinear(z)
        return (
            self.params["a"]
            / (1 + z) ** self.params["c"]
            * (m / ms) ** self.params["b"]
        )


class Duffy08(Bullock01_Power):
    _defaults = {"a": 6.71, "b": -0.091, "c": 0.44, "ms": 2e12}


class Zehavi11(Bullock01_Power):
    _defaults = {"a": 11.0, "b": -0.13, "c": 1.0, "ms": 2.26e12}


class Ludlow2016(CMRelation):
    # Note: only defined for NFW for now.
    _defaults = {
        "f": 0.02,  # Fraction of mass assembled at "formation"
        "C": 650,  # Constant scaling
    }

    def _eq6_zf(self, c, C, z):
        M2 = self.profile._h(1) / self.profile._h(c)
        rho_2 = self.delta_halo * c ** 3 * M2
        rhoc = rho_2 / C
        in_brackets = (
            rhoc * (self.cosmo.Om0 * (1 + z) ** 3 + self.cosmo.Ode0) - self.cosmo.Ode0
        ) / self.cosmo.Om0
        c = c[in_brackets > 0]
        in_brackets = in_brackets[in_brackets > 0]
        return c, in_brackets ** 0.33333 - 1.0

    def _eq7(self, f, C, m, z):
        cvec = np.logspace(0, 2, 400)

        # Calculate zf for all values in cvec
        cvec, zf = self._eq6_zf(cvec, C, z)

        # Mask out those that are unphysical
        mask = np.logical_or(np.logical_or(np.isnan(zf), np.isinf(zf)), zf < 0)
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
                else:
                    spl = interp1d(arg, cvec)
                    out[i] = spl(0.0)
            return out

    def cm(self, m, z=0):
        return self._eq7(self.params["f"], self.params["C"], m, z)


class Ludlow2016Empirical(CMRelation):
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

    def cm(self, m, z):
        warnings.warn(
            "Only use Ludlow2016Empirical c(m,z) relation when using Planck-like cosmology"
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
