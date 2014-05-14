#===============================================================================
# Some Imports
#===============================================================================
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.integrate as intg
import numpy as np
from scipy.optimize import minimize
# import scipy.special as sp

from hmf import MassFunction
from hmf._cache import cached_property, set_property
# import hmf.tools as ht
import tools
from profiles import get_profile
from hod import HOD
from bias import Bias as bias
from fort.routines import hod_routines as fort
from twohalo_wrapper import twohalo_wrapper as thalo

from copy import deepcopy

USEFORT = True
#===============================================================================
# The class itself
#===============================================================================
class HaloModel(object):
    '''
    Calculates several quantities using the halo model.

    Parameters
    ----------
    r : array_like, optional, default ``np.logspace(-2.0,1.5,100)`` 
        The scales at which the correlation function is calculated in Mpc/*h*

    M_min: scalar float
           the minimum mass of a halo containing a galaxy

    sigma: scalar float
           the width of the transition from 0 to 1 galaxies
           if None, is step-function.

    M_1 : scalar float
          the mean mass of a halo which contains 1 galaxy

    alpha: scalar float
           the slope of the satellite term

    M_0:    scalar float
            the minimum mass at which satellites can exist

    **kwargs: anything that can be used in the Perturbations class
    
    '''
    def __init__(self, r=None, M=None,
                 M_1=12.851, alpha=1.049, M_min=11.6222,
                 gauss_width=0.26, M_0=11.5047, fca=0.5, fcb=0, fs=1,
                 delta=None, x=1, hod_model='zehavi',
                 halo_profile='NFW', cm_relation='duffy', bias_model='tinker10',
                 central=True, nonlinear=True, scale_dependent_bias=True,
                 halo_exclusion="None", ng=None, ** hmf_kwargs):

        if M is None and ng is not None:
            M = np.linspace(8, 18, 500)
        elif M is None and ng is None:
            M = np.linspace(M_min, 18, 500)

        if r is None:
            r = np.logspace(-1.0, 2.0, 100)

        # A dictionary of all HOD parameters
        self._hodmod_params = {"M_1":M_1,
                              'alpha': alpha,
                              "M_min":M_min,
                              'gauss_width':gauss_width,
                              'M_0':M_0,
                              'fca':fca,
                              'fcb':fcb,
                              'fs':fs,
                              'delta':delta,
                              'x':x,
                              'hod_model':hod_model,
                              'central':central}

        self.hod = self._hodmod_params

        # Initially save parameters to the class.
        # We must do this because update() checks if values are different,
        # which doesn't work if there are no values there to begin with
        self.halo_profile = halo_profile
        self.cm_relation = cm_relation
        self.bias_model = bias_model
        self.r = r
        self.nonlinear = nonlinear
        self.halo_exclusion = halo_exclusion
        self.scale_dependent_bias = scale_dependent_bias
        if "cut_fit" not in hmf_kwargs:
            hmf_kwargs.update({"cut_fit":False})
        self.hmf = MassFunction(M=M, **hmf_kwargs)
        self.__ng = ng

        # Find mmin if we want to
        if ng is not None:
            mmin = self._find_m_min(ng)
            self._hodmod_params.update({"M_min":mmin})
            self.hod = self._hodmod_params
            self.hmf.update(M=np.linspace(mmin, 18, 500))

    def update(self, **kwargs):
        """
        Updates any parameter passed
        """
        # First do hod_model parameters - update them and delete entries from kwargs.
        hodmod_p = {k:v for k, v in kwargs.iteritems() if k in self._hodmod_params}
        if hodmod_p:
            self._hodmod_params.update(hodmod_p)
            if "M_min" in hodmod_p and not "ng" in kwargs:
                self.__ng = None
            else:
                if "ng" in kwargs:
                    self.__ng = kwargs.pop('ng')
                mmin = self._find_m_min(self.__ng)
                self._hodmod_params.update({"M_min":mmin})
                self.hmf.update(M=np.linspace(mmin, 18, 500))
            self.hod = self._hodmod_params

            # Delete the entries we've used from kwargs
            for k in hodmod_p:
                del kwargs[k]


        # Now go through the rest of the keys and set values
        for k in kwargs.keys():
            if hasattr(self, k):
                setattr(self, k, kwargs.pop(k))

        # MassFunction args
        if kwargs:
            self.hmf.update(**kwargs)

            if len(kwargs) > 1 or (len(kwargs) == 1 and "M" not in kwargs):
                del self.matter_power


#===============================================================================
# Set Properties
#===============================================================================
    @set_property("n_sat", "n_cen", "n_tot")
    def hod(self, val):
        """:class:`hod.hod.HOD` class with input parameters"""
        return HOD(**val)

    @set_property("matter_power")
    def nonlinear(self, val):
        """Logical indicating whether the power is to be nonlinear or not"""
        try:
            if val:
                return True
            else:
                return False
        except:
            raise ValueError("nonlinear must be a logical value")

    @set_property("profile")
    def halo_profile(self, val):
        """A string identifier for the halo density profile used"""
        return val

    @set_property("profile")
    def cm_relation(self, val):
        available = ['duffy', 'zehavi', "bullock_rescaled"]
        if val not in available:
            if isinstance(val, str):
                raise ValueError("cm_relation not acceptable: " + str(val))

        return val

    @set_property("bias")
    def bias_model(self, val):
        available = ["ST", 'seljak', "ma", "seljak_warren", "Tinker05", "tinker10"]
        if val not in available:
            raise ValueError("bias_model not acceptable: " + str(val))
        else:
            return val

    @set_property("corr_gal_2h")
    def halo_exclusion(self, val):
        """A string identifier for the type of halo exclusion used (or None)"""
        if val is None:
            val = "None"
        available = ["None", "sphere", "ellipsoid", "ng_matched", 'schneider']
        if val not in available:
            raise ValueError("halo_exclusion not acceptable: " + str(val) + " " + str(type(val)))
        else:
            return val

    @set_property("dm_corr", "_corr_gal_1h_ss", "_corr_gal_1h_cs", "_power_gal_1h_ss",
                  "corr_gal_1h", "corr_gal_2h")
    def r(self, val):
        try:
            if len(val) == 1:
                raise ValueError("r must be a sequence of length > 1")
        except TypeError:
            raise TypeError("r must be a sequence of length > 1")
        return np.array(val)

    @set_property("corr_gal_2h")
    def scale_dependent_bias(self, val):
        try:
            if val:
                return True
            else:
                return False
        except:
            raise ValueError("scale_dependent_bias must be a boolean/have logical value")

#===============================================================================
# Start the actual calculations
#===============================================================================
    @property
    def transfer(self):
        """ :class:`hmf.transfer.Transfer` object aliased from `self.hmf.transfer`"""
        return self.hmf.transfer

    @property
    def cosmo(self):
        """ :class:`hmf.cosmo.Cosmology` object aliased from `self.transfer.cosmo`"""
        return self.transfer.cosmo

    @cached_property("_corr_gal_1h_cs", "_power_gal_1h_ss", "corr_gal_1h")
    def n_sat(self):
        """Average satellite occupancy of halo of mass M"""
        return self.hod.ns(self.hmf.M)

    @cached_property("_corr_gal_1h_cs", "_power_gal_1h_ss", "corr_gal_1h")
    def n_cen(self):
        """Average satellite occupancy of halo of mass M"""
        return self.hod.nc(self.hmf.M)

    @cached_property("mean_gal_den", "n_gal")
    def n_tot(self):
        """Average satellite occupancy of halo of mass M"""
        return self.hod.ntot(self.hmf.M)

    @cached_property("corr_gal_2h", "bias_effective")
    def bias(self):
        """A class containing the elements necessary to calculate the halo bias"""
        return bias(self.hmf, self.bias_model)

    @cached_property("_corr_gal_1h_ss", "_corr_gal_1h_cs", "_power_gal_1h_ss",
                     "corr_gal_1h", "corr_gal_2h")
    def profile(self):
        """A class containing the elements necessary to calculate halo profile quantities"""
        if hasattr(self.halo_profile, "rho"):
            return self.halo_profile
        else:
            return get_profile(self.halo_profile,
                               self.hmf.delta_halo,
                               cm_relation=self.cm_relation,
                               z=self.transfer.z,
                               truncate=True,
                               **self.transfer._cpdict)

    @cached_property()
    def n_gal(self):
        """
        The total number density of galaxies in halos of mass M
        """
        return self.hmf.dndm * self.n_tot

    @cached_property("bias_effective", "mass_effective", "satellite_fraction",
                     "_corr_gal_1h_ss", "_corr_gal_1h_cs", "_power_gal_1h_ss",
                     "corr_gal_1h", "corr_gal_2h")
    def mean_gal_den(self):
        """
        The mean number density of galaxies
        """
        if self.__ng is not None:
            return self.__ng
        else:
            # Integrand is just the density of galaxies at mass M
            integrand = self.hmf.M * self.hmf.dndm * self.n_tot
            return intg.simps(integrand, dx=np.log(self.hmf.M[1]) - np.log(self.hmf.M[0]),
                              even="avg")


    @cached_property()
    def bias_effective(self):
        """
        The galaxy number weighted halo bias factor (Tinker 2005)
        """
        # Integrand is just the density of galaxies at mass M by bias
        integrand = self.hmf.M * self.hmf.dndm * self.n_tot * self.bias.bias

        b = intg.simps(integrand, dx=np.log(self.hmf.M[1]) - np.log(self.hmf.M[0]))
        return b / self.mean_gal_den

    @cached_property()
    def mass_effective(self):
        """
        Average group halo mass, or host-halo mass
        """
        # Integrand is just the density of galaxies at mass M by M
        integrand = self.hmf.M ** 2 * self.hmf.dndm * self.n_tot

        m = intg.simps(integrand, dx=np.log(self.hmf.M[1]) - np.log(self.hmf.M[0]))
        return m / self.mean_gal_den

    @cached_property("central_fraction")
    def satellite_fraction(self):
        # Integrand is just the density of galaxies at mass M
        integrand = self.hmf.M * self.hmf.dndm * self.n_sat
        s = intg.simps(integrand, dx=np.log(self.hmf.M[1]) - np.log(self.hmf.M[0]))
        return s / self.mean_gal_den

    @cached_property()
    def central_fraction(self):
        return 1 - self.satellite_fraction

    @cached_property("dm_corr")
    def matter_power(self):
        """The matter power used in calculations -- can be linear or nonlinear
        
        .. note :: Linear power is available through :attr:`.transfer.power`
        """
        if self.nonlinear:
            return self.transfer.nonlinear_power
        else:
            return self.transfer.power

    @cached_property("corr_gal_2h")
    def dm_corr(self):
        """
        The dark-matter-only two-point correlation function of the given cosmology
        """
        return tools.power_to_corr_ogata(np.exp(self.matter_power),
                                         self.transfer.lnk, self.r)

    @cached_property("_corr_gal_1h_ss")
    def _power_gal_1h_ss(self):
        """
        The sat-sat part of the 1-halo term of the galaxy power spectrum
        """
        u = self.profile.u(np.exp(self.transfer.lnk), self.hmf.M, norm='m')
        p = fort.power_gal_1h_ss(nlnk=len(self.transfer.lnk),
                                 nm=len(self.hmf.M),
                                 u=np.asfortranarray(u),
                                 dndm=self.hmf.dndm,
                                 nsat=self.n_sat,
                                 ncen=self.n_cen,
                                 mass=self.hmf.M,
                                 central=self._hodmod_params['central'])
        return p / self.mean_gal_den ** 2

    @cached_property("corr_gal_1h")
    def _corr_gal_1h_ss(self):
        return tools.power_to_corr_ogata(self._power_gal_1h_ss,
                                         self.transfer.lnk, self.r)

    @cached_property("corr_gal_1h")
    def _corr_gal_1h_cs(self):
        """The cen-sat part of the 1-halo galaxy correlations"""
        rho = self.profile.rho(self.r, self.hmf.M, norm="m")
        c = fort.corr_gal_1h_cs(nr=len(self.r),
                                nm=len(self.hmf.M),
                                r=self.r,
                                mass=self.hmf.M,
                                dndm=self.hmf.dndm,
                                ncen=self.n_cen,
                                nsat=self.n_sat,
                                rho=np.asfortranarray(rho),
                                mean_dens=self.cosmo.mean_dens,
                                delta_halo=self.hmf.delta_halo)
        return c / self.mean_gal_den ** 2

    @cached_property("corr_gal")
    def corr_gal_1h(self):
        """The 1-halo term of the galaxy correlations"""
        if self.profile.has_lam:
            rho = self.profile.rho(self.r, self.hmf.M, norm="m")
            lam = self.profile.lam(self.r, self.hmf.M)
            c = fort.corr_gal_1h(nr=len(self.r),
                                 nm=len(self.hmf.M),
                                 r=self.r,
                                 mass=self.hmf.M,
                                 dndm=self.hmf.dndm,
                                 ncen=self.n_cen,
                                 nsat=self.n_sat,
                                 rho=np.asfortranarray(rho),
                                 lam=np.asfortranarray(lam),
                                 central=self._hodmod_params['central'],
                                 mean_dens=self.cosmo.mean_dens,
                                 delta_halo=self.hmf.delta_halo)

            return c / self.mean_gal_den ** 2

        else:
            return self._corr_gal_1h_cs + self._corr_gal_1h_ss
#
#     def power_gal_2h(self, r_index=None):
#         """The 2-halo term of the galaxy power spectrum - NOT LOGGED"""
#
#         # Allocate the memory
#         pg2h = np.zeros_like(self.transfer.lnk)
#
#         # Get the bias
#         if self.scale_dependent_bias:
#             xi = self.dm_corr[r_index]
#             bias = self.bias.bias_scale(xi)
#         else:
#             bias = self.bias.bias
#
#         ng_integrand = (self.hmf.dndm * self.n_tot)[self.n_tot > 0]
#         integrand_m = ng_integrand * bias[self.n_tot > 0]
#         m = self.hmf.M[self.n_tot > 0]
#
#         #### First do stuff in any of the methods which is not k-dependent
#         if self.halo_exclusion == "ng_matched":
#             rv = tools.virial_radius(m, self.cosmo.mean_dens, self.hmf.delta_halo)
#             prob = tools.overlapping_halo_prob(self.r[r_index], rv, rv)
#             integ = np.outer(m * ng_integrand, m * ng_integrand) * prob
#             del prob, rv
#             ng = np.sqrt(tools.dblsimps(integ, dx=np.log(m[1]) - np.log(m[0]),
#                               dy=np.log(m[1]) - np.log(m[0])))
#             cumint_ng = intg.cumtrapz(ng_integrand, m)
#             index = np.where(cumint_ng > ng)[0][0]
#
#         elif self.halo_exclusion == "sphere":
#             mlim = tools.virial_mass(self.r[r_index],
#                                      self.cosmo.mean_dens,
#                                      self.hmf.delta_halo)
#             m = m[m < mlim]
#             integrand_m = integrand_m[m < mlim]
#
#         elif self.halo_exclusion in ["schneider", "None"]:
#                 mm = m
#
#         # # Now do slow k-dependent stuff
#         for i, lnk in enumerate(self.transfer.lnk):
#
#             integrand = integrand_m * self.profile.u(np.exp(lnk), m , norm='m')
#             if self.halo_exclusion == "sphere":
#                 integrand = integrand[m < mlim]
#                 mm = m[m < mlim]
#             elif self.halo_exclusion == "ng_matched":
#                 integrand = integrand[:index]
#                 mm = m[:index]
#
#             elif self.halo_exclusion == "ellipsoid":
#                 rv = tools.virial_radius(m, self.cosmo.mean_dens, self.hmf.delta_halo)
#                 prob = tools.overlapping_halo_prob(self.r[r_index], rv, rv)
#                 integ = np.outer(m * integrand, m * integrand) * prob
#                 del prob, rv
#                 pg2h[i] = np.sqrt(tools.dblsimps(integ, dx=np.log(m[1]) - np.log(m[0]),
#                                   dy=np.log(m[1]) - np.log(m[0])))
#
#             if self.halo_exclusion != "ellipsoid":
#                 if len(mm) > 2:
#                     pg2h[i] = intg.simps(integrand, mm)
#                 else:
#                     pg2h[i] = 0
#
# #         print "integrandm: ", integrand_m[:10] * self.cosmo.h ** 4, integrand_m[-10:] * self.cosmo.h ** 4
# #         print "pg2h: ", pg2h[:10] * self.cosmo.h ** 3, pg2h[-10:] * self.cosmo.h ** 3
#         pg2h = np.exp(self.matter_power) * pg2h ** 2 / self.mean_gal_den ** 2
#
#         if self.halo_exclusion == 'schneider':
#             # should be k,r/h, but we use k/h and r (which gives the same)
#             pg2h *= tools.exclusion_window(np.exp(self.transfer.lnk), r=3)
#
#         # We do this so log splines work later on :)
#         pg2h[pg2h == 0] = 1e-20
#
#         return pg2h

    @cached_property("corr_gal")
    def corr_gal_2h(self):
        """The 2-halo term of the galaxy correlation"""
        u = self.profile.u(np.exp(self.transfer.lnk), self.hmf.M , norm='m')
        return thalo(self.halo_exclusion, self.scale_dependent_bias,
                     self.hmf.M, self.bias.bias, self.n_tot,
                     self.hmf.dndm, self.transfer.lnk,
                     np.exp(self.matter_power), u, self.r, self.dm_corr,
                     self.mean_gal_den, self.hmf.delta_halo,
                     self.hmf.cosmo.mean_dens)

    @cached_property("projected_corr_gal")
    def  corr_gal(self):
        """The galaxy correlation function"""
        return self.corr_gal_1h + self.corr_gal_2h

    def _find_m_min(self, ng):
        """
        Calculate the minimum mass of a halo to contain a (central) galaxy 
        based on a known mean galaxy density
        """
        self.hmf.update(M=np.linspace(8, 18, 1000))
        if self._hodmod_params["hod_model"] == "zheng":
            self._hodmod_params["M_min"] = np.log10(self.hmf.M[0])
            x = HOD(**self._hodmod_params)
            integrand = self.hmf.M * self.hmf.dndm * x.ntot(self.hmf.M)
            integral = intg.cumtrapz(integrand[::-1], dx=np.log(self.hmf.M[1]) - np.log(self.hmf.M[0]))
            if integral[-1] < ng:
                raise NGException("Maximum mean galaxy density exceeded: " + str(integral[-1]))
            spline_int = spline(np.log(integral), np.log(self.hmf.M[:-1])[::-1], k=3)
            mmin = spline_int(np.log(ng)) / np.log(10)

        else:
            # Anything else requires us to do some optimization unfortunately.
            params = self._hodmod_params
            params["M_min"] = np.log10(self.hmf.M[0])
            x = HOD(**params)
            integrand = self.hmf.M * self.hmf.dndm * x.ntot(self.hmf.M)
            integral = intg.simps(integrand, dx=np.log(self.hmf.M[1]) - np.log(self.hmf.M[0]))
            if integral < ng:
                raise NGException("Maximum mean galaxy density exceeded: " + str(integral))

            def model(mmin):
                self._hodmod_params.update({"M_min":mmin})
                x = HOD(**self._hodmod_params)
                integrand = self.hmf.M * self.hmf.dndm * x.ntot(self.hmf.M)
                integral = intg.simps(integrand, dx=np.log(self.hmf.M[1]) - np.log(self.hmf.M[0]))
                return abs(integral - ng)

            res = minimize(model, self._hodmod_params["M_min"], tol=1e-3,
                           method="Nelder-Mead", options={"maxiter":200})
            mmin = res.x[0]

        return mmin

    @cached_property()
    def projected_corr_gal(self):
        """
        Projected correlation function w(r_p).

        From Beutler 2011, eq 6.

        To integrate perform a substitution y = x - r_p.
        """
        # We make a copy of the current instance but increase the number of
        # r and extend the range
        cr_max = max(80.0, 5 * self.r.max())

        # This is a bit of a hack, but make sure self has all parent attributes
        # self.hmf.dndm; self.matter_power
        c = deepcopy(self)
        c.update(r=10 ** np.arange(np.log10(self.r.min()), np.log10(cr_max), 0.05))
        fit = spline(np.log(c.r[c.corr_gal > 0]), np.log(c.corr_gal[c.corr_gal > 0]), k=3)
        p = np.zeros(len(self.r))

        for i, rp in enumerate(self.r):
            # # Get steepest slope.
            ydiff = fit.derivatives(np.log(rp))[1]
            a = max(1.3, -ydiff)
            frac = self._get_slope_frac(a)
            min_y = frac * 0.005 ** 2 * rp  # 2.5% accuracy??

            # Set the y vector for this rp
            y = np.logspace(np.log10(min_y), np.log10(max(80.0, 5 * rp) - rp), 1000)

            # Integrate
            integ_corr = np.exp(fit(np.log(y + rp)))
            integrand = integ_corr * (y + rp) / np.sqrt((y + 2 * rp) * y)
            p[i] = intg.simps(integrand, y) * 2

        return p

    def _get_slope_frac(self, a):
        frac = 2 ** (1 + 2 * a) * (7 - 2 * a ** 3 + 3 * np.sqrt(5 - 8 * a + 4 * a ** 2) + a ** 2 * (9 + np.sqrt(5 - 8 * a + 4 * a ** 2)) -
                           a * (13 + 3 * np.sqrt(5 - 8 * a + 4 * a ** 2))) * ((1 + np.sqrt(5 - 8 * a + 4 * a ** 2)) / (a - 1)) ** (-2 * a)
        frac /= (a - 1) ** 2 * (-1 + 2 * a + np.sqrt(5 - 8 * a + 4 * a ** 2))
        return frac
class NGException(Exception):
    pass
