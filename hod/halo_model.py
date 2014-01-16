#===============================================================================
# Some Imports
#===============================================================================
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.integrate as intg
import numpy as np
from scipy.optimize import minimize
# import scipy.special as sp

from hmf import MassFunction
# import hmf.tools as ht
import tools
from profiles import profiles
from hod import HOD
from bias import Bias as bias
from fort.routines import hod_routines as fort

# TODO: make hankel transform better -- its pretty wobbly.
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
                 halo_profile='nfw', cm_relation='duffy', bias_model='tinker',
                 central=True, nonlinear=True, scale_dependent_bias=True,
                 halo_exclusion="None", ng=None, ** hmf_kwargs):

        if M is None:
            M = np.linspace(8, 18, 500)

        if r is None:
            r = np.logspace(-2.0, 1.5, 100)

        # A dictionary of all HOD parameters
        self.hodmod_params = {"M_1":M_1,
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
        self.ng = ng
        self.hmf = MassFunction(cut_fit=False, **hmf_kwargs)

        # Find mmin if we want to
        if self.ng is not None:
            mmin = self._find_m_min()
            self.hodmod_params.update({"M_min":mmin})

        self._hodmod = HOD(**self.hodmod_params)

        # A couple of simple derived parameters
        self.n_cen = self._hodmod.nc(self.hmf.M)
        self.n_sat = self._hodmod.ns(self.hmf.M)
        self.n_tot = self._hodmod.ntot(self.hmf.M)

    def update(self, **kwargs):
        """
        Updates any parameter passed to it (if it is indeed different)
        
        This is not quite the same as hmf.MassFunction.update(). In the 
        HaloModel() class everything is so interrelated that there is no point
        in a chained approach with its extra complexity. Changing anything
        basically just deletes everything. We only keep properties that 
        may be used more than once or called more than once.
        """
        # We use a somewhat tiered approach here.
        # We first update all simple HOD parameters, then whatever is left
        # is sent to Perturbations(), which will weed out stupid values.

        update_hmf = False  # Get's set to true if any hmf parameters updated
        update_hod = False  # Set to true if any hod parameters updated
        update_other = False  # Set to true if anything else updated

        # First do hod_model parameters - update them and delete entries from kwargs.
        hodmod_p = {k:v for k, v in kwargs.iteritems() if k in self.hodmod_params}
        if hodmod_p:
            self.hodmod_params.update(hodmod_p)

            # Delete the entries we've used from kwargs
            for k in hodmod_p:
                del kwargs[k]

            update_hod = True

        # Now go through the rest of the keys and set values
        for k in kwargs.keys():
            if hasattr(self, k):
                try: doset = np.any(getattr(self, k) != kwargs[k])
                except ValueError: doset = not np.array_equal(getattr(self, k), kwargs[k])
                if k == "ng":
                    update_hod = True
                if doset:
                    setattr(self, k, kwargs.pop(k))
                    update_other = True

        # All that's left should be MassFunction() args
        # However, we don't want to call a new class each time,
        # we usually want to update.
        if kwargs:
            self.hmf.update(**kwargs)

            update_hmf = True
            if 'M' in kwargs:
                update_hod = True

            try: del self.__matter_power
            except AttributeError: pass

        if update_hod:
            # Find mmin if we want to
            if self.ng is not None:
                mmin = self._find_m_min()
                self.hodmod_params.update({"M_min":mmin})
            # Now actually update the hodmod class
            self._hodmod = HOD(**self.hodmod_params)
            self.n_cen = self._hodmod.nc(self.hmf.M)
            self.n_sat = self._hodmod.ns(self.hmf.M)
            self.n_tot = self._hodmod.ntot(self.hmf.M)

        if update_hod or update_hmf:
            if hasattr(self, "_HaloModel__mean_gal_den"): del self.__mean_gal_den

        # Now reset everything if anything has been updated.
        # The idea here is that any property may need to be called by the user.
        # If so, we don't want to double-up on any calculation, so we store
        # the value. Thus we need to delete it here if it already exists.
        if update_hmf or update_hod or update_other:
            for quantity in ["dm_corr", "corr_gal_1h",
                             "corr_gal_2h", "corr_gal", "power_gal_1h_ss",
                             "corr_gal_1h_ss", "corr_gal_1h_cs"]:
                if "_HaloModel__" + quantity in self.__dict__:
                    delattr(self, "_HaloModel__" + quantity)

#===============================================================================
# Set Properties
#===============================================================================
    @property
    def nonlinear(self):
        """Logical indicating whether the power is to be nonlinear or not"""
        return self.__nonlinear

    @nonlinear.setter
    def nonlinear(self, val):
        try:
            if val:
                self.__nonlinear = True
            else:
                self.__nonlinear = False
        except:
            raise ValueError("nonlinear must be a logical value")

    @property
    def halo_profile(self):
        """A string identifier for the halo density profile used"""
        return self.__halo_profile

    @halo_profile.setter
    def halo_profile(self, val):
        available_profs = ['nfw']
        if val not in available_profs:
            raise ValueError("halo_profile not acceptable: " + str(val))
        else:
            self.__halo_profile = val

    @property
    def cm_relation(self):
        """A string identifier for the concentration-mass relation used"""
        return self.__cm_relation

    @cm_relation.setter
    def cm_relation(self, val):
        available = ['duffy', 'zehavi']
        if val not in available:
            raise ValueError("cm_relation not acceptable: " + str(val))
        else:
            self.__cm_relation = val

    @property
    def bias_model(self):
        """A string identifier for the bias model used"""
        return self.__bias_model

    @bias_model.setter
    def bias_model(self, val):
        available = ["ST", 'seljak', "ma", "seljak_warren", "tinker"]
        if val not in available:
            raise ValueError("bias_model not acceptable: " + str(val))
        else:
            self.__bias_model = val

    @property
    def halo_exclusion(self):
        """A string identifier for the type of halo exclusion used (or None)"""
        return self.__halo_exclusion

    @halo_exclusion.setter
    def halo_exclusion(self, val):
        available = ["None", "sphere", "ellipsoid", "ng_matched", 'schneider']
        if val not in available:
            raise ValueError("halo_exclusion not acceptable: " + str(val))
        else:
            self.__halo_exclusion = val

    @property
    def r(self):
        """An array of values of the separation (in Mpc/h) to get xi(r) at"""
        return self.__r

    @r.setter
    def r(self, val):
        try:
            if len(val) == 1:
                raise ValueError("r must be a sequence of length > 1")
        except TypeError:
            raise TypeError("r must be a sequence of length > 1")
        self.__r = np.array(val)

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

    @property
    def bias(self):
        """A class containing the elements necessary to calculate the halo bias"""
        return bias(self.hmf, self.bias_model)

    @property
    def profile(self):
        """A class containing the elements necessary to calculate halo profile quantities"""
        return profiles(self.cosmo.mean_dens,
                        self.hmf.delta_halo,
                        profile=self.halo_profile,
                        cm_relation=self.cm_relation)


    @property
    def n_gal(self):
        """
        The total number density of galaxies in halos of mass M
        """
        return self.hmf.dndm * self.n_tot

    @property
    def mean_gal_den(self):
        """
        The mean number density of galaxies
        """
        try:
            return self.__mean_gal_den
        except:
            if self.ng is not None:
                self.__mean_gal_den = self.ng
                return self.ng
            else:
                # Integrand is just the density of galaxies at mass M
                integrand = self.hmf.M * self.hmf.dndm * self.n_tot

                self.__mean_gal_den = intg.simps(integrand, dx=np.log(self.hmf.M[1]) - np.log(self.hmf.M[0]))
                return self.__mean_gal_den

    @property
    def matter_power(self):
        """The matter power used in calculations -- can be linear or nonlinear
        
        .. note :: Linear power is available through :attr:`.transfer.power`
        """
        try:
            return self.__matter_power
        except:
            if self.nonlinear:
                self.__matter_power = self.transfer.nonlinear_power
            else:
                self.__matter_power = self.transfer.power
            return self.__matter_power

    @property
    def dm_corr(self):
        """
        The dark-matter-only two-point correlation function of the given cosmology
        """
        try:
            return self.__dm_corr
        except:
            fit = spline(self.transfer.lnk, self.matter_power, k=1)
            self.__dm_corr = tools.power_to_corr(fit, self.r)
#             self.__dm_corr = fort.power_to_corr(nlnk=len(self.transfer.lnk),
#                                                 nr=len(self.r),
#                                                 lnk=self.transfer.lnk,
#                                                 r=self.r,
#                                                 power=np.exp(self.matter_power))
            return self.__dm_corr

    @property
    def _power_gal_1h_ss(self):
        """
        The sat-sat part of the 1-halo term of the galaxy power spectrum
        """
        try:
            return self.__power_gal_1h_ss
        except AttributeError:
            u = self.profile.u(np.exp(self.transfer.lnk), self.hmf.M, self.hmf.transfer.z)
            self.__power_gal_1h_ss = fort.power_gal_1h_ss(nlnk=len(self.transfer.lnk),
                                                          nm=len(self.hmf.M),
                                                          u=np.asfortranarray(u),
                                                          dndm=self.hmf.dndm,
                                                          nsat=self.n_sat,
                                                          ncen=self.n_cen,
                                                          mass=self.hmf.M,
                                                          central=self.hodmod_params['central'])
            self.__power_gal_1h_ss /= self.mean_gal_den ** 2

        return self.__power_gal_1h_ss

    @property
    def _corr_gal_1h_ss(self):
        """The 1-halo galaxy correlation for sat-sat pairs"""
        try: return self.__corr_gal_1h_ss
        except:
            fit = spline(self.transfer.lnk, np.log(self._power_gal_1h_ss))
            self.__corr_gal_1h_ss = tools.power_to_corr(fit, self.r)
    #         result = fort.power_to_corr(nlnk=len(self.transfer.lnk),
    #                                     nr=len(self.r),
    #                                     lnk=self.transfer.lnk,
    #                                     r=self.r,
    #                                     power=self._power_gal_1h_ss)
            return self.__corr_gal_1h_ss

    @property
    def _corr_gal_1h_cs(self):
        """The cen-sat part of the 1-halo galaxy correlations"""
        try: return self.__corr_gal_1h_cs
        except:
            rho = self.profile.rho(self.r, self.hmf.M, self.hmf.transfer.z)
            self.__corr_gal_1h_cs = fort.corr_gal_1h_cs(nr=len(self.r),
                                         nm=len(self.hmf.M),
                                         r=self.r,
                                         mass=self.hmf.M,
                                         dndm=self.hmf.dndm,
                                         ncen=self.n_cen,
                                         nsat=self.n_sat,
                                         rho=np.asfortranarray(rho),
                                         mean_dens=self.cosmo.mean_dens,
                                         delta_halo=self.hmf.delta_halo)
            self.__corr_gal_1h_cs /= self.mean_gal_den ** 2
            return self.__corr_gal_1h_cs

    @property
    def corr_gal_1h(self):
        """The 1-halo term of the galaxy correlations"""
        try:
            return self.__corr_gal_1h
        except:
            if self.halo_profile == "nfw":

                rho = self.profile.rho(self.r, self.hmf.M, self.hmf.transfer.z)
                lam = self.profile.lam(self.r, self.hmf.M, self.hmf.transfer.z)
                self.__corr_gal_1h = fort.corr_gal_1h(nr=len(self.r),
                                                      nm=len(self.hmf.M),
                                                      r=self.r,
                                                      mass=self.hmf.M,
                                                      dndm=self.hmf.dndm,
                                                      ncen=self.n_cen,
                                                      nsat=self.n_sat,
                                                      rho=np.asfortranarray(rho),
                                                      lam=np.asfortranarray(lam),
                                                      central=self.hodmod_params['central'],
                                                      mean_dens=self.cosmo.mean_dens,
                                                      delta_halo=self.hmf.delta_halo)
                self.__corr_gal_1h /= self.mean_gal_den ** 2

            else:
                self.__corr_gal_1h = self._corr_gal_1h_cs + self._corr_gal_1h_ss

            return self.__corr_gal_1h

    def power_gal_2h(self, r_index=None):
        """The 2-halo term of the galaxy power spectrum - NOT LOGGED"""

        # Allocate the memory
        pg2h = np.zeros_like(self.transfer.lnk)

        # Get the bias
        if self.scale_dependent_bias:
            xi = self.dm_corr[r_index]
            bias = self.bias.bias_scale(xi)
        else:
            bias = self.bias.bias

        ng_integrand = (self.hmf.dndm * self.n_tot)[self.n_tot > 0]
        integrand_m = ng_integrand * bias[self.n_tot > 0]
        m = self.hmf.M[self.n_tot > 0]

        #### First do stuff in any of the methods which is not k-dependent
        if self.halo_exclusion == "ng_matched":
            rv = tools.virial_radius(m, self.cosmo.mean_dens, self.hmf.delta_halo)
            prob = tools.overlapping_halo_prob(self.r[r_index], rv, rv)
            integ = np.outer(m * ng_integrand, m * ng_integrand) * prob
            del prob, rv
            ng = np.sqrt(tools.dblsimps(integ, dx=np.log(m[1]) - np.log(m[0]),
                              dy=np.log(m[1]) - np.log(m[0])))
            cumint_ng = intg.cumtrapz(ng_integrand, m)
            index = np.where(cumint_ng > ng)[0][0]

        elif self.halo_exclusion == "sphere":
            mlim = tools.virial_mass(self.r[r_index],
                                     self.cosmo.mean_dens,
                                     self.hmf.delta_halo)
            m = m[m < mlim]

        elif self.halo_exclusion == "schneider":
                mm = m

        # # Now do slow k-dependent stuff
        for i, lnk in enumerate(self.transfer.lnk):
            integrand = integrand_m * self.profile.u(np.exp(lnk), m , self.transfer.z)

            if self.halo_exclusion == "sphere":
                integrand = integrand[m < mlim]
                mm = m[m < mlim]
            elif self.halo_exclusion == "ng_matched":
                integrand = integrand[:index]
                mm = m[:index]

            elif self.halo_exclusion == "ellipsoid":
                rv = tools.virial_radius(m, self.cosmo.mean_dens, self.hmf.delta_halo)
                prob = tools.overlapping_halo_prob(self.r[r_index], rv, rv)
                integ = np.outer(m * integrand, m * integrand) * prob
                del prob, rv
                pg2h[i] = np.sqrt(tools.dblsimps(integ, dx=np.log(m[1]) - np.log(m[0]),
                                  dy=np.log(m[1]) - np.log(m[0])))

            if self.halo_exclusion != "ellipsoid":
                if len(mm) > 2:
                    pg2h[i] = intg.simps(integrand, mm)
                else:
                    pg2h[i] = 0
        pg2h = np.exp(self.matter_power) * pg2h ** 2 / self.mean_gal_den ** 2

        if self.halo_exclusion == 'schneider':
            # should be k,r/h, but we use k/h and r (which gives the same)
            pg2h *= np.abs(tools.exclusion_window(np.exp(self.transfer.lnk), r=2))

        # We do this so log splines work later on :)
        pg2h[pg2h == 0] = 1e-20

        return pg2h

    @property
    def corr_gal_2h(self):
        """The 2-halo term of the galaxy correlation"""
        try:
            return self.__corr_gal_2h
        except:
            if not self.scale_dependent_bias and self.halo_exclusion in ["None", 'schneider']:
                fit = spline(self.transfer.lnk, np.log(self.power_gal_2h()))
                self.__corr_gal_2h = tools.power_to_corr(fit, self.r)

            else:
                self.__corr_gal_2h = np.zeros_like(self.r)
                for i, r in enumerate(self.r):
                    power = self.power_gal_2h(i)
                    fit = spline(self.transfer.lnk, np.log(power))
                    self.__corr_gal_2h[i] = tools.power_to_corr(fit, r)

            return self.__corr_gal_2h

    @property
    def  corr_gal(self):
        """The galaxy correlation function"""
        try:
            return self.__corr_gal
        except:
            self.__corr_gal = self.corr_gal_1h + self.corr_gal_2h

            return self.__corr_gal

    def _find_m_min(self):
        """
        Calculate the minimum mass of a halo to contain a (central) galaxy 
        based on a known mean galaxy density
        """

        if self.hodmod_params["hod_model"] == "zheng":
            self.hodmod_params.update({"M_min":np.log10(self.hmf.M[0])})
            x = HOD(**self.hodmod_params)
            integrand = self.hmf.M * self.hmf.dndm * x.ntot(self.hmf.M)
            integral = intg.cumtrapz(integrand[::-1], dx=np.log(self.hmf.M[1]) - np.log(self.hmf.M[0]))
            spline_int = spline(np.log(integral), np.log(self.hmf.M[:-1])[::-1], k=3)
            mmin = spline_int(np.log(self.ng)) / np.log(10)

        else:
            # Anything else requires us to do some optimization unfortunately.
            def model(mmin):
                self.hodmod_params.update({"M_min":mmin})
                x = HOD(**self.hodmod_params)
                integrand = self.hmf.M * self.hmf.dndm * x.ntot(self.hmf.M)
                integral = intg.simps(integrand, dx=np.log(self.hmf.M[1]) - np.log(self.hmf.M[0]))
                return abs(integral - self.ng)

            res = minimize(model, self.hodmod_params["M_min"], tol=1e-3,
                           method="Nelder-Mead", options={"maxiter":200})
            mmin = res.x[0]

        return mmin
