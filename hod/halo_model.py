#===============================================================================
# Some Imports
#===============================================================================
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.integrate as intg
import numpy as np
# import scipy.special as sp

from hmf import MassFunction
# import hmf.tools as ht
import tools
from profiles import profiles
from hod_models import HOD_models
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
                 central=False, nonlinear=False, scale_dependent_bias=True,
                 halo_exclusion=None, ** pert_kwargs):

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
        combined_dict = reduce(lambda a, d: a.update(d) or
                               a, [self.hodmod_params, pert_kwargs], {})


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

        self.update(M=M, halo_profile=halo_profile,
                    cm_relation=cm_relation, bias_model=bias_model,
                    r=r, nonlinear=nonlinear, scale_dependent_bias=scale_dependent_bias,
                    halo_exclusion=halo_exclusion, **combined_dict)


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

            # Now actually update the hodmod class
            self._hodmod = HOD_models(**self.hodmod_params)
            update_hod = True

        # Now go through the rest of the keys and set values
        for k in kwargs.keys():
            if hasattr(self, k):
                try: doset = np.any(getattr(self, k) != kwargs[k])
                except ValueError: doset = not np.array_equal(getattr(self, k), kwargs[k])
                if doset:
                    setattr(self, k, kwargs.pop(k))
                    update_other = True

        # All that's left should be MassFunction() args
        # However, we don't want to call a new class each time,
        # we usually want to update.
        if kwargs:
            try:
                self.hmf.update(cut_fit=False, **kwargs)
            except AttributeError:
                self.hmf = MassFunction(cut_fit=False, **kwargs)

            update_hmf = True
            if 'M' in kwargs:
                update_hod = True

            try: del self.__matter_power
            except: pass

        if update_hod:
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
        available = [None, "sphere_zheng", "ellipsoid", "ng_matched", 'schneider']
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
        The total number density of galaxies
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
            # Integrand is just the density of galaxies at mass M
            integrand = self.hmf.dndm * self.n_tot

            self.__mean_gal_den = intg.simps(integrand, self.hmf.M)
            return self.__mean_gal_den

    @property
    def matter_power(self):
        """The matter power used in calculations -- can be linear or nonlinear
        
        .. note :: Linear power is always available through self.transfer.power
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
            fit = spline(self.hmf.lnk, self.matter_power, k=1)
            self.__dm_corr = tools.power_to_corr(fit, self.r)
#             self.__dm_corr = fort.power_to_corr(nlnk=len(self.hmf.lnk),
#                                                 nr=len(self.r),
#                                                 lnk=self.hmf.lnk,
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
            u = self.profile.u(np.exp(self.hmf.lnk), self.hmf.M, self.hmf.z)
            self.__power_gal_1h_ss = fort.power_gal_1h_ss(nlnk=len(self.hmf.lnk),
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
            fit = spline(self.hmf.lnk, np.log(self._power_gal_1h_ss))
            self.__corr_gal_1h_ss = tools.power_to_corr(fit, self.r)
    #         result = fort.power_to_corr(nlnk=len(self.hmf.lnk),
    #                                     nr=len(self.r),
    #                                     lnk=self.hmf.lnk,
    #                                     r=self.r,
    #                                     power=self._power_gal_1h_ss)
            return self.__corr_gal_1h_ss

    @property
    def _corr_gal_1h_cs(self):
        """The cen-sat part of the 1-halo galaxy correlations"""
        try: return self.__corr_gal_1h_cs
        except:
            rho = self.profile.rho(self.r, self.hmf.M, self.hmf.z)
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

                rho = self.profile.rho(self.r, self.hmf.M, self.hmf.z)
                lam = self.profile.lam(self.r, self.hmf.M, self.hmf.z)
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
        if not self.scale_dependent_bias and self.halo_exclusion in [None, "schneider"]:
            u = self.profile.u(np.exp(self.hmf.lnk), self.hmf.M, self.hmf.z)

            pg2h = fort.power_gal_2h(nlnk=len(self.hmf.lnk),
                                     nm=len(self.hmf.M),
                                     u=np.asfortranarray(u),
                                     bias=self.bias.bias,
                                     nsat=self.n_sat,
                                     ncen=self.n_cen,
                                     dndm=self.hmf.dndm,
                                     mass=self.hmf.M)

        else:  # We must use a value of r to get pg2h
            pg2h = np.zeros_like(self.hmf.lnk)

            if self.scale_dependent_bias:
                xi = self.dm_corr[r_index]
                bias = self.bias.bias_scale(xi)
            else:
                bias = self.bias.bias

            if self.halo_exclusion is not None:
                integrand_m = self.hmf.dndm * self.n_tot

            for i, lnk in enumerate(self.hmf.lnk):
                u = self.profile.u(np.exp(lnk), self.hmf.M , self.hmf.z)
                integrand = self.n_tot * u * self.hmf.dndm * bias

                if self.halo_exclusion is None or self.halo_exclusion == 'schneider':
                    m = self.hmf.M

                elif self.halo_exclusion == "sphere_zheng":
                    mlim = tools.virial_mass(self.r[r_index],
                                             self.cosmo['mean_dens'],
                                             self.hmf.delta_halo)
                    integrand = integrand[self.hmf.M < mlim]
                    m = self.hmf.M[self.hmf.M < mlim]

                elif self.halo_exclusion == "ng_matched":
                    mmin = np.log(self.hmf.M[integrand_m > 0][0])
                    integrand_ms = spline(np.log(self.hmf.M[integrand_m > 0]),
                                         np.log(integrand_m[integrand_m > 0]), k=1)
                    def integ(m1, m2, r):
                        rv1, rv2 = tools.virial_radius(np.exp(np.array([m1, m2])),
                                                      self.cosmo.mean_dens,
                                                      self.hmf.delta_halo)

                        return np.exp(m1 + m2) * np.exp(integrand_ms(m1) + integrand_ms(m2)) * tools.overlapping_halo_prob(r, rv1, rv2)

                    ng, err = intg.dblquad(integ, mmin, np.log(self.hmf.M[-1]),
                                           lambda x: mmin, lambda x: np.log(self.hmf.M[-1]),
                                           args=(self.r[r_index],), epsrel=1e-3)

                    cumint_ng = intg.cumtrapz(integrand_m, self.hmf.M)
                    print cumint_ng[-1], ng
                    index = np.where(cumint_ng > ng)[0][0]
                    print index
                    mlim = self.hmf.M[index - 1] + (self.hmf.M[index] -
                                                     self.hmf.M[index - 1]) * \
                           (ng - cumint_ng[index - 1]) / (cumint_ng[index] -
                                                         cumint_ng[index - 1])

                    integrand = integrand[self.hmf.M < mlim]
                    m = self.hmf.M[self.hmf.M < mlim]

                elif self.halo_exclusion == "ellipsoid":
                    m = self.hmf.M[integrand > 0]
                    integrand = integrand[integrand > 0]
                    integrand = spline(np.log(m), np.log(integrand), k=1)

                    def integ(m1, m2, r):
                        rv1, rv2 = tools.virial_radius(np.exp(np.array([m1, m2])),
                                                      self.cosmo.mean_dens,
                                                      self.hmf.delta_halo)

                        return np.exp(m1 + m2) * np.exp(integrand(m1) + integrand(m2)) * tools.overlapping_halo_prob(r, rv1, rv2)

                    pg2h[i], err = intg.dblquad(integ, np.log(m[0]), np.log(m[-1]),
                                           lambda x: np.log(m[0]), lambda x: np.log(m[-1]),
                                           args=(self.r[r_index],), epsrel=1e-3)

                if self.halo_exclusion != "ellipsoid":
                    pg2h[i] = intg.simps(integrand, m)

        pg2h = np.exp(self.matter_power) * pg2h ** 2 / self.mean_gal_den ** 2

        if self.halo_exclusion == 'schneider':
            # should be k,r/h, but we use k/h and r (which gives the same)
            pg2h *= np.abs(tools.exclusion_window(np.exp(self.hmf.lnk), r=2))
        return pg2h

    @property
    def corr_gal_2h(self):
        """The 2-halo term of the galaxy correlation"""
        try:
            return self.__corr_gal_2h
        except:
            if not self.scale_dependent_bias and self.halo_exclusion in [None, 'schneider']:
                fit = spline(self.hmf.lnk, np.log(self.power_gal_2h()))
                self.__corr_gal_2h = tools.power_to_corr(fit, self.r)
#                 self.__corr_gal_2h = fort.power_to_corr(nlnk=len(self.hmf.lnk),
#                                                        nr=len(self.r),
#                                                        lnk=self.hmf.lnk,
#                                                        r=self.r,
#                                                        power=self.power_gal_2h())
            else:
                self.__corr_gal_2h = np.zeros_like(self.r)
                for i, r in enumerate(self.r):
                    power = self.power_gal_2h(i)
                    fit = spline(self.hmf.lnk, np.log(power))
                    self.__corr_gal_2h[i] = tools.power_to_corr(fit, r)
#                     self.__corr_gal_2h[i] = fort.power_to_corr(nlnk=len(self.hmf.lnk),
#                                                                nr=len(self.r),
#                                                                lnk=self.hmf.lnk,
#                                                                r=self.r,
#                                                                power=power)[0]
            return self.__corr_gal_2h

    @property
    def  corr_gal(self):
        """The galaxy correlation function"""
        try:
            return self.__corr_gal
        except:
            self.__corr_gal = self.corr_gal_1h + self.corr_gal_2h

            return self.__corr_gal
