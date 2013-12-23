'''
Created on Aug 5, 2013

@author: Steven
'''
'''
A class containing the methods to do HOD modelling
'''
#===============================================================================
# Some Imports
#===============================================================================
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.integrate as intg
import numpy as np
import scipy.special as sp

from hmf import Perturbations
import hmf.tools as ht
import tools
from profiles import profiles
from hod_models import HOD_models
from bias import Bias as bias
from fort.routines import hod_routines as fort

# TODO: make hankel transform better -- its pretty wobbly.
#===============================================================================
# The class itself
#===============================================================================
class HOD(object):
    '''
    A class containing methods to do HOD modelling

    The reason this is not in the update-property style of Perturbations, is 
    that 
    there is virtually nothing to incrementally update. Any change of cosmology
    affects the whole calculation virtually (except for the HOD n(M), which is
    trivial). Any change in HOD parameters DOES affect everything. There is almost
    no gain by doing it that way, and it is a lot more complicated, so we stick 
    with normal functions.
    
    INPUT PARAMETERS
        R:     The distances at which the dark matter correlation function is
               calculated in Mpc/h
               Default: np.linspace(1, 200, 200)

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
    def __init__(self, M=np.linspace(8, 18, 500), r=np.linspace(1, 200, 200),
                 M_1=12.851, alpha=1.049, M_min=11.6222,
                 gauss_width=0.26, M_0=11.5047, fca=0.5, fcb=0, fs=1,
                 delta=None, x=1, hod_model='zehavi',
                 halo_profile='nfw', cm_relation='duffy', bias_model='tinker',
                 central=False, nonlinear=False, scale_dependent_bias=True,
                 halo_exclusion=None, ** pert_kwargs):

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
        
        This is not quite the same as hmf.Perturbations.update(). In the 
        HOD() class everything is so interrelated that there is no point
        in a chained approach with its extra complexity. Changing anything
        basically just deletes everything. We only keep properties that 
        may be used more than once or called more than once.
        """
        # We use a somewhat tiered approach here.
        # We first update all simple HOD parameters, then whatever is left
        # is sent to Perturbations(), which will weed out stupid values.

        startfresh = False  # Get's set to true if anything is actually updated

        # First do hod_model parameters - update them and delete entries from kwargs.
        hodmod_p = {k:v for k, v in kwargs.iteritems() if k in self.hodmod_params}
        if hodmod_p:
            self.hodmod_params.update(hodmod_p)

            # Delete the entries we've used from kwargs
            for k in hodmod_p:
                del kwargs[k]

            # Now actually update the hodmod class
            self._hodmod = HOD_models(**self.hodmod_params)
            startfresh = True

        # Now go through the rest of the keys and set values
        for k in kwargs.keys():
            if k == 'halo_profile':
                new = kwargs.pop(k)
                if new != self.halo_profile:
                    self.halo_profile = new
                    startfresh = True

            elif k == 'cm_relation':
                new = kwargs.pop(k)
                if new != self.cm_relation:
                    self.cm_relation = new
                    startfresh = True

            elif k == 'bias_model':
                new = kwargs.pop(k)
                if new != self.bias_model:
                    self.bias_model = new
                    startfresh = True

            elif k == 'r':
                new = kwargs.pop(k)
                if np.any(new != self.r):
                    self.r = new
                    startfresh = True

            elif k == 'nonlinear':
                new = kwargs.pop(k)
                if new != self.nonlinear:
                    self.nonlinear = new
                    startfresh = True

            elif k == "halo_exclusion":
                new = kwargs.pop(k)
                if new != self.halo_exclusion:
                    self.halo_exclusion = new
                    startfresh = True

            elif k == "scale_dependent_bias":
                new = kwargs.pop(k)
                if new != self.scale_dependent_bias:
                    self.scale_dependent_bias = new
                    startfresh = True

        # All that's left should be Perturbations() args
        # However, we don't want to call a new class each time,
        # we usually want to update.
        if kwargs:
            try:
                self.pert.update(cut_fit=False, **kwargs)
            except AttributeError:
                self.pert = Perturbations(cut_fit=False, **kwargs)

            startfresh = True
            try: del self.__matter_power
            except: pass
        # #Now that we've updated the base parameters, we will need to
        # #re-calculate basically everything.
        # We only retain (and therefore must delete here) properties that are
        # either used more than once in the class, OR that we expect may want
        # to be accessed by the user and may cost a bit.
        if startfresh:
            try: del self.__n_cen
            except: pass
            try: del self.__n_sat
            except: pass
            try: del self.__n_tot
            except: pass
            try: del self.__mean_gal_den
            except: pass
            try: del self.__dm_corr
            except: pass
            try: del self.__corr_gal_1h
            except: pass
            try: del self.__corr_gal_2h
            except: pass
            try: del self.__corr_gal
            except: pass
            try: del self.__profile
            except: pass


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
    def bias(self):
        """A class containing the elements necessary to calculate the halo bias"""
        return bias(self.pert, self.bias_model)

    @property
    def profile(self):
        """A class containing the elements necessary to calculate halo profile quantities"""
        try:
            return self.__profile
        except:
            self.__profile = profiles(self.pert.cosmo_params.mean_dens,
                                      self.pert.delta_halo,
                                      profile=self.halo_profile,
                                      cm_relation=self.cm_relation)
            return self.__profile

    @property
    def n_cen(self):
        """ The average number of central galaxies at self.pert.M"""
        try:
            return self.__n_cen
        except:
            self.__n_cen = self._hodmod.nc(self.pert.M)
            return self.__n_cen

    @property
    def n_sat(self):
        """ The average number of satellite galaxies at self.pert.M"""
        try:
            return self.__n_sat
        except:
            self.__n_sat = self._hodmod.ns(self.pert.M)
            return self.__n_sat

    @property
    def n_tot(self):
        """The average number of galaxies at self.pert.M"""
        try:
            return self.__n_tot
        except:
            self.__n_tot = self._hodmod.ntot(self.pert.M)
            return self.__n_tot
    @property
    def n_gal(self):
        """
        The total number density of galaxies
        """
        return self.pert.dndm * self.n_tot

    @property
    def mean_gal_den(self):
        """
        The mean number density of galaxies
        """
        try:
            return self.__mean_gal_den
        except:
            # Integrand is just the density of galaxies at mass M
            integrand = self.pert.dndm * self.n_tot

            self.__mean_gal_den = intg.simps(integrand, self.pert.M)
            return self.__mean_gal_den

    @property
    def matter_power(self):
        """The matter power used in calculations -- can be linear or nonlinear
        Has the same k-vector as linear power in pert
        
        Linear power is always available through self.pert.power
        """
        try:
            return self.__matter_power
        except:
            if self.nonlinear:
                lnk, self.__matter_power = tools.non_linear_power(self.pert.lnk,
                                                                  **dict(self.pert.cosmo_params.pycamb_dict(),
                                                                       **self.pert._camb_options))
                # Now we need to normalise
                # FIXME: get index of self.lnkh which is above -5.3
                ind = np.where(self.pert.lnk > -5.3)[0][0]
                self.__matter_power += self.pert.power[ind] - self.__matter_power[0]
                self.__matter_power = np.concatenate((self.pert.power[:ind], self.__matter_power))
            else:
                self.__matter_power = self.pert.power
            return self.__matter_power

    @property
    def dm_corr(self):
        """
        The dark-matter-only two-point correlation function of the given cosmology
        """
        try:
            return self.__dm_corr
        except:
            fit = spline(self.pert.lnk, self.matter_power, k=1)
            self.__dm_corr = tools.power_to_corr(fit, self.r)
#             self.__dm_corr = fort.power_to_corr(nlnk=len(self.pert.lnk),
#                                                 nr=len(self.r),
#                                                 lnk=self.pert.lnk,
#                                                 r=self.r,
#                                                 power=np.exp(self.matter_power))
            return self.__dm_corr

    @property
    def _power_gal_1h_ss(self):
        """
        The sat-sat part of the 1-halo term of the galaxy power spectrum
        """
        u = self.profile.u(np.exp(self.pert.lnk), self.pert.M, self.pert.z)
        result = fort.power_gal_1h_ss(nlnk=len(self.pert.lnk),
                                      nm=len(self.pert.M),
                                      u=np.asfortranarray(u),
                                      dndm=self.pert.dndm,
                                      nsat=self.n_sat,
                                      ncen=self.n_cen,
                                      mass=self.pert.M,
                                      central=self.hodmod_params['central'])
        result /= self.mean_gal_den ** 2

        return result

    @property
    def _corr_gal_1h_ss(self):
        """The 1-halo galaxy correlation for sat-sat pairs"""
        fit = spline(self.pert.lnk, np.log(self._power_gal_1h_ss))
        result = tools.power_to_corr(fit, self.r)
#         result = fort.power_to_corr(nlnk=len(self.pert.lnk),
#                                     nr=len(self.r),
#                                     lnk=self.pert.lnk,
#                                     r=self.r,
#                                     power=self._power_gal_1h_ss)
        return result

    @property
    def _corr_gal_1h_cs(self):
        """The cen-sat part of the 1-halo galaxy correlations"""
        rho = self.profile.rho(self.r, self.pert.M, self.pert.z)
        result = fort.corr_gal_1h_cs(nr=len(self.r),
                                     nm=len(self.pert.M),
                                     r=self.r,
                                     mass=self.pert.M,
                                     dndm=self.pert.dndm,
                                     ncen=self.n_cen,
                                     nsat=self.n_sat,
                                     rho=np.asfortranarray(rho),
                                     mean_dens=self.pert.cosmo_params.mean_dens,
                                     delta_halo=self.pert.delta_halo)
        result /= self.mean_gal_den ** 2
        return result

    @property
    def corr_gal_1h(self):
        """The 1-halo term of the galaxy correlations"""
        try:
            return self.__corr_gal_1h
        except:
            if self.halo_profile == "nfw":

                rho = self.profile.rho(self.r, self.pert.M, self.pert.z)
                lam = self.profile.lam(self.r, self.pert.M, self.pert.z)
                self.__corr_gal_1h = fort.corr_gal_1h(nr=len(self.r),
                                                      nm=len(self.pert.M),
                                                      r=self.r,
                                                      mass=self.pert.M,
                                                      dndm=self.pert.dndm,
                                                      ncen=self.n_cen,
                                                      nsat=self.n_sat,
                                                      rho=np.asfortranarray(rho),
                                                      lam=np.asfortranarray(lam),
                                                      central=self.hodmod_params['central'],
                                                      mean_dens=self.pert.cosmo_params.mean_dens,
                                                      delta_halo=self.pert.delta_halo)
                self.__corr_gal_1h /= self.mean_gal_den ** 2

            else:
                self.__corr_gal_1h = self._corr_gal_1h_cs + self._corr_gal_1h_ss

            return self.__corr_gal_1h

    def power_gal_2h(self, r_index=None):
        """The 2-halo term of the galaxy power spectrum - NOT LOGGED"""
        if not self.scale_dependent_bias and self.halo_exclusion in [None, "schneider"]:
            u = self.profile.u(np.exp(self.pert.lnk), self.pert.M, self.pert.z)

            pg2h = fort.power_gal_2h(nlnk=len(self.pert.lnk),
                                     nm=len(self.pert.M),
                                     u=np.asfortranarray(u),
                                     bias=self.bias.bias,
                                     nsat=self.n_sat,
                                     ncen=self.n_cen,
                                     dndm=self.pert.dndm,
                                     mass=self.pert.M)

        else:  # We must use a value of r to get pg2h
            pg2h = np.zeros_like(self.pert.lnk)

            if self.scale_dependent_bias:
                xi = self.dm_corr[r_index]
                bias = self.bias.bias_scale(xi)
            else:
                bias = self.bias.bias

            if self.halo_exclusion is not None:
                integrand_m = self.pert.dndm * self.n_tot

            for i, lnk in enumerate(self.pert.lnk):
                u = self.profile.u(np.exp(lnk), self.pert.M , self.pert.z)
                integrand = self.n_tot * u * self.pert.dndm * bias

                if self.halo_exclusion is None or self.halo_exclusion == 'schneider':
                    m = self.pert.M

                elif self.halo_exclusion == "sphere_zheng":
                    mlim = tools.virial_mass(self.r[r_index],
                                             self.pert.cosmo_params['mean_dens'],
                                             self.pert.delta_halo)
                    integrand = integrand[self.pert.M < mlim]
                    m = self.pert.M[self.pert.M < mlim]

                elif self.halo_exclusion == "ng_matched":
                    mmin = np.log(self.pert.M[integrand_m > 0][0])
                    integrand_ms = spline(np.log(self.pert.M[integrand_m > 0]),
                                         np.log(integrand_m[integrand_m > 0]), k=1)
                    def integ(m1, m2, r):
                        rv1, rv2 = tools.virial_radius(np.exp(np.array([m1, m2])),
                                                      self.pert.cosmo_params.mean_dens,
                                                      self.pert.delta_halo)

                        return np.exp(m1 + m2) * np.exp(integrand_ms(m1) + integrand_ms(m2)) * tools.overlapping_halo_prob(r, rv1, rv2)

                    ng, err = intg.dblquad(integ, mmin, np.log(self.pert.M[-1]),
                                           lambda x: mmin, lambda x: np.log(self.pert.M[-1]),
                                           args=(self.r[r_index],), epsrel=1e-3)

                    cumint_ng = intg.cumtrapz(integrand_m, self.pert.M)
                    print cumint_ng[-1], ng
                    index = np.where(cumint_ng > ng)[0][0]
                    print index
                    mlim = self.pert.M[index - 1] + (self.pert.M[index] -
                                                     self.pert.M[index - 1]) * \
                           (ng - cumint_ng[index - 1]) / (cumint_ng[index] -
                                                         cumint_ng[index - 1])

                    integrand = integrand[self.pert.M < mlim]
                    m = self.pert.M[self.pert.M < mlim]

                elif self.halo_exclusion == "ellipsoid":
                    m = self.pert.M[integrand > 0]
                    integrand = integrand[integrand > 0]
                    integrand = spline(np.log(m), np.log(integrand), k=1)

                    def integ(m1, m2, r):
                        rv1, rv2 = tools.virial_radius(np.exp(np.array([m1, m2])),
                                                      self.pert.cosmo_params.mean_dens,
                                                      self.pert.delta_halo)

                        return np.exp(m1 + m2) * np.exp(integrand(m1) + integrand(m2)) * tools.overlapping_halo_prob(r, rv1, rv2)

                    pg2h[i], err = intg.dblquad(integ, np.log(m[0]), np.log(m[-1]),
                                           lambda x: np.log(m[0]), lambda x: np.log(m[-1]),
                                           args=(self.r[r_index],), epsrel=1e-3)

                if self.halo_exclusion != "ellipsoid":
                    pg2h[i] = intg.simps(integrand, m)

        pg2h = np.exp(self.matter_power) * pg2h ** 2 / self.mean_gal_den ** 2

        if self.halo_exclusion == 'schneider':
            # should be k,r/h, but we use k/h and r (which gives the same)
            pg2h *= np.abs(tools.exclusion_window(np.exp(self.pert.lnk), r=2))
        return pg2h

    @property
    def corr_gal_2h(self):
        """The 2-halo term of the galaxy correlation"""
        try:
            return self.__corr_gal_2h
        except:
            if not self.scale_dependent_bias and self.halo_exclusion in [None, 'schneider']:
                fit = spline(self.pert.lnk, np.log(self.power_gal_2h()))
                self.__corr_gal_2h = tools.power_to_corr(fit, self.r)
#                 self.__corr_gal_2h = fort.power_to_corr(nlnk=len(self.pert.lnk),
#                                                        nr=len(self.r),
#                                                        lnk=self.pert.lnk,
#                                                        r=self.r,
#                                                        power=self.power_gal_2h())
            else:
                self.__corr_gal_2h = np.zeros_like(self.r)
                for i, r in enumerate(self.r):
                    power = self.power_gal_2h(i)
                    fit = spline(self.pert.lnk, np.log(power))
                    self.__corr_gal_2h[i] = tools.power_to_corr(fit, r)
#                     self.__corr_gal_2h[i] = fort.power_to_corr(nlnk=len(self.pert.lnk),
#                                                                nr=len(self.r),
#                                                                lnk=self.pert.lnk,
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
