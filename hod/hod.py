'''
Created on Aug 5, 2013

@author: Steven
'''
'''
A class containing the methods to do HOD modelling
'''
#TODO: figure out what to do when M<M_0 -- is it just 0?? I'd say so...
###############################################################################
# Some Imports
###############################################################################
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.integrate as intg
import numpy as np
import scipy.special as sp

from hmf import Perturbations
import hmf.tools as ht
import tools
from profiles import profiles
from hod_models import HOD_models as hm
from bias import Bias as bias
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
    def __init__(self, M=np.linspace(8, 18, 1001), r=np.linspace(1, 200, 200),
                  M_1=10 ** 12.851,
                 M_c=10 ** 12.0, alpha=1.049, M_min=10 ** 11.6222,
                 gauss_width=0.26, M_0=10 ** 11.5047, fca=0.5, fcb=0, fs=1,
                 delta=None, x=1, hod_model='zehavi',
                 halo_profile='nfw', cm_relation='duffy', bias_model='tinker',
                 central=False, nonlinear=False, scale_dependent_bias=True,
                 halo_exclusion=None, ** pert_kwargs):

        #A dictionary of all HOD parameters
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

        self.update(M=M, halo_profile=halo_profile,
                    cm_relation=cm_relation, bias_model=bias_model,
                    r=r, nonlinear=nonlinear, scale_dependent_bias=scale_dependent_bias,
                    halo_exclusion=halo_exclusion, **combined_dict)


    def update(self, **kwargs):
        """
        Updates any input parameter in the best possible way - similar to hmf.Perturbations.update
        """
        #We use a somewhat tiered approach here.
        #We first update all simple HOD parameters, then whatever is left
        #is sent to Perturbations(), which will weed out stupid values.

        #EVERYTHING affects corr_gal_2h, but its tricky because there's a loophole,
        #so we just delete it here for now.
        del self.corr_gal_2h
        #First do hod_model parameters
        hodmod_p = {k:v for k, v in kwargs.iteritems() if k in self.hodmod_params}
        if hodmod_p:
            self.hodmod_params.update(hodmod_p)
            #Completely re-do hodmod as it is trivial.
            self.hodmod = hm(**self.hodmod_params)

            #Delete the entries we've used from kwargs
            for k in hodmod_p:
                del kwargs[k]

            del self.n_cen
            del self.n_sat
            del self.n_tot

        for k in kwargs.keys():
            if k == 'halo_profile':
                self.halo_profile = kwargs.pop(k)
            elif k == 'cm_relation':
                self.cm_relation = kwargs.pop(k)
            elif k == 'bias_model':
                self.bias_model = kwargs.pop(k)
            elif k == 'r':
                self.r = kwargs.pop(k)
            elif k == 'nonlinear':
                self.nonlinear = kwargs.pop(k)
                del self.matter_power
            elif k == "halo_exclusion":
                self.halo_exclusion = kwargs.pop(k)
            elif k == "scale_dependent_bias":
                self.scale_dependent_bias = kwargs.pop(k)
                del self.corr_gal_2h

        #All that's left should be Perturbations() args
        #However, we don't want to call a new class each time,
        #we usually want to update.
        if kwargs:
            try:
                self.pert.update(cut_fit=False, **kwargs)
            except AttributeError:
                self.pert = Perturbations(cut_fit=False, **kwargs)

            del self.bias
            del self.profile
            del self.mean_gal_den
            del self.n_gal
            del self.matter_power

#===============================================================================
# Set Properties
#===============================================================================
    @property
    def halo_profile(self):
        return self.__halo_profile

    @halo_profile.setter
    def halo_profile(self, val):
        available_profs = ['nfw']
        if val not in available_profs:
            raise ValueError("halo_profile not acceptable: " + str(val))
        else:
            self.__halo_profile = val

            del self.profile
    @property
    def cm_relation(self):
        return self.__cm_relation

    @cm_relation.setter
    def cm_relation(self, val):
        available = ['duffy', 'zehavi']
        if val not in available:
            raise ValueError("cm_relation not acceptable: " + str(val))
        else:
            self.__cm_relation = val
            del self.profile
    @property
    def bias_model(self):
        return self.__bias_model

    @bias_model.setter
    def bias_model(self, val):
        available = ["ST", 'seljak', "ma", "seljak_warren", "tinker"]
        if val not in available:
            raise ValueError("bias_model not acceptable: " + str(val))
        else:
            self.__bias_model = val
            del self.bias

    @property
    def halo_exclusion(self):
        return self.__halo_exclusion

    @halo_exclusion.setter
    def halo_exclusion(self, val):
        available = [None, "sphere_zheng", "ellipsoid", "ng_matched", 'schneider']
        if val not in available:
            raise ValueError("halo_exclusion not acceptable: " + str(val))
        else:
            self.__halo_exclusion = val
            del self.corr_gal_2h
            del self.power_gal_2h
    @property
    def r(self):
        return self.__r

    @r.setter
    def r(self, val):
        try:
            if len(val) == 1:
                raise ValueError("r must be a sequence of length > 1")
        except TypeError:
            raise TypeError("r must be a sequence of length > 1")

        self.__r = np.array(val)

        del self.dm_corr

#===============================================================================
# Start the actual calculations
#===============================================================================
    @property
    def bias(self):
        """Scale-independent bias"""
        try:
            return self.__bias
        except:
            self.__bias = bias(self.pert, self.bias_model)
            return self.__bias

    @bias.deleter
    def bias(self):
        try:
            del self.__bias
            del self.power_gal_2h
        except:
            pass

    @property
    def profile(self):
        try:
            return self.__profile
        except:
            self.__profile = profiles(self.pert.cosmo_params['mean_dens'],
                                      self.pert.delta_halo,
                                      profile=self.halo_profile,
                                      cm_relation=self.cm_relation)
            return self.__profile

    @profile.deleter
    def profile(self):
        try:
            del self.__profile
            del self.power_gal_2h
            del self._power_gal_1h_ss
            del self._corr_gal_1h_ss
            del self._corr_gal_1h_cs
        except:
            pass

    @property
    def n_cen(self):
        """ The average number of central galaxies at self.M"""
        try:
            return self.__n_cen
        except:
            self.__n_cen = self.hodmod.nc(self.pert.M *
                                          self.pert.cosmo_params['H0'] / 100)
            return self.__n_cen

    @n_cen.deleter
    def n_cen(self):
        try:
            del self.__n_cen
        except:
            pass

    @property
    def n_sat(self):
        """ The average number of satellite galaxies at self.M"""
        try:
            return self.__n_sat
        except:
            self.__n_sat = self.hodmod.ns(self.pert.M *
                                          self.pert.cosmo_params['H0'] / 100)
            return self.__n_sat

    @n_sat.deleter
    def n_sat(self):
        try:
            del self.__n_sat
        except:
            pass

    @property
    def n_tot(self):
        """The total number of satellite galaxies at self.M"""
        try:
            return self.__n_tot
        except:
            self.__n_tot = self.hodmod.ntot(self.pert.M *
                                            self.pert.cosmo_params['H0'] / 100)
            return self.__n_tot

    @n_tot.deleter
    def n_tot(self):
        try:
            del self.__n_tot
            del self.mean_gal_den
            del self.n_gal
        except:
            pass

    @property
    def n_gal(self):
        """
        An array of the same length as M giving number density of galaxies at M
        """
        try:
            return self.__n_gal
        except:
            self.__n_gal = self.pert.dndm * self.n_tot
            return self.__n_gal

    @n_gal.deleter
    def n_gal(self):
        try:
            del self.__n_gal
        except:
            pass
    @property
    def mean_gal_den(self):
        """
        Gets the mean number density of galaxies
        """
        try:
            return self.__mean_gal_den
        except:
            #Integrand is just the density of galaxies at mass M
            integrand = self.pert.dndm * self.n_tot

            self.__mean_gal_den = intg.simps(integrand, self.pert.M)
            return self.__mean_gal_den

    @mean_gal_den.deleter
    def mean_gal_den(self):
        try:
            del self.__mean_gal_den
            del self.power_gal_2h
            del self._power_gal_1h_ss
            del self._corr_gal_1h_ss
            del self._corr_gal_1h_cs
        except:
            pass

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
                lnk, self.__matter_power = tools.non_linear_power(self.pert.lnkh,
                                                                  **self.pert.camb_params)
                #Now we need to normalise
                #FIXME: get index of self.lnkh which is above -5.3
                ind = np.where(self.pert.lnkh > -5.3)[0][0]
                self.__matter_power += self.pert.power[ind] - self.__matter_power[0]
                self.__matter_power = np.concatenate((self.pert.power[:ind], self.__matter_power))
            else:
                self.__matter_power = self.pert.power
            return self.__matter_power

    @matter_power.deleter
    def matter_power(self):
        try:
            del self.__matter_power
            del self.power_gal_2h
            del self.dm_corr
        except:
            pass

    @property
    def dm_corr(self):
        """
        The dark-matter-only two-point correlation function of the given cosmology
        """
        try:
            return self.__dm_corr
        except:
            self.__dm_corr = tools.power_to_corr(np.exp(self.matter_power),
                                                 self.pert.lnk, self.r)
            return self.__dm_corr

    @dm_corr.deleter
    def dm_corr(self):
        try:
            del self.__dm_corr
        except:
            pass

    @property
    def power_gal_2h(self):
        """The 2-halo term of the galaxy power spectrum - NOT LOGGED"""
        try:
            print "Just returning it"
            return self.__power_gal_2h
        except:
            print "yup i'm getting power2h"
            #TODO: add scale-dependent bias,finite max M, weird ng
            #FIXME: nonlinear power is odd??
            self.__power_gal_2h = np.zeros_like(self.pert.lnk)
            for i, lnk in enumerate(self.pert.lnk):
                u = self.profile.u(np.exp(lnk), self.pert.M , self.pert.z)
                integrand = self.n_tot * u * self.pert.dndm * self.bias.bias

                self.__power_gal_2h[i] = intg.simps(integrand, self.pert.M)

            self.__power_gal_2h = np.exp(self.matter_power) * \
                              self.__power_gal_2h ** 2 / self.mean_gal_den ** 2

            if self.halo_exclusion == "schneider":
                self.__power_gal_2h *= tools.exclusion_window(np.exp(self.pert.lnk),
                                                              r=2)

            return self.__power_gal_2h

    @power_gal_2h.deleter
    def power_gal_2h(self):
        try:
            del self.__power_gal_2h
            del self.corr_gal_2h
        except:
            pass

    def _power_gal_2h_scale_dep(self, r_index):
        """The 2-halo term of the galaxy power spectrum, dependent on scale - NOT LOGGED"""
        #TODO: add scale-dependent bias,finite max M, weird ng
        #FIXME: nonlinear power is odd??
        pg2h = np.zeros_like(self.pert.lnk)
        xi = self.dm_corr[r_index]
        if self.scale_dependent_bias:
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

#                integrand_m = integrand_m[self.pert.M < mlim]
#                m = self.pert.M[self.pert.M < mlim]
#                ng = intg.simps(integrand_m, m)

            elif self.halo_exclusion == "ng_matched":
                mmin = np.log(self.pert.M[integrand_m > 0][0])
                integrand_ms = spline(np.log(self.pert.M[integrand_m > 0]),
                                     np.log(integrand_m[integrand_m > 0]), k=1)
                def integ(m1, m2, r):
                    rv1, rv2 = tools.virial_radius(np.exp(np.array([m1, m2])),
                                                  self.pert.cosmo_params['mean_dens'],
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
                                                  self.pert.cosmo_params['mean_dens'],
                                                  self.pert.delta_halo)

                    return np.exp(m1 + m2) * np.exp(integrand(m1) + integrand(m2)) * tools.overlapping_halo_prob(r, rv1, rv2)

                pg2h[i], err = intg.dblquad(integ, np.log(m[0]), np.log(m[-1]),
                                       lambda x: np.log(m[0]), lambda x: np.log(m[-1]),
                                       args=(self.r[r_index],), epsrel=1e-3)

            if self.halo_exclusion != "ellipsoid":
                pg2h[i] = intg.simps(integrand, m)

        pg2h = np.exp(self.matter_power) * pg2h ** 2 / self.mean_gal_den ** 2

        if self.halo_exclusion == 'schneider':
            if r_index == 1:
                #should be k,r/h, but we use k/h and r (which gives the same)
                pg2h *= tools.exclusion_window(np.exp(self.pert.lnk), r=2)
        return pg2h

    @property
    def _power_gal_1h_ss(self):
        """
        The sat-sat part of the 1-halo term of the galaxy power spectrum
        """
        try:
            return self.__power_gal_1h_ss
        except:
            self.__power_gal_1h_ss = np.zeros_like(self.pert.lnk)
            for i, lnk in enumerate(self.pert.lnk):
                u = self.profile.u(np.exp(lnk), self.pert.M , self.pert.z)
                if self.hodmod_params['central']:
                    integrand = self.n_cen * self.n_sat ** 2 * u ** 2 * self.pert.dndm
                else:
                    integrand = self.n_sat ** 2 * u ** 2 * self.pert.dndm

                self.__power_gal_1h_ss[i] = intg.simps(integrand, self.pert.M)

            self.__power_gal_1h_ss /= self.mean_gal_den ** 2
            return self.__power_gal_1h_ss

    @_power_gal_1h_ss.deleter
    def _power_gal_1h_ss(self):
        try:
            del self.__power_gal_1h_ss
            del self._corr_gal_1h_ss
        except:
            pass

    @property
    def _corr_gal_1h_ss(self):
        """The 1-halo galaxy correlation for sat-sat pairs"""
        try:
            return self.__corr_gal_1h_ss
        except:
            #If profile is nfw we can skip power and go straight to corr
            if self.halo_profile == 'nfw':
                self.__corr_gal_1h_ss = np.zeros_like(self.r)
                for i, r in enumerate(self.r):
                    lam = self.profile.lam(r, self.pert.M , self.pert.z)

                    if self.hodmod_params['central']:
                        integrand = self.pert.dndm * self.n_cen * self.n_sat ** 2\
                                    * lam / self.pert.M ** 2
                    else:
                        integrand = self.pert.dndm * self.n_sat ** 2 * lam / self.pert.M ** 2
                    self.__corr_gal_1h_ss[i] = intg.simps(integrand, self.pert.M)
                self.__corr_gal_1h_ss /= self.mean_gal_den ** 2
            #Otherwise we just convert the power
            else:
                self.__corr_gal_1h_ss = tools.power_to_corr(self._power_gal_1h_ss,
                                                            self.pert.lnk, self.r)
            return self.__corr_gal_1h_ss

    @_corr_gal_1h_ss.deleter
    def _corr_gal_1h_ss(self):
        try:
            del self.__corr_gal_1h_ss
            del self.corr_gal_1h
        except:
            pass

    @property
    def _corr_gal_1h_cs(self):
        """The cen-sat part of the 1-halo galaxy correlations"""
        try:
            return self.__corr_gal_1h_cs
        except:
            self.__corr_gal_1h_cs = np.zeros_like(self.r)
            for i, r in enumerate(self.r):

                rho = self.profile.rho(r, self.pert.M , self.pert.z)

                integrand = self.pert.dndm * 2 * self.n_cen * self.n_sat * rho
                m_min = tools.virial_mass(r, self.pert.cosmo_params['mean_dens'],
                                          self.pert.delta_halo)
                integrand[self.pert.M < m_min] = 0
                self.__corr_gal_1h_cs[i] = intg.simps(integrand,
                                                      self.pert.M)
            self.__corr_gal_1h_cs /= self.mean_gal_den ** 2
            return self.__corr_gal_1h_cs

    @_corr_gal_1h_cs.deleter
    def _corr_gal_1h_cs(self):
        try:
            del self.__corr_gal_1h_cs
            del self.corr_gal_1h
        except:
            pass

    @property
    def corr_gal_1h(self):
        """The 1-halo term of the galaxy correlations"""
        try:
            return self.__corr_gal_1h
        except:
            self.__corr_gal_1h = self._corr_gal_1h_cs + self._corr_gal_1h_ss
            return self.__corr_gal_1h

    @corr_gal_1h.deleter
    def corr_gal_1h(self):
        try:
            del self.__corr_gal_1h
            del self.corr_gal
        except:
            pass

    @property
    def corr_gal_2h(self):
        """The 2-halo term of the galaxy correlation"""
        try:
            return self.__corr_gal_2h
        except:
            if self.scale_dependent_bias or (self.halo_exclusion and self.halo_exclusion != 'schneider'):
                self.__corr_gal_2h = np.zeros_like(self.r)
                for i, r in enumerate(self.r):
                    power = self._power_gal_2h_scale_dep(i)

                    self.__corr_gal_2h[i] = tools.power_to_corr(power,
                                                                self.pert.lnk, r)[0]

            else:
                self.__corr_gal_2h = tools.power_to_corr(self.power_gal_2h,
                                                     self.pert.lnk, self.r)
            return self.__corr_gal_2h

    @corr_gal_2h.deleter
    def corr_gal_2h(self):
        try:
            del self.__corr_gal_2h
            del self.corr_gal
        except:
            pass

    @property
    def  corr_gal(self):
        """The galaxy correlation function"""
        try:
            return self.__corr_gal
        except:
            self.__corr_gal = self.corr_gal_1h + self.corr_gal_2h

            return self.__corr_gal

    @corr_gal.deleter
    def corr_gal(self):
        try:
            del self.__corr_gal
        except:
            pass

