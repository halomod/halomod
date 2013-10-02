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
    def __init__(self, M=np.linspace(8, 18, 1001), M_1=10 ** 12.851,
                 M_c=10 ** 12.0, alpha=1.049, M_min=10 ** 11.6222,
                 gauss_width=0.26, M_0=10 ** 11.5047, fca=0.5, fcb=0, fs=1,
                 delta=None, x=1, hod_model='zehavi',
                 halo_profile='nfw', cm_relation='duffy', bias_model='tinker',
                 r=np.linspace(1, 200, 200), central=False, ** pert_kwargs):

        #A dictionary of all HOD parameters
        self.hodmod_params = {"M_1":M_1,
                              "M_c":M_c,
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
                    r=r, **combined_dict)


    def update(self, **kwargs):
        """
        Updates any input parameter in the best possible way - similar to hmf.Perturbations.update
        """
        #We use a somewhat tiered approach here.
        #We first update all simple HOD parameters, then whatever is left
        #is sent to Perturbations(), which will weed out stupid values.

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
                del self.dm_corr

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
    def dm_corr(self):
        """
        The dark-matter-only two-point correlation function of the given cosmology
        """
        try:
            return self.__dm_corr
        except:
            self.__dm_corr = tools.power_to_corr(self.pert.power,
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
            return self.__power_gal_2h
        except:
            #TODO: add scale-dependent bias, non-linear power, finite max M, weird ng
            self.__power_gal_2h = np.zeros_like(self.pert.lnk)
            for i, lnk in enumerate(self.pert.lnk):
                u = self.profile.u(np.exp(lnk), self.pert.M , self.pert.z)
                integrand = self.n_tot * u * self.pert.dndm * self.bias.bias

                self.__power_gal_2h[i] = intg.simps(integrand, self.pert.M)

            self.__power_gal_2h = np.exp(self.pert.power) * \
                                  self.__power_gal_2h ** 2 / self.mean_gal_den ** 2
            return self.__power_gal_2h

    @power_gal_2h.deleter
    def power_gal_2h(self):
        try:
            del self.__power_gal_2h
            del self.corr_gal_2h
        except:
            pass


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
                self.__corr_gal_1h_ss = tools.power_to_corr(np.log(self._power_gal_1h_ss),
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

                self.__corr_gal_1h_cs[i] = intg.simps(integrand, self.pert.M)
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
            self.__corr_gal_2h = tools.power_to_corr(np.log(self.power_gal_2h),
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

