#===============================================================================
# Some Imports
#===============================================================================
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import scipy.integrate as intg
import numpy as np
from scipy.optimize import minimize
# import scipy.special as sp

from hmf import MassFunction
from hmf._cache import cached_property, parameter
# import hmf.tools as ht
import tools
import hod
from concentration import CMRelation
from halo_exclusion import Exclusion, NoExclusion
from fort.routines import hod_routines as fort
from copy import copy,deepcopy
from numpy import issubclass_
from hmf._framework import get_model,get_model_
import profiles
import bias
from hmf.filters import TopHat
import warnings

USEFORT = False

## TODO: I probably need to split this class into two: one for pure matter HaloModel,
##       one for galaxies (inheriting). This is because the Mmin setting should be
##       different for them both.
#===============================================================================
# The class itself
#===============================================================================
class HaloModel(MassFunction):
    '''
    Calculates several quantities using the halo model.

    Parameters
    ----------
    r : array_like, optional, default ``np.logspace(-2.0,1.5,100)``
        The scales at which the correlation function is calculated in Mpc/*h*

    **kwargs: anything that can be used in the MassFunction class

    '''
    rlog = True

    def __init__(self, rmin=0.1, rmax=50.0, rnum=20,
                 hod_model="Zehavi05",hod_params={},
                 profile_model='NFW', profile_params={},
                 concentration_model='Duffy08', concentration_params={},
                 bias_model="Tinker10",bias_params={},
                 sd_bias_model="Tinker_SD05",sd_bias_params={},
                 exclusion_model="NgMatched",exclusion_params={},
                 hc_spectrum="nonlinear",ng=None,Mmax=18,
                 ** hmf_kwargs):

        # Do Mass Function __init__ MUST BE DONE FIRST (to init Cache)
        super(HaloModel, self).__init__(Mmax=Mmax, **hmf_kwargs)

        # Initially save parameters to the class.
        self.hod_params = hod_params
        self.hod_model = hod_model
        self.profile_model,self.profile_params = profile_model,profile_params
        self.concentration_model,self.concentration_params =  concentration_model,concentration_params
        self.bias_model,self.bias_params = bias_model,bias_params
        self.sd_bias_model, self.sd_bias_params = sd_bias_model,sd_bias_params
        self.exclusion_model,self.exclusion_params = exclusion_model,exclusion_params

        self.rmin = rmin
        self.rmax = rmax
        self.rnum = rnum
        self.hc_spectrum = hc_spectrum
        # A special argument, making it possible to define M_min by mean density
        self.ng = ng

        # Find mmin if we want to
        if ng is not None:
            mmin = self._find_m_min(ng)
            self.hod_params = {"M_min":mmin}


    def update(self, **kwargs):
        """
        Updates any parameter passed
        """
        if "ng" in kwargs:
            self.ng = kwargs.pop('ng')
        elif "hod_params" in kwargs:
            if "M_min" in kwargs["hod_params"]:
                self.ng = None

        super(HaloModel, self).update(**kwargs)

        if self.ng is not None:
            mmin = self._find_m_min(self.ng)
            self.hod_params = {"M_min":mmin}

#===============================================================================
# Parameters
#===============================================================================
    @parameter
    def cut_fit(self, val):
        return False

    @parameter
    def ng(self, val):
        """Mean density of galaxies, ONLY if passed directly"""
        return val

    @parameter
    def bias_params(self, val):
        return val

    @parameter
    def hc_spectrum(self, val):
        return val

    @parameter
    def sd_bias_params(self, val):
        return val

    @parameter
    def hod_params(self, val):
        """Dictionary of parameters for the HOD model"""
        return val

    @parameter
    def hod_model(self, val):
        """:class:`~hod.HOD` class"""
        if not isinstance(val, basestring) and not issubclass_(val, hod.HOD):
            raise ValueError("hod_model must be a subclass of hod.HOD")
        return val

    @parameter
    def nonlinear(self, val):
        """Logical indicating whether the power is to be nonlinear or not"""
        try:
            if val:
                return True
            else:
                return False
        except:
            raise ValueError("nonlinear must be a logical value")

    @parameter
    def profile_params(self, val):
        """Dictionary of parameters for the Profile model"""
        return val

    @parameter
    def profile_model(self, val):
        """The halo density profile model"""
        if not isinstance(val, basestring) and not issubclass_(val, profiles.Profile):
            raise ValueError("profile_model must be a subclass of profiles.Profile")

        return get_model_(val,"halomod.profiles")

    @parameter
    def concentration_model(self, val):
        """A concentration-mass relation"""
        if not isinstance(val, basestring) and not issubclass_(val, CMRelation):
            raise ValueError("concentration_model must be a subclass of concentration.CMRelation")
        return val

    @parameter
    def bias_model(self, val):
        if not isinstance(val, basestring) and not issubclass_(val, bias.Bias):
            raise ValueError("bias_model must be a subclass of bias.Bias")
        return val

    @parameter
    def exclusion_params(self, val):
        """Dictionary of parameters for the Exclusion model"""
        return val

    @parameter
    def exclusion_model(self, val):
        """A string identifier for the type of halo exclusion used (or None)"""
        if val is None:
            val = "NoExclusion"

        if issubclass_(val, Exclusion):
            return val
        else:
            return get_model_(val, "halomod.halo_exclusion")

    @parameter
    def concentration_params(self, val):
        return val

    @parameter
    def rmin(self, val):
        return val

    @parameter
    def rmax(self, val):
        return val

    @parameter
    def rnum(self, val):
        return val

    @parameter
    def sd_bias_model(self, val):
        if not isinstance(val, basestring) and not issubclass_(val, bias.ScaleDepBias) and val is not None:
            raise ValueError("scale_dependenent_bias must be a subclass of bias.ScaleDepBias")
        return val

    #===========================================================================
    # Basic Quantities
    #===========================================================================
    @cached_property("rmin", "rmax", "rnum")
    def r(self):
        if type(self.rmin) == list or type(self.rmin) == np.ndarray:
            r = np.array(self.rmin)
        else:
            if self.rlog:
                r = np.exp(np.linspace(np.log(self.rmin), np.log(self.rmax), self.rnum))
            else:
                r = np.linspace(self.rmin, self.rmax, self.rnum)

        return r

    @cached_property("hod_model", "hod_params")
    def hod(self):
        if issubclass_(self.hod_model, hod.HOD):
            return self.hod_model(**self.hod_params)
        else:
            return get_model(self.hod_model, "halomod.hod", **self.hod_params)

    @cached_property("hod", "Mmax",  "dlog10m")
    def m(self):
        if self.Mmax < 17:
            warnings.warn("Mmax is less than 10^17 Msun/h, so integrations *may not* converge")
        return 10 ** np.arange(self.Mmin, self.Mmax, self.dlog10m)

    @cached_property("m",'hod')
    def _gm(self):
        """
        A galaxy mask -- i.e. a mask on mass which restricts the range to those where galaxies exist
        for the given HOD.
        """
        return self.m >= self.hod.mmin

    @cached_property("bias_model", "nu", "delta_c", "delta_halo", "n", "bias_params")
    def bias(self):
        """A class containing the elements necessary to calculate the halo bias"""

        if issubclass_(self.bias_model, bias.Bias):
            return self.bias_model(nu=self.nu, delta_c=self.delta_c,
                                   m=self.m,mstar=self.mass_nonlinear,
                                   delta_halo=self.delta_halo, n=self.n,Om0=self.cosmo.Om0,
                                   h = self.cosmo.h,sigma_8=self.sigma_8,
                                   **self.bias_params).bias()
        else:
            #FIXME: this is an ugly hack just to get things fast for the paper.
            if self.bias_model in ["Jing98","Seljak04"]:
                mstar = self.mass_nonlinear
            else:
                mstar = None
            return get_model(self.bias_model, "halomod.bias",
                             nu=self.nu, delta_c=self.delta_c,
                             m=self.m,mstar=mstar,
                             delta_halo=self.delta_halo, n=self.n,Om0=self.cosmo.Om0,
                             h = self.cosmo.h,sigma_8=self.sigma_8,
                             **self.bias_params).bias()

    @cached_property("concentration_model", "filter", "_power0", "mean_density0", "concentration_params",
                     "growth","delta_c")
    def cm(self):
        """A class containing the elements necessary to calculate the concentration-mass relation"""
        this_filter = copy(self.filter)
        this_filter.power = self._power0
        this_profile = self.profile_model(None,self.mean_density0, self.delta_halo, self.z, **self.profile_params)

        if issubclass_(self.concentration_model, CMRelation):
            return self.concentration_model(filter0=this_filter, mean_density0=self.mean_density0,
                                    growth=self.growth,delta_c=self.delta_c,rhos=this_profile._rho_s,
                                    cosmo = self.cosmo,
                                    **self.concentration_params)
        else:
            return get_model(self.concentration_model, "halomod.concentration",
                            filter0=this_filter, mean_density0=self.mean_density0,
                            growth=self.growth,delta_c=self.delta_c,rhos=this_profile._rho_s,
                            cosmo = self.cosmo,
                            **self.concentration_params)

    @cached_property("cm","m","z")
    def concentration(self):
        """
        The concentrations corresponding to `.m`
        """
        return self.cm.cm(self.m,self.z)

    @cached_property("profile_model", "profile_params","delta_halo", "cm", "z", "mean_density0")
    def profile(self):
        """A class containing the elements necessary to calculate halo profile quantities"""
        if issubclass_(self.profile_model, profiles.Profile):
            return self.profile_model(cm_relation=self.cm,
                                     mean_dens=self.mean_density0,
                                     delta_halo=self.delta_halo, z=self.z,
                                     **self.profile_params)
        else:
            return get_model(self.profile_model, "halomod.profiles",
                             cm_relation=self.cm,
                             mean_dens=self.mean_density0,
                             delta_halo=self.delta_halo, z=self.z,
                             **self.profile_params)

    @cached_property("sd_bias_model","corr_mm_base","sd_bias_params")
    def sd_bias(self):
        """A class containing relevant methods to calculate scale-dependent bias corrections"""
        if self.sd_bias_model is None:
            return None
        elif issubclass_(self.sd_bias_model, bias.ScaleDepBias):
            return self.sd_bias_model(self.corr_mm_base,**self.sd_bias_params)
        else:
            return get_model(self.sd_bias_model, "halomod.bias",
                             xi_dm = self.corr_mm_base,**self.sd_bias_params)

    #===========================================================================
    # Basic HOD Quantities
    #===========================================================================
    @cached_property("hod", "m")
    def n_sat(self):
        """Average satellite occupancy of halo of mass m"""
        return self.hod.ns(self.m)

    @cached_property("hod", "m")
    def n_cen(self):
        """Average satellite occupancy of halo of mass m"""
        return self.hod.nc(self.m)

    @cached_property("hod", "m")
    def n_tot(self):
        """Average satellite occupancy of halo of mass m"""
        return self.hod.ntot(self.m)

    #===========================================================================
    # Derived DM Quantities
    #===========================================================================
    # @cached_property("m","dndm","dlog10m")
    # def mean_matter_density(self):
    #     """
    #     The mean matter density in halos with mass greater than ``m[0]``.
    #     """
    #     return self.rho_gtm[0]

    @cached_property("m","dndm","dlog10m","bias","mean_density0")
    def bias_effective_matter(self):
        """
        The effective bias on linear scales for dark matter
        """
        integrand = self.m**2 * self.dndm * self.bias
        return intg.trapz(integrand,dx=np.log(10)*self.dlog10m)/self.mean_density0


    #===========================================================================
    # Derived HOD Quantities
    #===========================================================================
    @cached_property("m", "dndm", "n_tot", "ng")
    def mean_gal_den(self):
        """
        The mean number density of galaxies
        """
        if self.ng is not None:
            return self.ng
        else:
            integrand = self.m[self._gm] * self.dndm[self._gm] * self.n_tot[self._gm]
        return intg.trapz(integrand, dx=np.log(self.m[1] / self.m[0]))


    @cached_property("m", "dndm", "n_tot", "bias")
    def bias_effective(self):
        """
        The galaxy number weighted halo bias factor (Tinker 2005)
        """
        # Integrand is just the density of galaxies at mass m by bias
        integrand = self.m[self._gm] * self.dndm[self._gm] * self.n_tot[self._gm] * self.bias[self._gm]
        b = intg.trapz(integrand, dx=np.log(self.m[1] / self.m[0]))
        return b / self.mean_gal_den

    @cached_property("m", 'dndm', 'n_tot', "mean_gal_den")
    def mass_effective(self):
        """
        Average group halo mass, or host-halo mass (in log10 units)
        """
        # Integrand is just the density of galaxies at mass m by m
        integrand = self.m[self._gm] ** 2 * self.dndm[self._gm] * self.n_tot[self._gm]

        m = intg.trapz(integrand, dx=np.log(self.m[1] / self.m[0]))
        return np.log10((m / self.mean_gal_den))

    @cached_property("m", "dndm", "n_sat", "mean_gal_den")
    def satellite_fraction(self):
        # Integrand is just the density of satellite galaxies at mass m
        integrand = self.m[self._gm] * self.dndm[self._gm] * self.n_sat[self._gm]
        s = intg.trapz(integrand, dx=np.log(self.m[1] / self.m[0]))
        return s / self.mean_gal_den

    @cached_property("satellite_fraction")
    def central_fraction(self):
        return 1 - self.satellite_fraction

    @cached_property("dndm", "n_tot")
    def n_gal(self):
        """
        The total number density of galaxies in halos of mass m
        """
        return self.dndm * self.n_tot

    #===========================================================================
    # Halo/DM Statistics
    #===========================================================================
    @cached_property("power", "nonlinear_power","hc_spectrum")
    def _power_halo_centres(self):
        """
        Power spectrum of halo centres, unbiased.

        This defines the halo-centre power spectrum, which is a part of the 2-halo
        term calculation. Formally, we make the assumption that the halo-centre
        power spectrum is linearly biased, and this function returns

        .. math :: P^{hh}_c (k) /(b_1(m_1)b_2(m_2))

        This should probably be expanded to its own component module.
        """
        if self.hc_spectrum == "linear":
            return self.power
        elif self.hc_spectrum == "nonlinear":
            return self.nonlinear_power
        elif self.hc_spectrum == "filtered-nl":
            f = TopHat(None, None)
            return self.nonlinear_power * f.k_space(self.k * 2.0)

    @cached_property("power",'k','r')
    def corr_mm_lin(self):
        return tools.power_to_corr_ogata(self.power,self.k,self.r)

    @cached_property("nonlinear_power",'k','r')
    def corr_mm_halofit(self):
        return tools.power_to_corr_ogata(self.nonlinear_power,self.k,self.r)

    @cached_property("corr_mm_lin","corr_mm_halofit","hc_spectrum")
    def corr_mm_base(self):
        "The matter correlation function used throughout the calculations"
        if self.hc_spectrum=="linear":
            return self.corr_mm_lin
        elif self.hc_spectrum=="nonlinear" or self.hc_spectrum=="filtered-nl":
            return self.corr_mm_halofit

    #===========================================================================
    # 2-point DM statistics
    #===========================================================================
    @cached_property("profile","dndm","m","k","mean_density0","dlog10m",
                     "delta_halo")
    def power_mm_1h(self):
        """
        The halo model-derived nonlinear 1-halo matter power
        """
        u = self.profile.u(self.k, self.m, norm = "m")
        integrand = self.dndm * self.m**3  * u**2

        ### The following may not need to be done?
        # TODO: investigate what on earth to do here.
        # Basically, you need the 1-halo term to turn over at small k
        # Otherwise, it becomes larger than the 2-halo term
        # But this only occurs at like 10^-4 h/Mpc which is typically beyond range.
        r = np.pi/self.k/10 # The 10 is a complete heuristic hack.
        mmin = 4*np.pi * r**3 * self.mean_density0 * self.delta_halo/3
        mask = np.outer(self.m,np.ones_like(self.k)) < mmin
        integrand[mask.T] = 0

        return intg.trapz(integrand,dx=np.log(10)*self.dlog10m)/self.mean_density0**2

    @cached_property("profile","dndm","m","r","mean_density0","delta_halo","dlog10m")
    def corr_mm_1h(self):
        """
        The halo model-derived nonlinear 1-halo matter power
        """
        if self.profile.has_lam:
            lam = self.profile.lam(self.r, self.m, norm="m")
            integrand = self.dndm * self.m ** 3 * lam

            return intg.trapz(integrand,dx=np.log(10)*self.dlog10m)/self.mean_density0**2
        else:
            return tools.power_to_corr_ogata(self.power_mm_1h,self.k,self.r)

    @cached_property("profile","k","m","sd_bias_model","sd_bias","bias",
                     "exclusion_model","dndlnm",'r',"delta_halo","mean_density0",
                     "rho_gtm","_power_halo_centres")
    def power_mm_2h(self):
        "The 2-halo matter power spectrum"
        # TODO: check what to do here.
        # Basically, HMcode assumes that the large-scale power is equivalent
        # to the linear power, with no biasing. I think this *has* to be true
        # since the matter power is for *all* mass. But other codes (eg. chomp)
        # do the normal integral which includes biasing...
        if self.exclusion_model != NoExclusion:
            u = self.profile.u(self.k,self.m,norm="m")
            bias= np.ones_like(self.m)
            inst = self.exclusion_model(m=self.m,density=self.dndlnm,
                                        I=self.dndlnm*u/self.rho_gtm[0],bias=bias,r=self.r,
                                        delta_halo=self.delta_halo,
                                        mean_density=self.mean_density0,
                                        **self.exclusion_params)
            mult = inst.integrate()
        else:
            inst = 0
            mult = 1
        if hasattr(inst,"density_mod"):
            self.__density_mod = inst.density_mod
        else:
            self.__density_mod = self.mean_density0

        return mult * self._power_halo_centres

    @cached_property("power_mm_2h","k","r","rho_gtm","mean_density0")
    def corr_mm_2h(self):
        if len(self.power_mm_2h.shape)==2:
            corr = tools.power_to_corr_ogata_matrix(self.power_mm_2h,self.k,self.r)
        else:
            corr = tools.power_to_corr_ogata(self.power_mm_2h,self.k,self.r)

        ## modify by the new density
        return (self.__density_mod/self.mean_density0)**2 * (1+corr)-1

    @cached_property("corr_mm_1h", "corr_mm_2h")
    def corr_mm(self):
        """The halo-model-derived matter correlation function"""
        return self.corr_mm_1h + self.corr_mm_2h

    @cached_property("power_mm_1h","power_mm_2h")
    def power_mm(self):
        """ The halo-model-derived power spectrum """
        return self.power_mm_1h + self.power_mm_2h

    #===========================================================================
    # 2-point galaxy-galaxy (HOD) statistics
    #===========================================================================
    @cached_property("k", "m", "dndm", "n_sat", "n_cen", 'hod', 'profile', "mean_gal_den")
    def power_gg_1h_ss(self):
        """
        The sat-sat part of the 1-halo term of the galaxy power spectrum
        """
        if USEFORT:
            ## The fortran routine is very very slightly faster. Should remove it.
            u = self.profile.u(self.k, self.m[self._gm], norm='m')
            p = fort.power_gal_1h_ss(nlnk=len(self.k),
                                     nm=len(self.m[self._gm]),
                                     u=np.asfortranarray(u),
                                     dndm=self.dndm[self._gm],
                                     nsat=self.n_sat[self._gm],
                                     ncen=self.n_cen[self._gm],
                                     mass=self.m[self._gm],
                                     central=self.hod._central)
        else:
            u = self.profile.u(self.k, self.m[self._gm], norm='m')
            integ = u**2 * self.dndm[self._gm] * self.m[self._gm] * self.n_sat[self._gm]**2
            if self.hod._central:
                integ *= self.n_cen[self._gm]

            ### The following may not need to be done?
            # TODO: investigate what on earth to do here.
            # Basically, you need the 1-halo term to turn over at small k
            # Otherwise, it becomes larger than the 2-halo term
            # But this only occurs at like 10^-4 h/Mpc which is typically beyond range.
            r = np.pi/self.k/10  # The 10 is a complete heuristic hack.
            mmin = 4*np.pi*r ** 3*self.mean_density0*self.delta_halo/3
            mask = np.outer(self.m[self._gm], np.ones_like(self.k)) < mmin
            integ[mask.T] = 0

            p = intg.trapz(integ,dx=self.dlog10m*np.log(10))

        return p / self.mean_gal_den ** 2

    @cached_property("power_gg_1h_ss", "k", "r")
    def corr_gg_1h_ss(self):
        if self.profile.has_lam:
            lam = self.profile.lam(self.r, self.m[self._gm], norm="m")
            integ = self.m[self._gm]* self.dndm[self._gm] * self.n_sat[self._gm]**2* lam
            if self.hod._central:
                integ *= self.n_cen[self._gm]

            c = intg.trapz(integ,dx=self.dlog10m*np.log(10))

            return c / self.mean_gal_den ** 2
        else:
            return tools.power_to_corr_ogata(self.power_gg_1h_ss,
                                             self.k, self.r)

    @cached_property("k","m","dndm","n_cen","n_sat","profile","dlog10m",
                     'mean_gal_den')
    def power_gg_1h_cs(self):
        """The cen-sat part of the 1-halo galaxy-galaxy power"""
        u = self.profile.u(self.k,self.m[self._gm],norm="m")
        integ = self.dndm[self._gm] * 2 * self.n_cen[self._gm] * self.n_sat[self._gm] * u * self.m[self._gm]

        ### The following may not need to be done?
        # TODO: investigate what on earth to do here.
        # Basically, you need the 1-halo term to turn over at small k
        # Otherwise, it becomes larger than the 2-halo term
        # But this only occurs at like 10^-4 h/Mpc which is typically beyond range.
        r = np.pi/self.k/10 # The 10 is a complete heuristic hack.
        mmin = 4*np.pi * r**3 * self.mean_density0 * self.delta_halo/3
        mask = np.outer(self.m[self._gm],np.ones_like(self.k)) < mmin
        integ[mask.T] = 0

        c = intg.trapz(integ,dx=self.dlog10m*np.log(10))
        return c/self.mean_gal_den**2

    @cached_property("power_gg_1h_cs","power_gg_1h_ss")
    def power_gg_1h(self):
        """
        Total 1-halo galaxy power.
        """
        return self.power_gg_1h_cs + self.power_gg_1h_ss

    @cached_property("power_gg_1h","power_gg_2h")
    def power_gg(self):
        """
        Total galaxy power
        """
        return self.power_gg_1h+self.power_gg_2h

    @cached_property("r", "m", "dndm", "n_cen", "n_sat", "mean_density0",
                     "delta_halo", "mean_gal_den")
    def corr_gg_1h_cs(self):
        """The cen-sat part of the 1-halo galaxy correlations"""
        rho = self.profile.rho(self.r, self.m[self._gm], norm="m")
        if USEFORT:
            c = fort.corr_gal_1h_cs(nr=len(self.r),
                                    nm=len(self.m[self._gm]),
                                    r=self.r,
                                    mass=self.m[self._gm],
                                    dndm=self.dndm[self._gm],
                                    ncen=self.n_cen[self._gm],
                                    nsat=self.n_sat[self._gm],
                                    rho=np.asfortranarray(rho),
                                    mean_dens=self.mean_density0,
                                    delta_halo=self.delta_halo)
        else:
            #mmin = 4*np.pi * self.r**3 * self.mean_density0 * self.delta_halo/3
            #mask = np.repeat(self.m,len(self.r)).reshape(len(self.m),len(self.r)) < mmin
            integ = self.dndm[self._gm] * 2 * self.n_cen[self._gm] * self.n_sat[self._gm] * rho * self.m[self._gm]
            #integ[mask.T] = 0
            c = intg.trapz(integ,dx=self.dlog10m*np.log(10))

        return c / self.mean_gal_den ** 2


    @cached_property("r", "m", "dndm", "n_cen", "n_sat", "hod", "mean_density0", "delta_halo",
                     "mean_gal_den", "corr_gg_1h_cs", "corr_gg_1h_ss")
    def corr_gg_1h(self):
        """The 1-halo term of the galaxy correlations"""
        if self.profile.has_lam:
            rho = self.profile.rho(self.r, self.m[self._gm], norm="m")
            lam = self.profile.lam(self.r, self.m[self._gm], norm="m")
            if USEFORT:
                ## Using fortran only saves about 15% of time on this single routine (eg. 7ms --> 8.7ms)
                c = fort.corr_gal_1h(nr=len(self.r),
                                     nm=len(self.m[self._gm]),
                                     r=self.r,
                                     mass=self.m[self._gm],
                                     dndm=self.dndm[self._gm],
                                     ncen=self.n_cen[self._gm],
                                     nsat=self.n_sat[self._gm],
                                     rho=np.asfortranarray(rho),
                                     lam=np.asfortranarray(lam),
                                     central=self.hod._central,
                                     mean_dens=self.mean_density0,
                                     delta_halo=self.delta_halo)
            else:
                integ = self.m[self._gm]* self.dndm[self._gm] * self.n_sat[self._gm]*(self.n_sat[self._gm] * lam + 2*rho)
                if self.hod._central:
                    integ *= self.n_cen[self._gm]

                c = intg.trapz(integ,dx=self.dlog10m*np.log(10))

            return c / self.mean_gal_den ** 2 -1

        else:
            return self.corr_gg_1h_cs + self.corr_gg_1h_ss -1

    @cached_property("profile","k","m","sd_bias_model","sd_bias","bias",
                     "exclusion_model","dndlnm",'r',"delta_halo","mean_density0",
                     "rho_gtm","_power_halo_centres")
    def power_gg_2h(self):
        u = self.profile.u(self.k,self.m[self._gm],norm="m")
        if self.sd_bias_model is not None:
            bias = np.outer(self.sd_bias.bias_scale(),self.bias)
        else:
            bias = self.bias
        inst = self.exclusion_model(m=self.m[self._gm],density=self.n_tot[self._gm]*self.dndm[self._gm],
                                    I=self.n_tot[self._gm]*self.dndm[self._gm]*u/self.mean_gal_den,
                                    bias=bias[self._gm],r=self.r,delta_halo=self.delta_halo,
                                    mean_density=self.mean_density0,
                                    **self.exclusion_params)

        if hasattr(inst,"density_mod"):
            self.__density_mod = inst.density_mod
        else:
            self.__density_mod = np.ones_like(self.r) * self.mean_gal_den

        return inst.integrate() * self._power_halo_centres

    @cached_property("profile", "k", "m", "exclusion_model", "sd_bias_model",
                     "bias", "r", "mean_gal_den","power_gg_2h")
    def corr_gg_2h(self):
        """The 2-halo term of the galaxy correlation"""
        # if USEFORT:
        #     u = self.profile.u(self.k, self.m , norm='m')
        #     corr = thalo(self.exclusion_model, self.sd_bias_model,
        #                  self.m, self.bias, self.n_tot,
        #                  self.dndm, np.log(self.k),
        #                  self._power_halo_centres, u, self.r, self.corr_mm_base,
        #                  self.mean_gal_den, self.delta_halo,
        #                  self.mean_density0, 1)
        # else:
        if len(self.power_gg_2h.shape)==2:
            corr = tools.power_to_corr_ogata_matrix(self.power_gg_2h,self.k,self.r)
        else:
            corr = tools.power_to_corr_ogata(self.power_gg_2h,self.k,self.r)

        ## modify by the new density. This step is *extremely* sensitive to the
        ## exact value of __density_mod at large scales, where the ratio *should*
        ## be exactly 1.
        if self.r[-1] > 2*self.profile._mvir_to_rvir(self.m[-1]):
            try:
                self.__density_mod *= self.mean_gal_den/self.__density_mod[-1]
            except TypeError:
                pass
        return (self.__density_mod/self.mean_gal_den)**2 * (1+corr)-1
        # return corr

    @cached_property("corr_gg_1h", "corr_gg_2h")
    def corr_gg(self):
        """The galaxy correlation function"""
        return self.corr_gg_1h + self.corr_gg_2h + 1

    #===========================================================================
    # Other utilities
    #===========================================================================
    def _find_m_min(self, ng):
        """
        Calculate the minimum mass of a halo to contain a (central) galaxy
        based on a known mean galaxy density
        """

        self.power  # This just makes sure the power is gotten and copied
        c = deepcopy(self)
        c.update(hod_params={"M_min":8}, dlog10m=0.01)

        integrand = c.m * c.dndm * c.n_tot

        if self.hod.sharp_cut:
            integral = intg.cumtrapz(integrand[::-1], dx=np.log(c.m[1] / c.m[0]))

            if integral[-1] < ng:
                raise NGException("Maximum mean galaxy density exceeded: " + str(integral[-1]))

            ind = np.where(integral > ng)[0][0]

            m = c.m[::-1][1:][max(ind - 4, 0):min(ind + 4, len(c.m))]
            integral = integral[max(ind - 4, 0):min(ind + 4, len(c.m))]


            spline_int = spline(np.log(integral), np.log(m), k=3)
            mmin = spline_int(np.log(ng)) / np.log(10)
        else:
            # Anything else requires us to do some optimization unfortunately.
            integral = intg.simps(integrand, dx=np.log(c.m[1] / c.m[0]))
            if integral < ng:
                raise NGException("Maximum mean galaxy density exceeded: " + str(integral))

            def model(mmin):
                c.update(hod_params={"M_min":mmin})
                integrand = c.m * c.dndm * c.n_tot
                integral = intg.simps(integrand, dx=np.log(c.m[1] / c.m[0]))
                return abs(integral - ng)

            res = minimize(model, 12.0, tol=1e-3,
                           method="Nelder-Mead", options={"maxiter":200})
            mmin = res.x[0]

        return mmin


class NGException(Exception):
    pass
