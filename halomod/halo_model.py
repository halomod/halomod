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
from fort.routines import hod_routines as fort
from twohalo_wrapper import twohalo_wrapper as thalo
from twohalo_wrapper import dblsimps
# from hmf.filters import TopHat
from copy import deepcopy
from numpy import issubclass_
from hmf._framework import get_model
import profiles
import bias

USEFORT = True
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

    def __init__(self, rmin=0.1, rmax=50.0, rnum=20, hod_params={},
                 hod_model="Zehavi05",
                 halo_profile='NFW', cm_relation='duffy', bias_model="Tinker10",
                 nonlinear=True, scale_dependent_bias=True,
                 halo_exclusion="None", ng=None, nthreads_2halo=0,
                 proj_limit=None, bias_params={}, cm_params={}, ** hmf_kwargs):

        # Pre-process Some Arguments
        if "cut_fit" not in hmf_kwargs:
            hmf_kwargs.update({"cut_fit":False})

        # Do Mass Function __init__ MUST BE DONE FIRST (to init Cache)
        super(HaloModel, self).__init__(**hmf_kwargs)

        # Initially save parameters to the class.
        self.hod_params = hod_params
        self.hod_model = hod_model
        self.halo_profile = halo_profile
        self.cm_relation = cm_relation
        self.bias_model = bias_model
        self.rmin = rmin
        self.rmax = rmax
        self.rnum = rnum
        self.nonlinear = nonlinear
        self.halo_exclusion = halo_exclusion
        self.scale_dependent_bias = scale_dependent_bias
        self.proj_limit = proj_limit
        self.bias_params = bias_params
        self.nthreads_2halo = nthreads_2halo
        self.cm_params = cm_params
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
    def ng(self, val):
        """Mean density of galaxies, ONLY if passed directly"""
        return val

    @parameter
    def bias_params(self, val):
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
    def proj_limit(self, val):
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
    def halo_profile(self, val):
        """The halo density profile"""
        if not isinstance(val, basestring) and not issubclass_(val, profiles.Profile):
            raise ValueError("halo_profile must be a subclass of profiles.Profile")
        return val

    @parameter
    def cm_relation(self, val):
        """A concentration-mass relation"""
        if not isinstance(val, basestring) and not issubclass_(val, CMRelation):
            raise ValueError("cm_relation must be a subclass of concentration.CMRelation")
        return val

    @parameter
    def bias_model(self, val):
        if not isinstance(val, basestring) and not issubclass_(val, bias.Bias):
            raise ValueError("bias_model must be a subclass of bias.Bias")
        return val

    @parameter
    def halo_exclusion(self, val):
        """A string identifier for the type of halo exclusion used (or None)"""
        if val is None:
            val = "None"
        available = ["None", "sphere", "ellipsoid", "ng_matched", 'schneider']
        if val not in available:
            raise ValueError("halo_exclusion not acceptable: " + str(val) + " " + str(type(val)))
        else:
            return val

    @parameter
    def cm_params(self, val):
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
    @cached_property("rmin", "rmax", "rnum")
    def r(self):
        if type(self.rmin) == list or type(self.rmin) == np.ndarray:
            self.r = np.array(self.rmin)
        else:
            if self.rlog:
                return np.exp(np.linspace(np.log(self.rmin), np.log(self.rmax), self.rnum))
            else:
                return np.linspace(self.rmin, self.rmax, self.rnum)

    @cached_property("hod_model", "hod_params")
    def hod(self):
        if issubclass_(self.hod_model, hod.HOD):
            return self.hod_model(**self.hod_params)
        else:
            return get_model(self.hod_model, "halomod.hod", **self.hod_params)

    @cached_property("hod", "dlog10m")
    def M(self):
        return 10 ** np.arange(self.hod.mmin, 18, self.dlog10m)

    @cached_property("hod", "M")
    def n_sat(self):
        """Average satellite occupancy of halo of mass M"""
        return self.hod.ns(self.M)

    @cached_property("hod", "M")
    def n_cen(self):
        """Average satellite occupancy of halo of mass M"""
        return self.hod.nc(self.M)

    @cached_property("hod", "M")
    def n_tot(self):
        """Average satellite occupancy of halo of mass M"""
        return self.hod.ntot(self.M)

    @cached_property("bias_model", "nu", "delta_c", "delta_halo", "n", "bias_params")
    def bias(self):
        """A class containing the elements necessary to calculate the halo bias"""
        if issubclass_(self.bias_model, bias.Bias):
            return self.bias_model(nu=self.nu, delta_c=self.delta_c,
                                   delta_halo=self.delta_halo, n=self.n,
                                   **self.bias_params)
        else:
            return get_model(self.bias_model, "halomod.bias",
                             nu=self.nu, delta_c=self.delta_c,
                             delta_halo=self.delta_halo, n=self.n,
                             **self.bias_params)

    @cached_property("cm_relation", "nu", "z", "growth_model", "cosmolopy_dict",
                    "cm_params")
    def cm(self):
        """A class containing the elements necessary to calculate the concentration-mass relation"""
        if issubclass_(self.cm_relation, CMRelation):
            return self.cm_relation(nu=self.nu, z=self.z, growth=self.growth_model,
                                  M=self.M, **self.cm_params)
        else:
            return get_model(self.cm_relation, "halomod.concentration",
                           nu=self.nu, z=self.z, growth=self.growth_model,
                           M=self.M, **self.cm_params)


    @cached_property("halo_profile", "delta_halo", "cm_relation", "z", "omegam",
                     "omegav", "cm")
    def profile(self):
        """A class containing the elements necessary to calculate halo profile quantities"""
        if issubclass_(self.halo_profile, profiles.Profile):
            return self.halo_profile(cm_relation=self.cm_relation,
                                     mean_dens=self.mean_density0,
                                     delta_halo=self.delta_halo, z=self.z)
        else:
            return get_model(self.halo_profile, "halomod.profiles",
                             cm_relation=self.cm_relation,
                             mean_dens=self.mean_density0,
                             delta_halo=self.delta_halo, z=self.z)

    @cached_property("dndm", "n_tot")
    def n_gal(self):
        """
        The total number density of galaxies in halos of mass M
        """
        return self.dndm * self.n_tot

    @cached_property("M", "dndm", "n_tot", "ng")
    def mean_gal_den(self):
        """
        The mean number density of galaxies
        """
        if self.ng is not None:
            return self.ng
        else:
#             Integrand is just the density of galaxies at mass M
            integrand = self.M * self.dndm * self.n_tot
        return intg.simps(integrand, dx=np.log(self.M[1]) - np.log(self.M[0]),
                          even="first")


    @cached_property("M", "dndm", "n_tot", "bias")
    def bias_effective(self):
        """
        The galaxy number weighted halo bias factor (Tinker 2005)
        """
        # Integrand is just the density of galaxies at mass M by bias
        integrand = self.M * self.dndm * self.n_tot * self.bias
        b = intg.simps(integrand, dx=np.log(self.M[1]) - np.log(self.M[0]))

        return b / self.mean_gal_den

    @cached_property("M", 'dndm', 'n_tot', "mean_gal_den")
    def mass_effective(self):
        """
        Average group halo mass, or host-halo mass (in log10 units)
        """
        # Integrand is just the density of galaxies at mass M by M
        integrand = self.M ** 2 * self.dndm * self.n_tot

        m = intg.simps(integrand, dx=np.log(self.M[1]) - np.log(self.M[0]))
        return np.log10(m / self.mean_gal_den)

    @cached_property("M", "dndm", "n_sat", "mean_gal_den")
    def satellite_fraction(self):
        # Integrand is just the density of satellite galaxies at mass M
        integrand = self.M * self.dndm * self.n_sat
        s = intg.simps(integrand, dx=np.log(self.M[1]) - np.log(self.M[0]))
        return s / self.mean_gal_den

    @cached_property("satellite_fraction")
    def central_fraction(self):
        return 1 - self.satellite_fraction

    @cached_property("nonlinear", "power", "nonlinear_power")
    def matter_power(self):
        """The matter power used in calculations -- can be linear or nonlinear
        
        .. note :: Linear power is available through :attr:`.power`
        """
        if self.nonlinear:
            return self.nonlinear_power
        else:
            return self.power

    @cached_property("matter_power", 'lnk', 'r')
    def dm_corr(self):
        """
        The dark-matter-only two-point correlation function of the given cosmology
        """
        return tools.power_to_corr_ogata(np.exp(self.matter_power),
                                         self.lnk, self.r)

    @cached_property("lnk", "M", "dndm", "n_sat", "n_cen", 'hod', 'profile', "mean_gal_den")
    def _power_gal_1h_ss(self):
        """
        The sat-sat part of the 1-halo term of the galaxy power spectrum
        """
        u = self.profile.u(np.exp(self.lnk), self.M, norm='m')
        p = fort.power_gal_1h_ss(nlnk=len(self.lnk),
                                 nm=len(self.M),
                                 u=np.asfortranarray(u),
                                 dndm=self.dndm,
                                 nsat=self.n_sat,
                                 ncen=self.n_cen,
                                 mass=self.M,
                                 central=self.hod._central)
        return p / self.mean_gal_den ** 2

    @cached_property("_power_gal_1h_ss", "lnk", "r")
    def _corr_gal_1h_ss(self):
        return tools.power_to_corr_ogata(self._power_gal_1h_ss,
                                         self.lnk, self.r)

    @cached_property("r", "M", "dndm", "n_cen", "n_sat", "mean_dens", "delta_halo", "mean_gal_den")
    def _corr_gal_1h_cs(self):
        """The cen-sat part of the 1-halo galaxy correlations"""
        rho = self.profile.rho(self.r, self.M, norm="m")
        c = fort.corr_gal_1h_cs(nr=len(self.r),
                                nm=len(self.M),
                                r=self.r,
                                mass=self.M,
                                dndm=self.dndm,
                                ncen=self.n_cen,
                                nsat=self.n_sat,
                                rho=np.asfortranarray(rho),
                                mean_dens=self.mean_dens,
                                delta_halo=self.delta_halo)
        return c / self.mean_gal_den ** 2

    @cached_property("r", "M", "dndm", "n_cen", "n_sat", "hod", "mean_dens", "delta_halo",
                     "mean_gal_den", "_corr_gal_1h_cs", "_corr_gal_1h_ss")
    def corr_gal_1h(self):
        """The 1-halo term of the galaxy correlations"""
        if self.profile.has_lam:
            rho = self.profile.rho(self.r, self.M, norm="m")
            lam = self.profile.lam(self.r, self.M, norm="m")
            c = fort.corr_gal_1h(nr=len(self.r),
                                 nm=len(self.M),
                                 r=self.r,
                                 mass=self.M,
                                 dndm=self.dndm,
                                 ncen=self.n_cen,
                                 nsat=self.n_sat,
                                 rho=np.asfortranarray(rho),
                                 lam=np.asfortranarray(lam),
                                 central=self.hod._central,
                                 mean_dens=self.mean_dens,
                                 delta_halo=self.delta_halo)

            return c / self.mean_gal_den ** 2

        else:
            return self._corr_gal_1h_cs + self._corr_gal_1h_ss

    @cached_property("profile", "lnk", "M", "halo_exclusion", "scale_dependent_bias",
                     "bias", "n_tot", 'dndm', "matter_power", "r", "dm_corr",
                     "mean_gal_den", "delta_halo", "mean_dens")
    def corr_gal_2h(self):
        """The 2-halo term of the galaxy correlation"""
        u = self.profile.u(np.exp(self.lnk), self.M , norm='m')
        corr_2h = thalo(self.halo_exclusion, self.scale_dependent_bias,
                     self.M, self.bias, self.n_tot,
                     self.dndm, self.lnk,
                     np.exp(self.matter_power), u, self.r, self.dm_corr,
                     self.mean_gal_den, self.delta_halo,
                     self.mean_dens, self.nthreads_2halo)
        return corr_2h

    @cached_property("corr_gal_1h", "corr_gal_2h")
    def  corr_gal(self):
        """The galaxy correlation function"""
        return self.corr_gal_1h + self.corr_gal_2h

    def _find_m_min(self, ng):
        """
        Calculate the minimum mass of a halo to contain a (central) galaxy 
        based on a known mean galaxy density
        """

        self.power  # This just makes sure the power is gotten and copied
        c = deepcopy(self)
        c.update(hod_params={"M_min":8}, dlog10m=0.01)

        integrand = c.M * c.dndm * c.n_tot

        if self.hod.sharp_cut:
            integral = intg.cumtrapz(integrand[::-1], dx=np.log(c.M[1]) - np.log(c.M[0]))

            if integral[-1] < ng:
                raise NGException("Maximum mean galaxy density exceeded: " + str(integral[-1]))

            ind = np.where(integral > ng)[0][0]

            m = c.M[::-1][1:][max(ind - 4, 0):min(ind + 4, len(c.M))]
            integral = integral[max(ind - 4, 0):min(ind + 4, len(c.M))]


            spline_int = spline(np.log(integral), np.log(m), k=3)
            mmin = spline_int(np.log(ng)) / np.log(10)
        else:
            # Anything else requires us to do some optimization unfortunately.
            integral = intg.simps(integrand, dx=np.log(c.M[1]) - np.log(c.M[0]))
            if integral < ng:
                raise NGException("Maximum mean galaxy density exceeded: " + str(integral))

            def model(mmin):
                c.update(hod_params={"M_min":mmin})
                integrand = c.M * c.dndm * c.n_tot
                integral = intg.simps(integrand, dx=np.log(c.M[1]) - np.log(c.M[0]))
                return abs(integral - ng)

            res = minimize(model, 12.0, tol=1e-3,
                           method="Nelder-Mead", options={"maxiter":200})
            mmin = res.x[0]

        return mmin

    @cached_property("r", "corr_gal", "proj_limit")
    def projected_corr_gal(self):
        """
        Projected correlation function w(r_p).

        From Beutler 2011, eq 6.

        To integrate perform a substitution y = x - r_p.
        """
        lnr = np.log(self.r)
        lnxi = np.log(self.corr_gal)

        p = np.zeros(len(self.r))

        # Calculate correlation to higher scales for better integration
        if self.proj_limit is None:
            rlim = max(80.0, 5 * self.rmax)
        else:
            rlim = self.proj_limit

        print "RLIM, RMAX", rlim, self.rmax
        if rlim > self.rmax:
            upper_h = deepcopy(self)
            dr = self.r[1] / self.r[0]
            upper_h.update(rmin=self.rmax * dr, rmax=rlim, rnum=20)

            fit = spline(np.concatenate((self.r, upper_h.r)),
                         np.concatenate((self.corr_gal, upper_h.corr_gal)), k=3)  # [self.corr_gal > 0] maybe?
            print "FIT: ", fit(0.1)
        else:
            fit = spline(self.r, self.corr_gal, k=3)  # [self.corr_gal > 0] maybe?
            print "fit: ", fit(0.1)
        f_peak = 0.01
        a = 0

        for i, rp in enumerate(self.r):
            if a != 1.3 and i < len(self.r) - 1:
                # Get slope at rp (== index of power law at rp)
                ydiff = (lnxi[i + 1] - lnxi[i]) / (lnr[i + 1] - lnr[i])
                # if the slope is flatter than 1.3, it will converge faster, but to make sure, we cut at 1.3
                a = max(1.3, -ydiff)
                theta = self._get_theta(a)

            min_y = theta * f_peak ** 2 * rp

            # Get the upper limit for this rp
            lim = np.log10(rlim - rp)

            # Set the y vector for this rp
            y = np.logspace(np.log10(min_y), lim, 1000)

            # Integrate
            integ_corr = fit(y + rp)
            integrand = integ_corr / np.sqrt((y + 2 * rp) * y)
            p[i] = intg.simps(integrand, y) * 2

        return p

    def _get_theta(self, a):
        theta = 2 ** (1 + 2 * a) * (7 - 2 * a ** 3 + 3 * np.sqrt(5 - 8 * a + 4 * a ** 2) + a ** 2 * (9 + np.sqrt(5 - 8 * a + 4 * a ** 2)) -
                           a * (13 + 3 * np.sqrt(5 - 8 * a + 4 * a ** 2))) * ((1 + np.sqrt(5 - 8 * a + 4 * a ** 2)) / (a - 1)) ** (-2 * a)
        theta /= (a - 1) ** 2 * (-1 + 2 * a + np.sqrt(5 - 8 * a + 4 * a ** 2))
        return theta

    def angular_corr_gal(self, f, theta_min, theta_max, theta_num, logtheta=True,
                         x_min=0, x_max=10000):
        """
        Calculate the angular correlation function w(theta).
        
        From Blake+08, Eq. 33
        
        Parameters
        ----------
        f : function
            A function of a single variable which returns the normalised
            quantity of sources at a given comoving distance x.
            
        theta_min : float
            Minimum theta value (in radians)
            
        theta_max : float
            Maximum theta value (in radians)
            
        theta_num : int
            Number of theta values
            
        logtheta : bool, True
            Whether to use logspace for theta values
        """
        # Set up theta values
        if theta_min <= 0 or theta_min >= theta_max:
            raise ValueError("theta_min must be > 0 and < theta_max")
        if theta_max >= 180.0:
            raise ValueError("theta_max must be < pi")

        if logtheta:
            theta = 10 ** np.linspace(np.log10(theta_min), np.log10(theta_max), theta_num)
        else:
            theta = np.linspace(theta_min, theta_max, theta_num)

        if x_min <= 0 or x_min >= x_max:
            raise ValueError("x_min must be >0 and < x_max")


        # Initialise result
        w = np.zeros_like(theta)

        umin = -3
        umax = 2.2

        rmax = np.sqrt((10 ** umax) ** 2 + theta_max ** 2 * x_max ** 2)
        if rmax > 1.2 * self.rmax:
            print "WARNING: likely bad extrapolation in angular c.f. rmax = %s >> %s" % (rmax, self.rmax)

        # Setup vectors u^2,x^2 and f(x)
        u = np.logspace(umin, umax, 500)
        u2 = u * u
        x = np.logspace(np.log10(x_min), np.log10(x_max), 500)
        x2 = x * x
        du = u[1] - u[0]
        dx = x[1] - x[0]
        xfx = x * f(x) ** 2  # multiply by x because of log integration

        # Set up spline for xi(r)
        xi = spline(self.r, self.corr_gal, k=3)

        for i, th in enumerate(theta):
            # Set up matrix integrand (for double-integration).
            r = np.sqrt(np.add.outer(th * th * x2, u2)).flatten()  # # needs to be 1d for spline eval
            integrand = np.einsum("ij,i,j->ij", xi(r).reshape((len(x2), len(u2))), xfx, u)  # reshape here for dblsimps, mult by u for log int
            w[i] = dblsimps(integrand, dx, du)

        return w * 2 * np.log(10) ** 2, theta

class NGException(Exception):
    pass






