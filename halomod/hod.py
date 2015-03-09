
import numpy as np
import scipy.special as sp
from hmf._framework import Model
_allmodels = ["Zehavi05", "Zheng05", "Contreras"]

class HOD(Model):
    """
    Halo Occupation Distribution model base class.
    
    This class defines three methods -- the average central galaxies, average
    satellite galaxies and total galaxies. 
    
    The total number of galaxies can take two forms: one if there MUST be a
    central galaxy to have a satellite, and the other if not. 
    
    This class should not be called directly. The user
    should call a derived class.
    
    Derived classes of :class:`HOD` should define two methods: :method:`nc` and 
    :method:`ns` (central and satellite distributions respectively).
    Additionally, any parameters of the model should have their names and
    defaults defined as class variables. 
    
    The exception to this is the M_min parameter, which is defined for every
    model (it may still be defined to modify the default). This parameter acts
    as the one that may be set via the mean number density given all the other
    parameters. If the model has a sharp cutoff at low mass, corresponding to
    M_min, the extra parameter sharp_cut may be set to True, allowing for simpler
    setting of M_min via this route. 
    
    See the derived classes in this module for examples of how to define derived
    classes of :class:`HOD`.
    """
    _defaults = {"M_min": 11}
    sharp_cut = False

    def __init__(self, central=True, **model_parameters):

        self._central = central
        super(HOD, self).__init__(**model_parameters)
    def nc(self, M):
        pass

    def ns(self, M):
        pass

    def ntot(self, M):
        if self._central:
            return self.nc(M) * (1.0 + self.ns(M))
        else:
            return self.nc(M) + self.ns(M)

    @property
    def mmin(self):
        return self.params["M_min"]

class Zehavi05(HOD):
    """
    Three-parameter model of Zehavi (2005)
    
    Parameters
    ----------
    M_min : float, default = 11.6222
        Minimum mass of halo that supports a central galaxy
        
    M_1 : float, default = 12.851
        Mass of a halo which on average contains 1 satellite
        
    alpha : float, default = 1.049
        Index of power law for satellite galaxies
    """
    _defaults = {"M_min":11.6222,
                 "M_1":12.851,
                 "alpha":1.049}
    sharp_cut = True

    def nc(self, M):
        """
        Number of central galaxies at mass M
        """
        n_c = np.zeros_like(M.value)
        n_c[M.value >= 10 ** self.params["M_min"]] = 1

        return n_c

    def ns(self, M):
        """
        Number of satellite galaxies at mass M
        """
        return (M.value / 10 ** self.params["M_1"]) ** self.params["alpha"]

class Zheng05(HOD):
    """
    Five-parameter model of Zehavi (2005)
    
    Parameters
    ----------
    M_min : float, default = 11.6222
        Minimum mass of halo that supports a central galaxy
        
    M_1 : float, default = 12.851
        Mass of a halo which on average contains 1 satellite
        
    alpha : float, default = 1.049
        Index of power law for satellite galaxies
        
    sig_logm : float, default = 0.26
        Width of smoothed cutoff
        
    M_0 : float, default = 11.5047
        Minimum mass of halo containing satellites
    """
    _defaults = {"M_min":11.6222,
                 "M_1":12.851,
                 "alpha":1.049,
                 "M_0":11.5047,
                 "sig_logm":0.26
                 }

    def nc(self, M):
        """
        Number of central galaxies at mass M
        """
        nc = 0.5 * (1 + sp.erf((np.log10(M.value) - self.params["M_min"]) / self.params["sig_logm"]))
        return nc

    def ns(self, M):
        """
        Number of satellite galaxies at mass M
        """
        ns = np.zeros_like(M.value)
        ns[M > 10 ** self.params["M_0"]] = ((M.value[M.value > 10 ** self.params["M_0"]] - 10 ** self.params["M_0"]) / 10 ** self.params["M_1"]) ** self.params["alpha"]
        return ns

    @property
    def mmin(self):
        return self.params["M_min"] - 5 * self.params["sig_logm"]

class Contreras(HOD):
    """
    Nine-parameter model of Contreras (2009)
    
    Parameters
    ----------
    M_min : float, default = 11.6222
        Minimum mass of halo that supports a central galaxy
        
    M_1 : float, default = 12.851
        Mass of a halo which on average contains 1 satellite
        
    alpha : float, default = 1.049
        Index of power law for satellite galaxies
        
    sig_logm : float, default = 0.26
        Width of smoothed cutoff
        
    M_0 : float, default = 11.5047
        Minimum mass of halo containing satellites
        
    fca : float, default = 0.5
        fca
        
    fcb : float, default = 0
        fcb
        
    fs : float, default = 1
        fs
        
    delta : float, default  = 1
        delta
        
    x : float, default = 1
        x
    """
    _defaults = {"M_min":11.6222,
                 "M_1":12.851,
                 "alpha":1.049,
                 "M_0":11.5047,
                 "sig_logm":0.26,
                 "fca":0.5,
                 "fcb":0,
                 "fs":1,
                 "delta":1,
                 "x":1
                 }

    def nc(self, M):
        """
        Number of central galaxies at mass M
        """
        return self.params["fcb"] * (1 - self.params["fca"]) * np.exp(np.log10(M.value / 10 ** self.params["M_min"]) ** 2 / (2 * (self.params["x"] * self.params["sig_logm"]) ** 2)) + self.params["fca"] * (1 + sp.erf(np.log10(M.value / 10 ** self.params["M_min"]) / self.params["x"] / self.params["sig_logm"]))

    def ns(self, M):
        """
        Number of satellite galaxies at mass M
        """
        return self.params["fs"] * (1 + sp.erf(np.log10(M.value / 10 ** self.params["M_1"]) / self.params["delta"])) * (M.value / 10 ** self.params["M_1"]) ** self.params["alpha"]

class HI(HOD):
    _defaults = {"M_min":11.6222,
                 "M_1":12.851,
                 "alpha":1.049,
                 "M_0":11.5047,
                 "sig_logm":0.26,
                 }

    def nc(self, M):
        return np.exp(-0.5 * (np.log10(M.value) - self.params["M_min"]) ** 2 / self.params["sig_logm"] ** 2)

    def ns(self, M):
        """
        Number of satellite galaxies at mass M
        """
        return (M.value / 10 ** self.params["M_1"]) ** self.params["alpha"]

    @property
    def mmin(self):
        return self.params["M_min"] - 5 * self.params["sig_logm"]
