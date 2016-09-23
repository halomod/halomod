"""
Module for defining HOD classes.

The HOD class exposes methods that deal directly with occupation statistics and don't interact with the broader halo
model. These include things like the average satellite/central occupation, total occupation, and pair counts.

The most subtle/important thing to note about these classes are the assumptions surrounding the satellite/central
decomposition. So here are the assumptions:

1. The average satellite occupancy is taken to be the average over *all* haloes, with and without centrals. This has
   subtle implications for how to mock up the galaxy population, because if one requires a central before placing a
   satellite, then the avg. number of satellites placed into *available* haloes is increased if the central occupation
   is less than 1.

2. We provide the option to enforce a "central condition", that is, the requirement that a central be found in a halo
   before any satellites are observed. To enforce this, set ``central=True`` in the constructor of any HOD. This has
   some ramifications:

3. If the central condition is enforced, then for all HOD classes (except see point 5), the mean satellite occupancy is
   modified. If the defined occupancy is Ns', then the returned occupancy is Ns = Nc*Ns'. This merely ensures that Ns=0
   when Nc=0. The user should note that this will change the interpretation of parameters in the Ns model, unless Nc is
   a simple step function.

4. The pair-wise counts involve a term <Nc*Ns>. When the central condition is enforced, this reduces trivially to <Ns>.
   However, if the central condition is not enforced we *assume* that the variates Nc and Ns are uncorrelated, and
   use <Nc*Ns> = <Nc><Ns>.

5. A HOD class that is defined with the central condition intrinsically satisfied, the class variable
   ``central_condition_inherent`` can be set to True in the class definition, which will avoid the extra modification.
   Do note that just because the class is specified such that the central condition can be satisfied (i.e. <Ns> is 0
   when <Nc> is zero), and thus the ``central_condition_inherent`` is True, does not mean that it is entirely enforced.
   The pairwise counts still depend on whether the user assumes that the central condition is enforced or not, which must
   be set at instantiation.

6. By default, the central condition is *not* enforced.
"""


import numpy as np
import scipy.special as sp
from hmf._framework import Component
#_allmodels = ["Zehavi05", "Zheng05", "Contreras"]

class HOD(Component):
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
    Additionally, as with all :class:`hmf._framework.Model` classes,
    each class should specify its parameters in a _defaults dictionary at
    class-level.

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
    central_condition_inherent = False

    def __init__(self, central=False, **model_parameters):

        self._central = central
        super(HOD, self).__init__(**model_parameters)

    def _nc(self, m):
        pass

    def nc(self,m):
        return self._nc(m)

    def _ns(self, m):
        pass

    def ns(self,m):
        if self._central and not self.central_condition_inherent:
            return self.nc(m)*self._ns(m)
        else:
            return self._ns(m)

    def ntot(self, m):
        return self.nc(m)+self.ns(m)

    def ss_pairs(self,m):
        return self.ns(m)**2

    def cs_pairs(self,m):
        if self._central:
            return self.ns(m)
        else:
            return self.nc(m)*self.ns(m)

    def tot_pairs(self,m):
        return self.ss_pairs(m) + self.cs_pairs(m)

    @property
    def mmin(self):
        return None

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
    central_condition_inherent = True

    def _nc(self, M):
        """
        Number of central galaxies at mass M
        """
        n_c = np.zeros_like(M)
        n_c[M >= 10 ** self.params["M_min"]] = 1

        return n_c

    def _ns(self, M):
        """
        Number of satellite galaxies at mass M
        """
        return (M / 10 ** self.params["M_1"]) ** self.params["alpha"]

    @property
    def mmin(self):
        return self.params["M_min"]

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

    def _nc(self, M):
        """
        Number of central galaxies at mass M
        """
        nc = 0.5 * (1 + sp.erf((np.log10(M) - self.params["M_min"]) / self.params["sig_logm"]))
        return nc

    def _ns(self, M):
        """
        Number of satellite galaxies at mass M
        """
        ns = np.zeros_like(M)
        ns[M > 10 ** self.params["M_0"]] = ((M[M > 10 ** self.params["M_0"]] - 10 ** self.params["M_0"]) / 10 ** self.params["M_1"]) ** self.params["alpha"]
        return ns

    @property
    def mmin(self):
        return self.params["M_min"] - 5 * self.params["sig_logm"]

class Contreras13(HOD):
    """
    Nine-parameter model of Contreras (2013)

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

    def _nc(self, M):
        """
        Number of central galaxies at mass M
        """
        return self.params["fcb"] * (1 - self.params["fca"]) * np.exp(-np.log10(M / 10 ** self.params["M_min"]) ** 2 / (2 * (self.params["x"] * self.params["sig_logm"]) ** 2)) + self.params["fca"] * (1 + sp.erf(np.log10(M / 10 ** self.params["M_min"]) / self.params["x"] / self.params["sig_logm"]))

    def _ns(self, M):
        """
        Number of satellite galaxies at mass M
        """
        return self.params["fs"] * (1 + sp.erf(np.log10(M / 10 ** self.params["M_1"]) / self.params["delta"])) * (M / 10 ** self.params["M_1"]) ** self.params["alpha"]

class Geach12(Contreras13):
    """
    8-parameter model of Geach et. al. (2012). This is identical to `Contreras13`,
    but with `x==1`.
    """
    pass

class Tinker05(Zehavi05):
    """
    3-parameter model of Tinker et. al. (2005).
    """
    _defaults = {"M_min":11.6222,
                 "M_1":12.851,
                 "M_cut":12.0}

    def _ns(self,M):
        out = self.nc(M)
        return out*np.exp(-10**self.params["M_cut"]/(M-10**self.params["M_min"]))*(M/10**self.params["M_1"])


class HI(HOD):
    _defaults = {"M_min":11.6222,
                 "M_1":12.851,
                 "alpha":1.049,
                 "M_0":11.5047,
                 "sig_logm":0.26,
                 }

    def _nc(self, M):
        return np.exp(-0.5 * (np.log10(M) - self.params["M_min"]) ** 2 / self.params["sig_logm"] ** 2)

    def _ns(self, M):
        """
        Number of satellite galaxies at mass M
        """
        return (M / 10 ** self.params["M_1"]) ** self.params["alpha"]

    @property
    def mmin(self):
        return self.params["M_min"] - 5 * self.params["sig_logm"]
