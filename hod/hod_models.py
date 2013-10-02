'''
Created on Sep 30, 2013

@author: Steven
'''
import numpy as np
import scipy.special as sp

class HOD_models(object):

    def __init__(self, hod_model, M_1=10 ** 12.851, alpha=1.049, M_min=10 ** 11.6222,
                 gauss_width=0.26, M_c=10 ** 12.0, M_0=10 ** 11.5047, fca=0.5, fcb=0, fs=1, delta=None, x=1,
                 central=True):

        #Save parameters to self. Many of these are used in multiple models.
        self.M_1 = M_1
        self.M_min = M_min
        self.M_0 = M_0
        self.alpha = alpha
        self.M_min = M_min
        self.gauss_width = gauss_width
        self.fca = fca
        self.fcb = fcb
        self.fs = fs
        self.delta = delta
        self.x = x
        self.M_c = M_c
        self.central = central

        if hod_model == 'zheng':
            self.nc = self._nc_zheng
            self.ns = self._ns_zheng

        elif hod_model == 'zehavi':
            self.nc = self._nc_zehavi
            self.ns = self._ns_zehavi

        elif hod_model == 'contreras':
            self.nc = self._nc_contreras
            self.ns = self._ns_contreras

        elif hod_model == 'geach':
            self.nc = self._nc_geach
            self.ns = self._ns_geach

    def _nc_zheng(self, M):
        """
        Defines the central galaxy number for 3-param model of Zheng (2005)
        """
        n_c = np.zeros_like(M)
        n_c[M > self.M_min] = 1

        return n_c

    def _nc_zehavi(self, M):
        """
        Defines the central galaxy number for 5-param model of Zehavi (2005)
        """
        nc = 0.5 * (1 + sp.erf((np.log10(M) - np.log10(self.M_min)) / self.gauss_width))
        return nc
    def _nc_contreras(self, M):
        """
        Defines the central galaxy number for 9-param model of Contreras (2012)
        """
        return self.fcb * (1 - self.fca) * np.exp(np.log10(M / self.M_c) ** 2 / (2 * (self.x * self.gauss_width) ** 2)) + self.fca * (1 + sp.erf(np.log10(M / self.M_c) / self.x / self.gauss_width))

    def _ns_zheng(self, M):
        """
        Defines the satellite galaxy number for 3-param model of Zheng(2005)
        """
        return (M / self.M_1) ** self.alpha

    def _ns_zehavi(self, M):
        """
        Defines the satellite galaxy number for 5-param model of Zehavi(2005)
        """
        ns = np.zeros_like(M)
        ns[M > self.M_0] = ((M[M > self.M_0] - self.M_0) / self.M_1) ** self.alpha

        return ns

    def _ns_contreras(self, M):
        """
        Defines the satellite galaxy number for 9-param model of Contreras(2012)
        """
        return self.fs * (1 + sp.erf(np.log10(M / self.M_1) / self.delta)) * (M / self.M_1) ** self.alpha


    def ntot(self, M):
        if self.central:
            return self.nc(M) * (1.0 + self.ns(M))
        else:
            return self.nc(M) + self.ns(M)
