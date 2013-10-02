'''
Created on Sep 9, 2013

@author: Steven
'''
import numpy as np
import scipy.integrate as intg

def power_to_corr(lnP, lnk, R):
    """
    Calculates the correlation function given a power spectrum
    
    NOTE: no check is done to make sure k spans [0,Infinity] - make sure of this before you enter the arguments.
    
    INPUT
        lnP: vector of values for the log power spectrum
        lnk: vector of values (same length as lnP) giving the log wavenumbers for the power (EQUALLY SPACED)
        r:   radi(us)(i) at which to calculate the correlation
    """

    k = np.exp(lnk)
    P = np.exp(lnP)

    if not np.iterable(R):
        R = [R]

    corr = np.zeros_like(R)

    for i, r in enumerate(R):
        integ = P * k ** 2 * np.sin(k * r) / r


        corr_cum = (0.5 / np.pi ** 2) * intg.cumtrapz(integ, dx=lnk[1] - lnk[0])

        #Try this: we cut off the integral when we can no longer fit 5 steps between zeros.
        max_k = np.pi / (5 * r * (np.exp(lnk[1] - lnk[0]) - 1))
        max_index = np.where(k < max_k)[-1][-1]
        #Take average of last 20 values before the max_index
        corr[i] = np.mean(corr_cum[max_index - 20:max_index])


    return corr

