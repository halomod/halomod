'''
Created on Oct 14, 2013

@author: Steven

This module contains numerous functions for fitting HOD models to correlation
function data. It uses MCMC techniques to do so.
'''
#===============================================================================
# IMPORTS
#===============================================================================
import numpy as np
import emcee
from hod import HOD
from scipy.stats import norm

class HODFit(object):
    """
    A class that gets a fit for a HOD model. Accepts same keywords as HOD, but
    also data,var,r which are the r,xi and var(xi) for the observed data
    
    The idea here is that we are given some data points for xi(r) -- both r 
    and xi. Attached to each point is a standard deviation sd from the data (we
    treat each point as independent for now at least). Then each point may be
    represented as the model, HOD_xi(r) plus a contribution from a normal dist.
    
    INPUT
    r    -- array-like float, the scales at which the data is given
    data -- array-like float, the values of the galaxy correlation function from data
    sd   -- array-like float, the uncertainty on each point of the data
    initial - dict, a series of HOD() class parameters that are to be fitted,
            with their initial guesses supplied, and a prior
            
            Example: initial = {"alpha":[1.049,"norm",1.049,0.01], #normal prior with mean,sd = 1.049,0.01
                                "omegab":[0.05,"unif",0,0.2]}      #Uniform prior with limits 0,0.2
            
    **hodkwargs - any other parameters that are sent to HOD (these won't be fit)
    """

    def __init__(self, r, data, sd, initial={}, **hodkwargs):
        #SAVE THE OBSERVED DATA TO THE CLASS
        self.r = r
        self.data = data
        self.sd = sd
        self.initial = initial

        if len(initial) == 0:
            raise ValueError("initial must be at least length 1")

        #Get the number of variables for MCMC
        self.ndim = len(initial)
        #Save which attributes are updatable for HOD as a list
        self.attrs = [k for k in initial]
        #Update the kwargs for HOD with initial (which is just a dict with
        #HOD parameters that are subject to change in MCMC).
        hodkwargs.update(initial)

        #Initialise the HOD object
        self.h = HOD(r=self.r, **hodkwargs)

    def _logprob(self, vals):
        """
        Creates a HOD object and gets the logp value compared to data
        """
        ll = 0

        #First check all values are inside boundaries (if bounds given), and
        #Get the logprob of the priors-- uniform priors don't count here
        i = 0
        for k, v in self.initial.iteritems():
            if v[1] == "norm":

                ll += norm.logpdf(vals[i], loc=v[2], scale=v[3])
            elif v[1] == "unif":
                if vals[i] < v[2]:
                    vals[i] = v[2]
                elif vals[i] > v[3]:
                    vals[i] = v[3]
            i += 1

        #First rebuild the hod dict from given vals
        hoddict = {attr:val for attr, val in zip(self.attrs, vals)}

        self.h.update(**hoddict)

        #The logprob of the model
        model = self.h.corr_gal
        ll += np.sum(norm.logpdf(self.data, loc=model, scale=self.sd))

        return ll

    def run_mc(self, nwalkers=100, nsamples=100, burnin=10, thin=50, filename=None):
        """
        Does the heavy lifting of the MCMC algorithm using emcee
        """
        #Get an array of initial values
        initial = np.array([val[0] for k, val in self.initial.iteritems()])

        #Get an initial value for all walkers, around a small ball near the initial guess
        p0 = (np.repeat(initial, nwalkers) * np.random.normal(loc=1.0, scale=0.2, size=self.ndim * nwalkers)).reshape((nwalkers, self.ndim))


        sampler = emcee.EnsembleSampler(nwalkers, self.ndim, self._logprob)

        #Run a burn-in
        pos, prob, state = sampler.run_mcmc(p0, burnin)
        sampler.reset()

        #Run the actual run
        sampler.run_mcmc(pos, nsamples, thin=thin)

        return sampler.flatchain, np.mean(sampler.acceptance_fraction)


