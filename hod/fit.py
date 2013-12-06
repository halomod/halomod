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
import sys
from scipy.optimize import minimize
import copy

import scipy.sparse as sp
import scipy.sparse.linalg as spln
#===============================================================================
# The Model
#===============================================================================

def model(parm, priors, h, attrs, data, sd):
    """
    Calculate the log probability of a HMF model given data
    
    Parameters
    ----------
    parm : sequence
        The position of the model. Takes arbitrary parameters.
        
    initial : dict
        A dictionary with keys as names of parameters in ``parm`` and values
        as a 4-item list. The first item is the guessed value for the parameter,
        the second is a string identifying whether the prior is normal (``norm``)
        or uniform (``unif``). The third and fourth are the uniform lower and
        upper boundaries or the mean and standard deviation of the normal
        distribution.
        
    h : ``hod.HOD`` instance
        A fully realised instance is best as it may make updating it faster
        
    attrs : list
        A list of the names of parameters passed in ``parm``. Corresponds to the
        keys of ``initial``.
        
    data : array_like
        The measured correlation function.
        
    sd : array_like
        Uncertainty in the measured correlation function
        
    Returns
    -------
    ll : float
        The log likelihood of the model at the given position.
        
    """
    # print "parm before: ", parm
    ll = 0

    for prior in priors:
        # A uniform prior doesn't change likelihood but returns -inf if outside bounds
        if isinstance(prior, Uniform):
            index = attrs.index(prior.name)
            if parm[index] < prior.low or parm[index] > prior.high:
                return -np.inf

        elif isinstance(prior, Normal):
            index = attrs.index(prior.name)
            ll += norm.logpdf(parm[index], loc=prior.mean, scale=prior.sd)

        elif isinstance(prior, MultiNorm):
            indices = [attrs.index(name) for name in prior.name]
            ll += _lognormpdf(np.array(parm[indices]), np.array(prior.means),
                              prior.cov)

    # print "parm after:", parm
    # Rebuild the hod dict from given vals
    hoddict = {attr:val for attr, val in zip(attrs, parm)}

    # print h.__dict__

    h.update(**hoddict)
    # print "updated"
    # print h.__dict__
    # The logprob of the model
    model = h.corr_gal.copy()  # data + np.random.normal(scale=0.1)
    # print "saved model"
    ll += np.sum(norm.logpdf(data, loc=model, scale=sd))

    # print "got here"
    return ll

def fit_hod(r, data, sd, priors, guess=[], nwalkers=100, nsamples=100, burnin=10,
           thin=50, nthreads=8, filename=None, **hodkwargs):
    """
    Run an MCMC procedure to fit a model correlation function to data
    
    Parameters
    ----------
    r : array_like
        The scales at which to perform analysis. Must be the same as the input
        data
        
    data : array_like
        The measured correlation function at ``r``
        
    sd : array_like
        The uncertainty in the measured correlation function
        
    priors : list of ``Uniform``, ``Gaussian`` or ``MultiGaussian`` objects
        A dictionary with keys as names of parameters in ``parm`` and values
        as a 4-item list. The first item is the guessed value for the parameter,
        the second is a string identifying whether the prior is normal (``norm``)
        or uniform (``unif``). The third and fourth are the uniform lower and
        upper boundaries or the mean and standard deviation of the normal
        distribution.
        
    guess : array_like, default ``None``
        Where to start the chain. If ``None``, will get central values from the
        distributions.
        
    nwalkers : int, default 100
        Number of walkers to use for Affine-Invariant Ensemble Sampler
        
    nsamples : int, default 100
        Number of samples that *each walker* will perform.
        
    burnin : int, default 10
        Number of samples from each walker that will be initially erased as burnin
        
    thin : int, default 50
        Keep 1 in every ``thin`` samples.
        
    nthreads : int, default 1
        Number of threads to use in sampling.
        
    \*\*hmfkwargs : arguments
        Any argument that could be sent to ``hmf.Perturbations``
        
    Returns
    -------
    flatchain : array_like, ``shape = (len(initial),nwalkers*nsamples)``
        The MCMC chain, with each parameter as a column
        
    acceptance_fraction : float
        The acceptance fraction for the MCMC run. Should be ....
    
        
    """

    if len(priors) == 0:
        raise ValueError("initial must be at least length 1")

    # ##NOW TRY A LAPLACE APPROXIMATION TO FIND THE MODE
    # res = minimize()


    # Save which attributes are updatable for HOD as a list
    attrs = []
    for prior in priors:
        if isinstance(prior.name, basestring):
            attrs += [prior.name]
        else:
            attrs += prior.name

     # Get the number of variables for MCMC
    ndim = len(attrs)

    # Check that attrs are all applicable
    for a in attrs:
        if a not in ["wdm_mass", "delta_halo", "sigma_8", "n", "omegab", 'omegac',
                     "omegav", "omegak", "H0", "M_1", 'alpha', "M_min", 'gauss_width',
                     'M_0', 'fca', 'fcb', 'fs', 'delta', 'x', 'omegab_h2',
                     'omegac_h2', 'h']:
            raise ValueError(a + " is not a valid variable for MCMC in HOD")

    # Make sure hodkwargs is ok
    if nthreads > 1:
        numthreads = 1
    else:
        numthreads = 0
    if 'NumThreads' in hodkwargs:
        del hodkwargs['NumThreads']

    # Initialise the HOD object - use all available cpus for this
    h = HOD(r=r, ThreadNum=0, **hodkwargs)

    # It's better to get a corr_gal instance now and then the updates are faster
    h.corr_gal

    # Now update numthreads for MCMC parallelisation
    h.update(ThreadNum=numthreads)

    # Set guess if not set
    if len(guess) != len(attrs):
        guess = []
        for prior in priors:
            if isinstance(prior, Uniform):
                print "uniform"
                guess += [(prior.high + prior.low) / 2]
            elif isinstance(prior, Normal):
                print "norm"
                guess += [prior.mean]
            elif isinstance(prior, MultiNorm):
                print "multinorm"
                print prior.means
                guess += prior.means.tolist()

    # Get an array of initial values
    guess = np.array(guess)

    # Get an initial value for all walkers, around a small ball near the initial guess
    stacked_val = guess.copy()
    for i in range(nwalkers - 1):
        stacked_val = np.vstack((guess, stacked_val))
    p0 = stacked_val * np.random.normal(loc=1.0, scale=0.2, size=ndim * nwalkers).reshape((nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, model,
                                    args=[priors, h, attrs, data, sd],
                                    threads=nthreads)

    # Run a burn-in
    if burnin:
        pos, prob, state = sampler.run_mcmc(p0, burnin)
        sampler.reset()
    else:
        pos = p0
    # Run the actual run
    sampler.run_mcmc(pos, nsamples, thin=thin)

    return sampler.flatchain, np.mean(sampler.acceptance_fraction)

class Uniform(object):
    def __init__(self, param, low, high):
        self.name = param
        self.low = low
        self.high = high

class Normal(object):
    def __init__(self, param, mean, sd):
        self.name = param
        self.mean = mean
        self.sd = sd

class MultiNorm(object):
    def __init__(self, params, means, cov):
        self.name = params
        self.means = means
        self.cov = cov

def _lognormpdf(x, mu, S):
    """ Calculate gaussian probability density of x, when x ~ N(mu,sigma) """
    nx = len(S)
    tmp = -0.5 * (nx * np.log(2 * np.pi) + np.linalg.slogdet(S)[1])

    err = x - mu
    if (sp.issparse(S)):
        numerator = spln.spsolve(S, err).T.dot(err)
    else:
        numerator = np.linalg.solve(S, err).T.dot(err)

    return tmp - numerator
