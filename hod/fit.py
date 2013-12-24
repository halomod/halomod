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
from multiprocessing import Pool, cpu_count
import scipy.sparse as sp
import scipy.sparse.linalg as spln
import time
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
        
    priors : list of prior classes
        A list containing objects that are either of the Uniform(), Normal() or 
        MultiNorm() class. These specify the prior information on each parameter.
        
    h : ``hod.HOD`` instance
        A fully realised instance is best as it may make updating it faster
        
    attrs : list
        A list of the names of parameters passed in ``parm``. Corresponds to the
        names of the objects in ``priors``.
        
    data : array_like
        The measured correlation function.
        
    sd : array_like
        Uncertainty in the measured correlation function
        
    Returns
    -------
    ll : float
        The log likelihood of the model at the given position.
        
    """
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

    # Rebuild the hod dict from given vals
    hoddict = {attr:val for attr, val in zip(attrs, parm)}
    h.update(**hoddict)

    # The logprob of the model
    model = h.corr_gal.copy()

    ll += np.sum(norm.logpdf(data, loc=model, scale=sd))

    return ll

def fit_hod(r, data, sd, priors, guess=[], nwalkers=100, nsamples=100, burnin=10,
           thin=50, nthreads=8, filename=None, chunks=None, **hodkwargs):
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
        
    filename : str, default ``None``
        A path to a file to which to write results sequentially
        
    chunks : int, default ``None``
        How many samples to run before appending results to file. Only
        applicable if ``filename`` is provided.
        
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
        raise ValueError("priors must be at least length 1")

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

    hodkwargs.update({"ThreadNum":nthreads})

    # Initialise the HOD object
    h = HOD(r=r, **hodkwargs)

    # It's better to get a corr_gal instance then the updates could be faster
    # BUT because of some reason we need to hack this and do it in a map() function
    h = Pool(1).apply(create_hod, [h])

    # re-set the number of threads used in pycamb to 1
    hodkwargs.update({"ThreadNum":1})

    # auto-calculate the number of threads to use if not set.
    if not nthreads:
        nthreads = cpu_count()

    # Set guess if not set
    if len(guess) != len(attrs):
        guess = []
        for prior in priors:
            if isinstance(prior, Uniform):
                guess += [(prior.high + prior.low) / 2]
            elif isinstance(prior, Normal):
                guess += [prior.mean]
            elif isinstance(prior, MultiNorm):
                guess += prior.means.tolist()

    guess = np.array(guess)

    # Get an initial value for all walkers, around a small ball near the initial guess
    stacked_val = guess.copy()
    for i in range(nwalkers - 1):
        stacked_val = np.vstack((guess, stacked_val))

    i = 0
    for prior in priors:
        if isinstance(prior, Uniform):
            stacked_val[:, i] += np.random.normal(loc=0.0, scale=0.05 * (prior.high - prior.low), size=nwalkers)
            i += 1
        elif isinstance(prior, Normal):
            stacked_val[:, i] += np.random.normal(loc=0.0, scale=prior.sd, size=nwalkers)
            i += 1
        elif isinstance(prior, MultiNorm):
            for j, name in enumerate(prior.name):
                stacked_val[:, i] += np.random.normal(loc=0.0, scale=np.sqrt(prior.cov[j, j]), size=nwalkers)
                i += 1

    sampler = emcee.EnsembleSampler(nwalkers, ndim, model,
                                    args=[priors, h, attrs, data, sd],
                                    threads=nthreads)

    # Run a burn-in
    if burnin:
        pos, prob, state = sampler.run_mcmc(stacked_val, burnin)
        sampler.reset()
    else:
        pos = stacked_val


    # Run the actual run
    if filename is None:
        sampler.run_mcmc(pos, nsamples, thin=thin)
    else:
        header = "# " + "\t".join(attrs) + "\n"
        with open(filename, "w") as f:
            f.write(header)

        if chunks is None:
            chunks = nsamples

        start = time.time()
        for i, result in enumerate(sampler.sample(pos, iterations=nsamples)):
            if (i + 1) % chunks == 0:
                print "Done ", 100 * float(i + 1) / nsamples , "%. Time per sample: ", (time.time() - start) / ((i + 1) * nwalkers)
                with open(filename, "w") as f:
                    f.write(header)
                    np.savetxt(f, sampler.flatchain[sampler.flatchain[:, 0] != 0.0, :])
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

def create_hod(h):
    h.corr_gal
    return h

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
