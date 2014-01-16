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
from halo_model import HaloModel
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

def model(parm, priors, h, attrs, data, sd, verbose=0):
    """
    Calculate the log probability of a HaloModel model given correlation data
    
    Parameters
    ----------
    parm : list of floats
        The position of the model. Takes arbitrary parameters.
        
    priors : list of prior classes
        A list containing instances of :class:`.Uniform`, :class:`.Normal` or 
        :class:`.MultiNorm` classes. These specify the prior information on each 
        parameter.
        
    h : :class:`~halo_model.HaloModel` instance
        An instance of :class:`~halo_model.HaloModel` with the desired options
        set. Variables of the estimation are updated within the routine.  
        
    attrs : list
        A list of the names of parameters passed in :attr:`.parm`.
        
    data : array_like
        The measured correlation function.
        
    sd : array_like
        Uncertainty in the measured correlation function
        
    verbose : int, default 0
        How much to write to screen.
        
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
                ll = -np.inf

        elif isinstance(prior, Normal):
            index = attrs.index(prior.name)
            ll += norm.logpdf(parm[index], loc=prior.mean, scale=prior.sd)

        elif isinstance(prior, MultiNorm):
            indices = [attrs.index(name) for name in prior.name]
            ll += _lognormpdf(np.array(parm[indices]), np.array(prior.means),
                              prior.cov)

    # Rebuild the hod dict from given vals
    if not np.isinf(ll):
        hoddict = {attr:val for attr, val in zip(attrs, parm)}
        h.update(**hoddict)
        # The logprob of the model
        ll += np.sum(norm.logpdf(data, loc=h.corr_gal, scale=sd))

    if verbose > 0:
        print parm
        print "Likelihood: ", ll
    return ll


def fit_hod(r, data, sd, priors, h, guess=[], nwalkers=100, nsamples=100, burnin=0,
            nthreads=0, filename=None, chunks=None, verbose=0, find_peak_first=False,
            **kwargs):
    """
    Estimate the parameters in :attr:`.priors` using AIES MCMC.
    
    This routine uses the emcee package to run an MCMC procedure, fitting 
    parameters passed in :attr:`.priors` to the given galaxy correlation 
    function.
    
    Parameters
    ----------
    r : array_like
        The scales at which to perform analysis, corresponding to input data.
        
    data : array_like
        The measured correlation function at :attr:`r`
        
    sd : array_like
        The uncertainty in the measured correlation function, same length as 
        :attr:`r`.
        
    priors : list of prior classes
        A list containing instances of :class:`.Uniform`, :class:`.Normal` or 
        :class:`.MultiNorm` classes. These specify the prior information on each 
        parameter.
    
    h : instance of :class:`~halo_model.HaloModel`
        This instance will be updated with the variables of the minimization.
        Other desired options should have been set upon instantiation.
        
    guess : array_like, default []
        Where to start the chain. If empty, will get central values from the
        distributions.
        
    nwalkers : int, default 100
        Number of walkers to use for Affine-Invariant Ensemble Sampler
        
    nsamples : int, default 100
        Number of samples that *each walker* will perform.
        
    burnin : int, default 10
        Number of samples from each walker that will be initially erased as 
        burnin. Note, this performs *additional* iterations, rather than 
        consuming iterations from :attr:`.nsamples`.
                
    nthreads : int, default 0
        Number of threads to use in sampling. If nought, will automatically 
        detect number of cores available.
        
    filename : str, default ``None``
        A path to a file to which to write results sequentially. If ``None``,
        will not write anything out.
        
    chunks : int, default ``None``
        Number of samples to run before appending results to file. Only
        applicable if :attr:`.filename` is provided.
        
    verbose : int, default 0
        The verbosity level.
        
    find_peak_first : bool, default False
        Whether to perform a minimization routine before using MCMC to find a 
        good guess to begin with. Could reduce necessary burn-in.
        
    \*\*kwargs :
        Arguments passed to :func:`fit_hod_minimize` if :attr:`find_peak_first`
        is ``True``.
        
    Returns
    -------
    flatchain : array_like, ``shape = (len(initial),nwalkers*nsamples)``
        The MCMC chain, with each parameter as a column. Note that each walker 
        is semi-independent and they are interleaved. Calling 
        ``flatchain.reshape((nwalkers, -1, ndim))`` will retrieve the proper
        structure.
        
    acceptance_fraction : float
        The acceptance fraction for the MCMC run. Should be between 0.2 and 0.5.
    
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
        if a not in _vars:
            raise ValueError(a + " is not a valid variable for MCMC in HaloModel")

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

    if find_peak_first:
        res = fit_hod_minimize(r, data, sd, priors, h, guess=guess,
                               verbose=verbose, **kwargs)
        guess = res.x

    # Get an initial value for all walkers, around a small ball near the initial guess
    stacked_val = guess.copy()
    for i in range(nwalkers - 1):
        stacked_val = np.vstack((guess, stacked_val))

    i = 0
    for prior in priors:
        if isinstance(prior, Uniform):
            stacked_val[:, i] += np.random.normal(loc=0.0, scale=0.05 *
                                                  min((guess[i] - prior.low),
                                                      (prior.high - guess[i])),
                                                  size=nwalkers)
            i += 1
        elif isinstance(prior, Normal):
            stacked_val[:, i] += np.random.normal(loc=0.0, scale=prior.sd,
                                                  size=nwalkers)
            i += 1
        elif isinstance(prior, MultiNorm):
            for j in range(len(prior.name)):
                stacked_val[:, i] += np.random.normal(loc=0.0, scale=np.sqrt(prior.cov[j, j]),
                                                      size=nwalkers)
                i += 1

    # Set the pycamb ThreadNum to 1 so it doesn't interfere with the map function.
    h.update(ThreadNum=1)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, model,
                                    args=[priors, h, attrs, data, sd, verbose],
                                    threads=nthreads)

    if verbose:
        print attrs

    # Run a burn-in
    if burnin:
        pos = sampler.run_mcmc(stacked_val, burnin)[0]
        sampler.reset()
    else:
        pos = stacked_val

    # Run the actual run
    if filename is None:
        sampler.run_mcmc(pos, nsamples)
    else:
        header = "# " + "\t".join(attrs) + "\n"
        with open(filename, "w") as f:
            f.write(header)

        if chunks is None:
            chunks = nsamples

        start = time.time()
        for i, result in enumerate(sampler.sample(pos, iterations=nsamples)):
            if (i + 1) % chunks == 0:
                if verbose:
                    print "Done ", 100 * float(i + 1) / nsamples ,
                    print "%. Time per sample: ", (time.time() - start) / ((i + 1) * nwalkers)
                with open(filename, "w") as f:
                    # need to write out nwalkers to be able to read the file back in properly
                    f.write("# %s\n" % (nwalkers))
                    f.write(header)
                    np.savetxt(f, sampler.flatchain[sampler.flatchain[:, 0] != 0.0, :])

    return sampler.flatchain, np.mean(sampler.acceptance_fraction)

def fit_hod_minimize(r, data, sd, priors, h, guess=[], verbose=0,
                     method="Nelder-Mead", disp=False, maxiter=30, tol=None):
    """
    Run an optimization procedure to fit a model correlation function to data.
    
    Parameters
    ----------
    r : array_like
        The scales at which to perform analysis, corresponding to input data.
        
    data : array_like
        The measured correlation function at :attr:`r`
        
    sd : array_like
        The uncertainty in the measured correlation function, same length as 
        :attr:`r`.
        
    h : instance of :class:`~halo_model.HaloModel`
        This instance will be updated with the variables of the minimization.
        Other desired options should have been set upon instantiation.
        
    priors : list of prior classes
        A list containing instances of :class:`.Uniform`, :class:`.Normal` or 
        :class:`.MultiNorm` classes. These specify the prior information on each 
        parameter.
        
    guess : array_like, default []
        Where to start the chain. If empty, will get central values from the
        distributions.
        
    verbose : int, default 0
        The verbosity level. 
        
    method : str, default ``"Nelder-Mead"``
        The optimizing routine (see `scipy.optimize.minimize` for details).
        
    disp : bool, default False
        Whether to display optimization information while running.
        
    maxiter : int, default 30
        Maximum number of iterations
        
    tol : float, default None
        Tolerance for termination
        
    Returns
    -------
    res : instance of :class:`scipy.optimize.Result`
        Contains the results of the minimization. Important attributes are the
        solution vector :attr:`x`, the number of iterations :attr:`nit`, whether
        the minimization was a success :attr:`success`, and the exit message 
        :attr:`message`.
         
    """
    # Save which attributes are updatable for HOD as a list
    attrs = []
    for prior in priors:
        if isinstance(prior.name, basestring):
            attrs += [prior.name]
        else:
            attrs += prior.name

    # Check that attrs are all applicable
    for a in attrs:
        if a not in _vars:
            raise ValueError(a + " is not a valid variable for optimization in HaloModel")

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

    def negmod(parm, priors, h, attrs, data, sd, verbose):
        return -model(parm, priors, h, attrs, data, sd, verbose)

    res = minimize(negmod, guess, (priors, h, attrs, data, sd, verbose), tol=tol,
                   method=method, options={"disp":disp, "maxiter":maxiter})

    return res

_vars = ["wdm_mass", "delta_halo", "sigma_8", "n", "omegab", 'omegac',
         "omegav", "omegak", "H0", "M_1", 'alpha', "M_min", 'gauss_width',
         'M_0', 'fca', 'fcb', 'fs', 'delta', 'x', 'omegab_h2', 'omegac_h2', 'h']

class Uniform(object):
    """
    A Uniform prior.
    
    Parameters
    ----------
    param : str
        The name of the parameter
    
    low : float
        The lower bound of the parameter
        
    high : float
        The upper bound of the parameter
        
    """
    def __init__(self, param, low, high):
        self.name = param
        self.low = low
        self.high = high

class Normal(object):
    """
    A Gaussian prior.
    
    Parameters
    ----------
    param : str
        Name of the parameter
        
    mean : float
        Mean of the prior distribution
        
    sd : float
        The standard deviation of the prior distribution
    """
    def __init__(self, param, mean, sd):
        self.name = param
        self.mean = mean
        self.sd = sd

class MultiNorm(object):
    """
    A Multivariate Gaussian prior
    
    Parameters
    ----------
    params : list of str
        Names of the parameters (in order)
        
    means : list of float
        Mean vector of the prior distribution
        
    cov : ndarray
        Covariance matrix of the prior distribution
    """
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
