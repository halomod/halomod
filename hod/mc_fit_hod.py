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

#===============================================================================
# The Model
#===============================================================================
def get_cg(inst):
    return inst.corr_gal

def model(parm, initial, h, attrs, data, sd):
    """
    Given a HOD() object and an arbitrary set of parameters/priors, 
    returns log prob of position (parm)
    """
    print "parm before: ", parm
    ll = 0

    #First check all values are inside boundaries (if bounds given), and
    #Get the logprob of the priors-- uniform priors don't count here
    #We reflect the variables off the edge if we need to
    i = 0
    for k, v in initial.iteritems():
        j = 0
        if v[1] == "norm":

            ll += norm.logpdf(parm[i], loc=v[2], scale=v[3])
            print ll
        elif v[1] == "unif":
            while parm[i] < v[2] or parm[i] > v[3]:
                j += 1
                if j > 10:
                    parm[i] = np.random.rand() * (v[3] - v[2]) + v[2]
                if parm[i] < v[2]:
                    parm[i] = 2 * v[2] - parm[i]
                elif parm[i] > v[3]:
                    parm[i] = 2 * v[3] - parm[i]
        i += 1

    print "parm after:", parm
    #First rebuild the hod dict from given vals
    hoddict = {attr:val for attr, val in zip(attrs, parm)}

    #print h.__dict__

    h.update(**hoddict)

    #print h.__dict__
    #The logprob of the model
    model = get_cg(h)  #data + np.random.normal(scale=0.1)
    ll += np.sum(norm.logpdf(data, loc=model, scale=sd))

    print "got here"
    return ll, parm

def fit_hod(r, data, sd, initial={}, nwalkers=100, nsamples=100, burnin=10,
           thin=50, nthreads=8, filename=None, **hodkwargs):
    """
    Runs the mcmc procedure to fit HOD() continuous params to data
    """


    if len(initial) == 0:
        raise ValueError("initial must be at least length 1")

    ###NOW TRY A LAPLACE APPROXIMATION TO FIND THE MODE
    #res = minimize()

    #Get the number of variables for MCMC
    ndim = len(initial)
    #Save which attributes are updatable for HOD as a list
    attrs = [k for k in initial]

    #Make sure hodkwargs is ok
    if nthreads > 1:
        numthreads = 1
    else:
        numthreads = 0
    if 'NumThreads' in hodkwargs:
        del hodkwargs['NumThreads']
    if 'r' in hodkwargs:
        del hodkwargs['r']

    #Initialise the HOD object - use all available cpus for this
    h = HOD(r=r, ThreadNum=0, **hodkwargs)
    #It's better to get a corr_gal instance now and then the updates are faster
    h.corr_gal

    #Now update numthreads for MCMC parallelisation
    h.update(ThreadNum=numthreads)

    #Get an array of initial values
    initial_val = np.array([val[0] for k, val in initial.iteritems()])
    #Get an initial value for all walkers, around a small ball near the initial guess
    stacked_val = initial_val
    for i in range(nwalkers - 1):
        stacked_val = np.vstack((initial_val, stacked_val))
    p0 = stacked_val * np.random.normal(loc=1.0, scale=0.2, size=ndim * nwalkers).reshape((nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, model,
                                    args=[initial, h, attrs, data, sd],
                                    threads=nthreads)

    #Run a burn-in
    pos, prob, state = sampler.run_mcmc(p0, burnin)
    sampler.reset()

    #Run the actual run
    sampler.run_mcmc(pos, nsamples, thin=thin)

    return sampler.flatchain, np.mean(sampler.acceptance_fraction)
