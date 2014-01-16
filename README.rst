---
hod
---

`hod` is a python application that provides a flexible and simple interface for 
dealing with the Halo Model of Dark Matter Halos. It comes with several HOD 
models, halo density profiles, bias models and Mass Function models (through the 
`hmf` package, by the same people).

For a given choice of parameters (for each of the above models), it can 
calculate the large-scale structure correlation function. There is also a module
which enables fitting the correlation function model to data via MCMC.

It includes some parallelisation capabilities also (which are necessary for the
MCMC fitting for more than a couple of parameters).


INSTALLATION
------------
The only tricky part of the installation should be installing `pycamb`. See
notes in the readme of `hmf`. You will also need a fortran compiler. Once `hmf`
is installed properly, simply use ``pip install hod``.
    					 
USAGE
-----
`hod` can be used interactively (for instance in `ipython`) or in a script. 
To use interactively, in `ipython` do something like the following:

>>> from hod import HOD
>>> h = HOD()
>>> galcorr = h.corr_gal
>>> bias = h.bias.bias
>>> ...

All parameters to ``HOD()`` have defaults so none must be specified. There are 
quite a few that CAN be specified however. Check the docstring to see the
details. Furthermore, as `hod` extends the functionality of `hmf`, almost all
parameters accepted by ``hmf.Perturbations()`` can be used (check its docstring). 
The exception is `cut_fit`, which is necessarily set to `False` in `hod`. 

To change the parameters (cosmological or otherwise), one should use the 
``update()`` method, if a ``HOD()`` object already exists. For example

 >>> h = HOD()
 >>> h.update(r=np.linspace(0.1,1,1000)) #update scale vector
 >>> corr_2h = h.corr_gal_2h #The 2-halo term of the galaxy correlation function

One can access any of the properties of the ``Perturbations()`` class for the 
given parameters through the ``pert`` attribute of ``HOD``:

 >>> h = HOD()
 >>> mass_variance = h.pert.sigma
 >>> mass_function = h.pert.dndlnm


HISTORY
-------
1.2.0 - 
		Added ng_matched and ellipsoid halo exclusion options
		Cleaned up and enhanced performance of halo exclusion in general
		Added ability to pass mean density and evaluate M_min from this.
		
1.1.6 - January 15, 2014
		Corrected auto-calc of nthreads in fit_hod()
		Fixed chunks output in fit_hod()
		Fixed API for updated hmf API
		Cleaned up the update process considerably
		Added an optimization routine to fit
		
1.1.5 - December 23, 2013
		Added automatic cpu counting in fit_hod()
		Better setting of number of threads in fit_hod()
		matter_power is now not deleted unless a cosmo param is changed.
		Fixed writing out file in chunks in fit_hod()
		Added some std output when chunks used in fit_hod()
		
1.1.4 - December 19, 2013
		Much needed fix for fit_hod(), in which a crash occured if nthreads>1
		
1.1.3 - December 12, 2013
		Documentation upgrade (sphinx/numpydoc format)
		:func:`fit_hod()` can now periodically output results to a file.
		Schneider halo exclusion now abs(W(kr)). This is as much a hack as the 
		exclusion itself. It goes negative and we need to take logs.
		Better initial estimation in :func:`fit_hod()`
		
1.1.2 - December 10, 2013
		A few bugfixes to match the slightly modified API of ``hmf`` v1.2.x
		
1.1.1 - December 6, 2013
		Bugfixes to :func:`fit_hod()` routine
		
1.1.0 - December 5, 2013
		Added multivariate guassian priors
		Updated to reflect changes in hmf API
		
1.0.0 - November 22, 2013
		MCMC routines now work properly -- all basic routines are in place.
		
0.7.0 - October 16, 2013
		Added ability to get HOD, cosmo params from given xi(r) data using mcmc
		
0.6.1 - October 10, 2013
		Added schneider halo_exclusion option
		
0.6.0 - October 4, 2013
		Added halo exclusion options (Most Buggy)
		Added scale-dependent bias
		Added lower mvir bound on 1h term
		Fixed nonlinear P(k)
		
0.5.1 - October 2, 2013
		Added nonlinear P(k) option
		
0.5.0 - October 2, 2013
		First working version