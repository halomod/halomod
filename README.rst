THIS IS THE README FOR hod
--------------------------------------------------------------------------------

hod is a python application that provides a flexible and simple interface for 
dealing with the Halo Model of Dark Matter Halos. It comes with several HOD 
models, halo density profiles, bias models and Mass Function models (through the 
hmf package, by the same people).

For a given choice of parameters (for each of the above models), it can 
calculate the large-scale structure correlation function. There is also a module
which enables fitting the correlation function model to data via MCMC.

It includes some parallelisation capabilities also (which are necessary for the
MCMC fitting for more than a couple of parameters).


INSTALLATION
--------------------------------------------------------------------------------
---- Install Requirements ------
1. Install Python
2. Install all dependencies of hmf (see readme for hmf).
	
---- Install hod --------------
>>> pip install hod 
    					 
USAGE
--------------------------------------------------------------------------------
hod can be used interactively (for instance in ipython) or in a script. 
To use interactively, in ipython do something like the following:

>>> from hod import HOD
>>> h = HOD()
>>> galcorr = h.corr_gal
>>> bias = h.bias.bias
>>> ...

All parameters to HOD() have defaults so none must be specified. There are 
quite a few that CAN be specified however. Check the docstring to see the
details. Furthermore, as hod extends the functionality of hmf, almost all
parameters accepted by hmf.Perturbations() can be used (check its docstring). 
The exception is cut_fit, which is necessarily set to False in hod. 

To change the parameters (cosmological or otherwise), one should use the 
update() method, if a HOD() object already exists. For example
 >>> h = HOD()
 >>> h.update(r=np.linspace(0.1,1,1000)) #update scale vector
 >>> corr_2h = h.corr_gal_2h #The 2-halo term of the galaxy correlation function

One can access any of the properties of the Perturbations() class for the 
given parameters through the pert attribute of HOD:

 >>> h = HOD()
 >>> mass_variance = h.pert.sigma
 >>> mass_function = h.pert.dndlnm


HISTORY
--------------------------------------------------------------------------------
1.1.1 - December 6, 2013
		Bugfixes to fit_hod() routine
		
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