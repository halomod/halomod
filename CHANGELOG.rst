Releases
========

dev-version
-----------

v2.0.1 [26 Feb 2021]
----------------------

v2.0.3 [26 Feb 2021]
----------------------

v2.0.2 [26 Feb 2021]
----------------------

Changes
+++++++
* Update tutorial to match the current version.
* new ``validation()`` method from ``hmf v3.3.4`` is used to check validity
  of simple inputs, like ``k_hm`` and ``r``.

2.0.0 [25th Nov 2020]
---------------------
This is a **major** new release that brings halomod into the properly
Python 3 world, and tightens up a lot of issues. It also corresponds to the
submission of `Murray, Diemer and Chen (2020) <https://arxiv.org/abs/2009.14066>`_.

Features
++++++++

* More accurate hankel transforms due to higher-res background table of r values
* ExtendedSpline class which does better (and more customized) extrapolation on splines.
* Added ability to change the underlying distribution of satellite and central counts.
* Python 3!
* New ``UnityBias`` component
* Ability to do cross-correlations between one tracer and another
* Delineation between DM and halos in power spectra
* Much more documentation
* Addition of ``CoredNFW`` profile
* Added ``mean_tracer_den_unit`` -- the mean density of whatever tracer you use.
* Interfaces with COLOSSUS for concentration and bias.
* New CLI interface ``halomod run``
* New examples, including an example of parameter fitting with ``halomod`` and ``emcee``.
* New hod, density profile and concentration model, with new example, for HI.

Changes
+++++++

* Removed dynamic ``mmin``.
* No more support for Python 2.

Bugfixes
++++++++

* Fix for ``DblSphere`` halo exclusion model.
* Fix for when no tracer profile/concentration params are given. Now use the halo params
  iff the tracer model is equal to the halo one.
* Fix for halo profile numerical FT.
* Fix #59 -- can't have non-dict ``model_params``.
* Fix for calculation of halo centre power spectrum
* Fix for tracer params initialisation.
* Fix amplitude of ``angular_corr_gal`` when using ``p_of_z=True``
  (`Issue #63 <https://github.com/steven-murray/halomod/issues/63>`_,
   `PR #72 <https://github.com/steven-murray/halomod/pull/72>`_)

v1.4.5
------
Enhancements
++++++++++++
* Added option to HaloModel class to turn off forcing the turnover at large scales of the 1-halo term power spectrum.


v1.4.4
------
15th February, 2017

Features
++++++++
* New power_hh method which calculates the properly-biased halo-halo power spectrum in a given mass range.

Enhancements
++++++++++++
* Fortran routines have not been used for quite a few versions, but now dependence on having gfortran has been
  removed by default. To install fortran routines, use "WITH_FORTRAN=True pip install halomod". There's no reason
  you should do this though.
* Updated decorators to connect with new hmf caching system
* populate routine now supports multi-processing for several-times speedup.

Bugfixes
++++++++
* fixed a bug in populate routine where if there were zero satellites it would raise an exception.


v1.4.3
------
23rd September, 2016

Features
++++++++
* Function to populate a halo catalogue with HOD-derived galaxies now properly implemented.
* Populate routine also returns an array with indices of the halos associated with each galaxy.

Enhancements
++++++++++++
* When matching mean galaxy density to given value, the first guess is now the DM Mmin, rather than arbitrary 8.
* Better error message for NGException
* mean_tracer_den now returns the *calculated* mean density, rather than tracer_density if it exists.
* If supplied redshift is outside redshift selection for AngularCF, warning is printed.
* Mmin now set to 0 by default, to enable better matter-matter statistics.
* Entirely revised system for HODs, especially concerning the "central condition":
  * Ns, not Ntot, now modified, giving consistent results for all derived quantities
  * Pair-counts now intrinsic to HOD class.
  * ``populate()`` routine handles both cases -- where centrals are required, and not.
  * Documentation in HOD module explaining the assumptions made.

Bugfixes
++++++++
* __density_mod_mm added so that __density_mod not overwritten when getting matter correlations.
* __density_mod_mm modified to account for the fact that m[0] != 0, when halo exclusion performed.
* Several fixes for correlation functions not being counts (+1 errors)



v1.4.2
------
2nd September, 2016

Bugfixes
++++++++
* Fixed setting of _tm (missing power of 10!)


v1.4.1
------
31st August, 2016

Features
++++++++
* Einasto halo_profile added, with analytic h(c), and numerical u(K,c).
* Concentration relations from Ludlow+2016 added -- both empirical and analytic model.

Enhancements
++++++++++++
* Changed some default values in halo profiles to be in line with common expectation.
* HOD models now by default have the ``mmin`` property as ``None``, which results in the galaxy mass range
  equalling the DM mass range.

Bugfixes
++++++++
* Fixed extra white-space bug in version number
* Several fixes for WDM models to bring them into line with hmf v2+
* Fixed issue with Mmin not affecting m on update.
* Fixed bug when setting halo_profile with a class rather than a string.
* Fixed bug in Geach/Contreras HOD models where they were effectively receiving a sharp cut in m (thanks to @prollejazz)

v1.4.0
------
1st August, 2016

There have have been *so many* changes since the last formal update to this package, that
it is almost pointless to list them. v1.4.0 is the first version to support hmf v2+, and
be well modularised. There are still several things that need doing reasonably urgently,
so I assume that several versions will follow rather rapidly. Tests have been performed
against other codes for this version, though they have not been formally included yet.
