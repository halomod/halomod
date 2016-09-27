Releases
========

Development Version
~~~~~~~~~~~~~~~~~~~
Enhancements
++++++++++++
* Fortran routines have not been used for quite a few versions, but now dependence on having gfortran has been
  removed by default. To install fortran routines, use "WITH_FORTRAN=True pip install halomod". There's no reason
  you should do this though.

Older Versions
~~~~~~~~~~~~~~
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
* mean_gal_den now returns the *calculated* mean density, rather than ng if it exists.
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
* Fixed setting of _gm (missing power of 10!)


v1.4.1
------
31st August, 2016

Features
++++++++
* Einasto profile added, with analytic h(c), and numerical u(K,c).
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
* Fixed bug when setting profile with a class rather than a string.
* Fixed bug in Geach/Contreras HOD models where they were effectively receiving a sharp cut in m (thanks to @prollejazz)

v1.4.0
------
1st August, 2016

There have have been *so many* changes since the last formal update to this package, that
it is almost pointless to list them. v1.4.0 is the first version to support hmf v2+, and
be well modularised. There are still several things that need doing reasonably urgently,
so I assume that several versions will follow rather rapidly. Tests have been performed
against other codes for this version, though they have not been formally included yet.