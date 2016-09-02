Releases
========

Development Version
~~~~~~~~~~~~~~~~~~~
Features
++++++++

Enhancements
++++++++++++

Bugfixes
++++++++


Older Versions
~~~~~~~~~~~~~~
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