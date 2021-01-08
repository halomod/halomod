halomod
=======

.. image:: https://github.com/steven-murray/halomod/workflows/Tests/badge.svg
    :target: https://github.com/steven-murray/halomod
.. image:: https://badge.fury.io/py/halomod.svg
    :target: https://badge.fury.io/py/halomod
.. image:: https://codecov.io/gh/steven-murray/halomod/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/steven-murray/halomod
.. image:: https://img.shields.io/pypi/pyversions/halomod.svg
    :target: https://pypi.org/project/halomod/
.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

``halomod`` is a python application that provides a flexible and simple interface for
dealing with the Halo Model of Dark Matter Halos, built on
`hmf <https://github.com/steven-murray/hmf>`_.

It also has a full-feature web application at https://thehalomod.app -- check it out!

Full Documentation
------------------
`Read the docs. <https://halomod.readthedocs.io>`_

Features
--------
* All the features of ``hmf`` (several transfer function models, 15+ HMF fitting functions,
  efficient caching etc.)
* Extended components for halo model:

  * 10+ **halo bias** models, plus scale-dependent bias from Tinker (2005).
  * 3+ basic **concentration-mass-redshift relations**, including the analytic Bullock (2001) model
  * Several plug-and-play **halo-exclusion** models (including ng-matched from Tinker+2005).
  * 5+ built-in **HOD parametrisations**.
  * Many built-in **halo profiles**, including NFW, generalised NFW, Hernquist etc.
  * Support for **WDM models**.

* All basic quantities such as 3D correlations and power spectra, and projected 2PCF.
* Several derived quantities (eg. effective bias, satellite fraction).
* All models/components specifically written to be easily extendable.
* Simple routine for populating a halo catalogue with galaxies via a HOD.
* CLI script for producing any quantity included.

Usage
-----
``halomod`` can be used interactively (for instance in ``ipython`` or a ``jupyter`` notebook)
or in a script.
To use interactively, in ``ipython`` do something like the following::

    >>> from halomod import HaloModel
    >>> hm = HaloModel()  # Construct the object
    >>> help(hm)          # Lists many of the available quantities.
    >>> galcorr = hm.corr_auto_tracer
    >>> bias = hm.bias
    >>> ...

All parameters to ``HaloModel`` have defaults so none need to be specified. There are
quite a few that *can* be specified however. Check the docstring to see the
details. Furthermore, as ``halomod`` extends the functionality of
`hmf <https://github.com/steven-murray/hmf>`_, almost all parameters accepted by
``hmf.MassFunction()`` can be used (check its docstring).

To change the parameters (cosmological or otherwise), one should use the
``update()`` method, if a ``HaloModel()`` object already exists. For example

>>> hm.update(rmin=0.1,rmax=1.0,rnum=100) #update scale vector
>>> corr_2h = hm.corr_2h_auto_tracer #The 2-halo term of the galaxy correlation function

Since ``HaloModel`` is a sub-class of ``MassFunction``, all the quantities associated
with the hmf are also included, so for example

>>> mass_variance = hm.sigma
>>> mass_function = hm.dndm
>>> linear_power = hm.power

Any parameter which defines a model choice (eg. a bias model) is named ``<component>_model``,
so for example, the bias model is called ``bias_model``. *Every* model has an associated
parameter called ``<component>_params``, which is a dictionary of parameters to that
model. The available choices for this dictionary depend on the model chosen (so that the
Sheth-Tormen HMF has a different set of parameters than does the Tinker+2008 model).
Within the constructed object, the actual model is instantiated and saved as
``<component>``, so that this object can be accessed, and several internal methods can
be called. *Some* of these are exposed directly by the ``HaloModel`` class (eg. one can
call ``hm.n_sat`` directly, which itself calls a method of the ``hm.hod`` component).

Using the CLI
~~~~~~~~~~~~~
You can also run ``halomod`` from the command-line. For basic usage, do::

    halomod run --help

Configuration for the run can be specified on the CLI or via a TOML file (recommended).
An example TOML file can be found in `examples/example_run_config.toml <examples/example_run_config>`_.
Any parameter specifiable in the TOML file can alternatively be specified on the commmand
line after an isolated double-dash, eg.::

    halomod run -- z=1.0 hmf_model='SMT01' bias_model='Tinker10'

Acknowledgments
---------------
Thanks to Florian Beutler, Chris Blake and David Palamara
who have all contributed significantly to the ideas, implementation and testing
of this code.

Some parts of the code have been adapted from, influenced by or tested against:

* chomp (https://github.com/JoeMcEwen/chomp)
* AUM  (https://github.com/surhudm/aum)
* HMcode (https://github.com/alexander-mead/HMcode/)

Along with these, several other private codes have been compared to.

Attribution
-----------
Please cite

1. `Murray, Power and Robotham (2013) <https://arxiv.org/abs/1306.6721>`_ and/or https://ascl.net/1412.006 (whichever is more appropriate)
2. `Murray, Diemer and Chen (2020) <https://arxiv.org/abs/2009.14066>`_

if you find this code useful in your research. Please also consider starring the GitHub repository.
