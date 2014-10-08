'''
Created on Sep 9, 2013

@author: Steven
'''
import setuptools
from numpy.distutils.core import setup, Extension
import os
import sys

version = '1.2.2'

if sys.argv[-1] == "publish":
    os.system("python setup.py install")
    os.system("python setup.py sdist upload")
    os.system("python setup.py bdist_egg upload")
    sys.exit()


fort = Extension('halomod.fort.routines', ['halomod/fort/routines.f90'],
                     extra_f90_compile_args=['-O3', '-Wall', '-Wtabs'],
                     f2py_options=['--quiet', 'only:', 'power_gal_2h',
                                   'power_gal_1h_ss', 'corr_gal_1h_ss',
                                   'corr_gal_1h_cs', 'power_to_corr',
                                   'corr_gal_1h', 'get_subfind_centres', ':'])

# hankel = Extension('hod.fort.hankel', ['hod/fort/power_to_corr.f90', 'hod/fort/utils.f90'],
#                      extra_f90_compile_args=['-O3', '-Wall', '-Wtabs'],
#                      f2py_options=['--quiet', 'skip:', 'interp', ':'])

corr_2h = Extension('halomod.fort.twohalo', ['halomod/fort/twohalo.f90'],
                    extra_f90_compile_args=['-Wall', '-Wtabs', '-fopenmp'],
                    f2py_options=['only:', "power_to_corr", "twohalo", "dblsimps", ":"],
                    libraries=['gomp']
                    )

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    os.system("python setup.py bdist_egg upload")
    sys.exit()

if __name__ == "__main__":
    setup(
        name="halomod",
        version=version,
        install_requires=['hmf',
                          'mpmath'],
        scripts=['scripts/pophod',
                 'scripts/halomod-fit'],
        author="Steven Murray",
        author_email="steven.murray@uwa.edu.au",
        description="A Halo Model calculator built on hmf",
        long_description=read('README.rst'),
        license='MIT',
        keywords="halo occupation distribution",
        url="https://github.com/steven-murray/halomod",
        ext_modules=[fort, corr_2h],
        packages=['halomod', 'halomod.fort']
        )
