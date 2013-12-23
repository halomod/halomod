'''
Created on Sep 9, 2013

@author: Steven
'''
import setuptools
from numpy.distutils.core import setup, Extension
import os
import sys

version = '1.1.5'

if sys.argv[-1] == "publish":
    os.system("python setup.py install")
    os.system("python setup.py sdist upload")
    os.system("python setup.py bdist_egg upload")
    sys.exit()

fort = Extension('hod.fort.routines', ['hod/fort/routines.f90'],
                     extra_f90_compile_args=['-O3', '-Wall', '-Wtabs'],
                     f2py_options=['--quiet', 'only:', 'power_gal_2h',
                                   'power_gal_1h_ss', 'corr_gal_1h_ss',
                                   'corr_gal_1h_cs', 'power_to_corr',
                                   'corr_gal_1h', ':'])
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    os.system("python setup.py bdist_egg upload")
    sys.exit()

if __name__ == "__main__":
    setup(
        name="hod",
        version=version,
        install_requires=['hmf'],
        author="Steven Murray",
        author_email="steven.murray@uwa.edu.au",
        description="A HOD calculator built on hmf",
        long_description=read('README.rst'),
        license='BSD',
        keywords="halo occupation distribution",
        url="https://github.com/steven-murray/hod",
        ext_modules=[fort],
        packages=['hod', 'hod.fort']
        )
