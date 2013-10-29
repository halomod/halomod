'''
Created on Sep 9, 2013

@author: Steven
'''
import setuptools
from numpy.distutils.core import setup, Extension
import os
version = '0.7.0'

fort = Extension('hod.fort.routines', ['hod/fort/routines.f90'],
                     extra_f90_compile_args=['-O3', '-Wall', '-Wtabs'],
                     f2py_options=['--quiet', 'only:', 'power_gal_2h',
                                   'power_gal_1h_ss', 'corr_gal_1h_ss',
                                   'corr_gal_1h_cs', 'power_to_corr', ':'])
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()
if __name__ == "__main__":
    setup(
        name="hod",
        version=version,
        requires=['numpy',
                  'scipy',
                  'hmf'],
        author="Steven Murray",
        author_email="steven.murray@uwa.edu.au",
        description="A HOD calculator built on hmf",
        long_description=read('README'),
        license='BSD',
        keywords="halo occupation distribution",
        url="https://github.com/steven-murray/hod",
        ext_modules=[fort],
        packages=['hod', 'hod.fort']
        )
