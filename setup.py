
import setuptools
from numpy.distutils.core import setup, Extension
import os
import sys
import io
import re

def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8")
    ) as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist upload")
    os.system("python setup.py bdist_wheel upload")
    sys.exit()


fort = Extension('halomod.fort.routines', ['halomod/fort/routines.f90'],
                     extra_f90_compile_args=['-O3', '-Wall', '-Wtabs'],
                     f2py_options=['--quiet', 'only:', 'power_gal_2h',
                                   'power_gal_1h_ss', 'corr_gal_1h_ss',
                                   'corr_gal_1h_cs', 'power_to_corr',
                                   'corr_gal_1h', 'get_subfind_centres', ':'])

corr_2h = Extension('halomod.fort.twohalo', ['halomod/fort/twohalo.f90'],
                    extra_f90_compile_args=['-Wall', '-Wtabs', '-fopenmp'],
                    f2py_options=['only:', "power_to_corr", "twohalo", "dblsimps", ":"],
                    libraries=['gomp']
                    )

if __name__ == "__main__":
    setup(
        name="halomod",
        version=find_version("halomod","__init__.py"),
        install_requires=['hmf>=2.0.0',
                          'mpmath',
                          'cached_property'],
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
        packages=['halomod', 'halomod.fort'],
        package_data={"halomod":['data/*']}
        )
