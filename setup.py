'''
Created on Sep 9, 2013

@author: Steven
'''
from setuptools import setup, find_packages
import os
version = '0.7.0'
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="hod",
    version=version,
    packages=['hod'],
    install_requires=[],
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
    # could also include long_description, download_url, classifiers, etc.
)
