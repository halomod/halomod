from setuptools import setup, find_packages

import os
import sys
import io
import re


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


if sys.argv[-1] == "publish":
    os.system("rm dist/*")
    os.system("python setup.py sdist")
    os.system("python setup.py bdist_wheel")
    os.system("twine upload dist/*")
    sys.exit()


test_req = [
    "coverage>=4.5.1",
    "pytest>=3.5.1",
    "pytest-cov>=2.5.1",
    "pre-commit",
    "mpmath>=1.0.0",
]

docs_req = [
    "Sphinx==1.7.5",
    "numpydoc>=0.8.0",
    "nbsphinx",
]

if __name__ == "__main__":
    setup(
        name="halomod",
        use_scm_version=True,
        setup_requires=["setuptools_scm"],
        install_requires=[
            "hmf>=3.0.8",
            "mpmath",
            "cached_property",
            "numpy",
            "scipy",
            "colossus",
        ],
        extras_require={
            "docs": docs_req,
            "tests": test_req,
            "dev": docs_req + test_req,
        },
        scripts=["scripts/pophod", "scripts/halomod-fit"],
        author="Steven Murray",
        author_email="steven.g.murray@asu.edu",
        description="A Halo Model calculator built on hmf",
        long_description=read("README.rst"),
        license="MIT",
        keywords="halo occupation distribution",
        url="https://github.com/steven-murray/halomod",
        packages=find_packages("src"),
        # package_data={"halomod": ["data/*"]},
    )
