"""The halomod package.

This package computes halo model quantities for dark matter and tracers.
"""

from __future__ import annotations

import contextlib

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    from importlib_metadata import PackageNotFoundError, version

with contextlib.suppress(PackageNotFoundError):
    __version__ = version(__name__)

__all__ = [
    "bias",
    "concentration",
    "cross_correlations",
    "functional",
    "halo_exclusion",
    "halo_model",
    "hod",
    "integrate_corr",
    "profiles",
    "tools",
    "wdm",
    "DMHaloModel",
    "HaloModel",
    "TracerHaloModel",
    "AngularCF",
    "ProjectedCF",
    "projected_corr_gal",
]

from . import (
    bias,
    concentration,
    cross_correlations,
    functional,
    halo_exclusion,
    halo_model,
    hod,
    integrate_corr,
    profiles,
    tools,
    wdm,
)
from .halo_model import DMHaloModel, HaloModel, TracerHaloModel
from .integrate_corr import AngularCF, ProjectedCF, projected_corr_gal
