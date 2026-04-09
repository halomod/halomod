r"""Module defining functional approaches to generating halo model quantities."""

from __future__ import annotations

from hmf import Framework, get_hmf

from .halo_model import HaloModel


def get_halomodel(
    required_attrs, get_label=True, kls=HaloModel, fast_kwargs: dict | None = None, **kwargs
) -> list[Framework]:
    r"""
    Yield framework instances for all combinations of parameters supplied.

    Returns a :func:`~hmf.helpers.functional.get_hmf`, with `framework =`
    :class:`~halomod.halo_model.HaloModel`. See
    :func:`~hmf.helpers.functional.get_hmf` for input parameters and yields.
    """
    fast_kwargs = fast_kwargs or {
        "transfer_fit": "BBKS",
        "lnk_min": -4,
        "lnk_max": 2,
        "dlnk": 1,
        "Mmin": 13,
        "dlog10m": 0.5,
        "rmin": 10,
        "rmax": 20,
        "rnum": 4,
        "halo_exclusion": "None",
        "nonlinear": False,
        "scale_dependent_bias": False,
        "hod_model": "Zehavi05",
    }
    return get_hmf(required_attrs, get_label, kls, fast_kwargs, **kwargs)
