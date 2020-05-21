from .halo_model import HaloModel
from hmf import get_hmf


def get_halomodel(
    required_attrs,
    get_label=True,
    kls=HaloModel,
    fast_kwargs={
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
    },
    **kwargs
):
    return get_hmf(required_attrs, get_label, kls, fast_kwargs, **kwargs)
