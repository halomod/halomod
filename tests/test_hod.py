import pytest

import numpy as np

from halomod import TracerHaloModel, hod
from halomod.hod import Zehavi05, Zehavi05Marked, Zehavi05WithMax, Leauthaud11


def test_zehavi_marked():
    """Test the marked zehavi model to ensure it's the same as Zehavi05 by default."""
    m = np.logspace(10, 15, 100)
    z05 = Zehavi05()
    z05m = Zehavi05Marked()

    assert np.allclose(z05.nc(m), z05m.nc(m))
    assert np.allclose(z05.ns(m), z05m.ns(m))
    assert np.allclose(z05.total_occupation(m), z05m.total_occupation(m))
    assert np.allclose(z05.total_pair_function(m), z05m.total_pair_function(m))


def test_zehavi_max():
    """Test the max zehavi model to ensure it's the same as Zehavi05 by default."""
    m = np.logspace(10, 15, 100)
    z05 = Zehavi05()
    z05m = Zehavi05WithMax(M_max=18)

    assert np.allclose(z05.nc(m), z05m.nc(m))
    assert np.allclose(z05.ns(m), z05m.ns(m))
    assert np.allclose(z05.total_occupation(m), z05m.total_occupation(m))
    assert np.allclose(z05.total_pair_function(m), z05m.total_pair_function(m))


@pytest.mark.parametrize(
    "hodr",
    (hod.Spinelli19,),
)
def test_positive_hod(hodr):
    hm = TracerHaloModel(hod_model=hodr)
    m = np.logspace(10, 15, 100)
    assert np.all(hm.hod.central_occupation(m) >= 0)
    assert np.all(hm.hod.satellite_occupation(m) >= 0)
    assert np.all(hm.hod._tracer_per_central(m) >= 0)
    assert np.all(hm.hod._tracer_per_satellite(m) >= 0)
    assert hm.mean_tracer_den_unit >= 0


def test_leauthaud11():
    m = np.logspace(10, 15, 100)
    logmstar = np.linspace(7, 12, 100)

    l11 = Leauthaud11()

    assert np.isfinite(l11._central_occupation(m)).all()
    assert (l11._central_occupation(m) >= 0).all()

    assert np.isfinite(l11._satellite_occupation(m)).all()
    assert (l11._satellite_occupation(m) >= 0).all()

    assert np.isfinite(l11.mean_log_halo_mass(logmstar)).all()
    assert (l11.mean_log_halo_mass(logmstar) >= 0).all()

    assert np.isfinite(l11.mean_log_stellar_mass(m)).all()
    assert (l11.mean_log_stellar_mass(m) >= 0).all()
