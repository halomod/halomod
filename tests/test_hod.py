from halomod.hod import Zehavi05, Zehavi05Marked, Zehavi05WithMax, ContinuousPowerLaw
from halomod import hod, TracerHaloModel
import numpy as np
import pytest


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
    "hodr", (hod.Spinelli19,),
)
def test_positive_hod(hodr):
    hm = TracerHaloModel(hod_model=hodr)
    m = np.logspace(10, 15, 100)
    assert np.all(hm.hod.central_occupation(m) >= 0)
    assert np.all(hm.hod.satellite_occupation(m) >= 0)
    assert np.all(hm.hod._tracer_per_central(m) >= 0)
    assert np.all(hm.hod._tracer_per_satellite(m) >= 0)
    assert hm.mean_tracer_den_unit >= 0
