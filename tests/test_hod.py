from halomod.hod import Zehavi05, Zehavi05Marked, Zehavi05WithMax, ContinuousPowerLaw
import numpy as np


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
