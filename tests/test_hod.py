import numpy as np
import pytest

from halomod import TracerHaloModel, hod
from halomod.hod import Zehavi05, Zehavi05Marked, Zehavi05WithMax


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
    [hod.Spinelli19],
)
def test_positive_hod(hodr):
    hm = TracerHaloModel(hod_model=hodr, transfer_model="EH")
    m = np.logspace(10, 15, 100)
    assert np.all(hm.hod.central_occupation(m) >= 0)
    assert np.all(hm.hod.satellite_occupation(m) >= 0)
    assert np.all(hm.hod._tracer_per_central(m) >= 0)
    assert np.all(hm.hod._tracer_per_satellite(m) >= 0)
    assert hm.mean_tracer_den_unit >= 0


def test_tinker05_no_divide_by_zero_warning():
    """Tinker05._satellite_occupation must not raise RuntimeWarning when m == M_min.

    The satellite formula contains exp(-M_cut / (m - M_min)), which produces a
    divide-by-zero for m == M_min.  The result is zero (exp(-inf)), but numpy
    used to emit a RuntimeWarning.  We now suppress that with np.errstate.
    """
    import warnings

    from halomod.hod import Tinker05

    hod_model = Tinker05()
    m_min = 10 ** hod_model.params["M_min"]
    # Evaluate exactly at m == m_min, which is the degenerate case.
    m = np.array([m_min * 0.5, m_min, m_min * 2.0])

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = hod_model.satellite_occupation(m)

    # The result must be finite (zero at and below m_min, positive above).
    assert np.all(np.isfinite(result))
    assert result[0] == 0.0  # below m_min
    assert result[1] == 0.0  # exactly at m_min: exp(-inf) = 0
    assert result[2] > 0.0  # above m_min
