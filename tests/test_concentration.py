import pytest
from halomod import concentration as cm
import numpy as np
from hmf import MassFunction
from hmf.halos.mass_definitions import SOCritical, SOVirial, SOMean
from halomod import TracerHaloModel


@pytest.fixture(scope="function")
def mass():
    return np.logspace(10, 15, 100)


@pytest.mark.parametrize("mdef", [SOMean, SOCritical, SOVirial])
def test_duffy(mass, mdef):
    duffy = cm.Duffy08(sample="full", mdef=mdef())
    duffyc = cm.make_colossus_cm("duffy08")(mdef=mdef())

    assert np.allclose(duffy.cm(mass), duffyc.cm(mass))


@pytest.mark.parametrize("lu16", [cm.Ludlow2016, cm.Ludlow2016Empirical])
def test_ludlow_vs_colossus(lu16):
    """Test the Ludlow relation between native and colossus implementations."""
    mf = MassFunction(transfer_model="EH")

    L16Colossus = cm.make_colossus_cm(model="ludlow16")

    l16 = lu16(filter0=mf.normalised_filter)
    l16c = L16Colossus(filter0=mf.normalised_filter)

    m = np.logspace(10, 15, 100)

    # TODO: for masses of ~1e10, halomod gets c(m) = 16, while colossus gets ~12. Need to fix.
    assert np.allclose(l16.cm(m), l16c.cm(m), rtol=0.3)


def test_lud16em_warning():
    mf = MassFunction()
    l16 = cm.Ludlow2016Empirical(filter0=mf.normalised_filter)
    m = np.logspace(11, 15, 100)
    with pytest.warns(UserWarning):
        l16.cm(m, z=1)


def test_lud16_scalarm():
    mf = MassFunction()
    L16Colossus = cm.make_colossus_cm(model="ludlow16")
    l16 = cm.Ludlow2016(filter0=mf.normalised_filter)
    l16c = L16Colossus(filter0=mf.normalised_filter)

    assert np.allclose(l16.cm(1e12), l16c.cm(1e12), rtol=0.2)


@pytest.mark.parametrize(
    "cmr",
    (
        cm.Bullock01,
        cm.Bullock01Power,
        cm.Maccio07,
        cm.Duffy08,
        cm.Zehavi11,
        cm.Ludlow16,
        cm.Ludlow16Empirical,
    ),
)
def test_decreasing_cm(cmr):
    hm = TracerHaloModel(halo_concentration_model=cmr)
    m = np.logspace(10, 15, 100)
    assert np.all(np.diff(hm.halo_concentration.cm(m, z=0)) <= 0)
