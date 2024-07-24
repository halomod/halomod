import numpy as np
import pytest
from halomod import TracerHaloModel
from halomod import concentration as cm
from hmf import MassFunction
from hmf.halos.mass_definitions import SOCritical, SOMean, SOVirial


@pytest.fixture()
def mass():
    return np.logspace(10, 15, 100)


@pytest.mark.parametrize("mdef", [SOMean, SOCritical, SOVirial])
def test_duffy(mass, mdef):
    duffy = cm.Duffy08(sample="full", mdef=mdef())
    duffyc = cm.make_colossus_cm("duffy08")(mdef=mdef())

    with pytest.warns(
        UserWarning,
        match="Some masses or redshifts are outside the validity of the concentration model",
    ):
        assert np.allclose(duffy.cm(mass), duffyc.cm(mass))


@pytest.mark.parametrize("lu16", [cm.Ludlow16, cm.Ludlow16Empirical])
def test_ludlow_vs_colossus(lu16):
    """Test the Ludlow relation between native and colossus implementations."""
    mf = MassFunction(transfer_model="EH")

    L16Colossus = cm.make_colossus_cm(model="ludlow16")

    l16 = lu16(filter0=mf.normalised_filter)
    l16c = L16Colossus(filter0=mf.normalised_filter)

    m = np.logspace(10, 15, 100)

    # TODO: for masses of ~1e10, halomod gets c(m) = 16, while colossus gets ~12. Need to fix.
    assert np.allclose(l16.cm(m), l16c.cm(m), rtol=0.3)


def test_lud16_scalarm():
    mf = MassFunction(transfer_model="EH")
    L16Colossus = cm.make_colossus_cm(model="ludlow16")
    l16 = cm.Ludlow16(filter0=mf.normalised_filter)
    l16c = L16Colossus(filter0=mf.normalised_filter)

    assert np.allclose(l16.cm(1e12), l16c.cm(1e12), rtol=0.2)


@pytest.mark.filterwarnings("ignore:Requested mass definition")
@pytest.mark.parametrize(
    "cmr",
    [
        cm.Bullock01,
        cm.Bullock01Power,
        cm.Maccio07,
        cm.Duffy08,
        cm.Zehavi11,
        cm.Ludlow16,
        cm.Ludlow16Empirical,
    ],
)
def test_decreasing_cm(cmr):
    # mass definition is not right for all these, but it doesn't matter for this test.
    hm = TracerHaloModel(halo_concentration_model=cmr, transfer_model="EH")
    m = np.logspace(10, 15, 100)
    assert np.all(np.diff(hm.halo_concentration.cm(m, z=0)) <= 0)
