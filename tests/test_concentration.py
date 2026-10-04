import numpy as np
import pytest
from hmf import MassFunction
from hmf.halos.mass_definitions import SOCritical, SOMean, SOVirial

from halomod import TracerHaloModel
from halomod import concentration as cm


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


# Both implementations use the same (non-native) mass definition, so the mismatch is moot.
@pytest.mark.filterwarnings("ignore:Requested mass definition")
@pytest.mark.parametrize("z", [1.0, 2.0, 4.0])
def test_ludlow16_vs_colossus_high_z(z):
    """Ludlow16 at z > 0 agrees with the independent COLOSSUS implementation.

    With hmf 3.6.0, ``GrowthFactor.growth_factor`` was wrong for redshift arrays
    reaching the radiation era, which pinned Ludlow16 at its c=100 bracket edge for
    z >= 2 (#268).
    """
    m = np.logspace(10, 15, 20)
    hm = TracerHaloModel(transfer_model="EH", halo_concentration_model="Ludlow16", z=z)
    l16 = hm.halo_concentration
    l16c = cm.make_colossus_cm(model="ludlow16")(filter0=l16.filter, cosmo=l16.cosmo, mdef=l16.mdef)

    c = l16.cm(m, z=z)
    assert np.all((c > 1.5) & (c < 15))
    assert np.allclose(c, l16c.cm(m, z=z), rtol=0.06)


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

    # we test for the interpolation as well
    hm_interp = TracerHaloModel(
        halo_concentration_model=cm.interp_concentration(cmr), transfer_model="EH"
    )
    assert np.all(np.diff(hm_interp.halo_concentration.cm(m, z=0)) <= 0)


@pytest.mark.filterwarnings("ignore:Requested mass definition")
@pytest.mark.parametrize("z", [0.0, 2.0, 6.0, 10.0])
def test_cm_mass_nonlinear_matches_hmf(z):
    """CMRelation.mass_nonlinear agrees with hmf's independent root find (#265).

    At z=10 the old ``minimize``-based search silently returned ~6e-16.
    """
    hm = TracerHaloModel(z=z, transfer_model="EH", halo_concentration_model="Bullock01Power")
    assert hm.halo_concentration.mass_nonlinear(z) == pytest.approx(hm.mass_nonlinear, rel=1e-3)


@pytest.mark.filterwarnings("ignore:Requested mass definition")
def test_cm_mass_nonlinear_raises_when_undefined():
    """At z=15, sigma(R) D(z) < delta_c on all scales, so there is no nonlinear mass."""
    hm = TracerHaloModel(z=15, transfer_model="EH", halo_concentration_model="Bullock01Power")
    with pytest.raises(ValueError, match="Cannot find the nonlinear mass"):
        hm.halo_concentration.mass_nonlinear(15.0)
