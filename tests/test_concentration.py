import pytest
from halomod import concentration as cm
import numpy as np
from hmf import MassFunction
from hmf.halos.mass_definitions import SOCritical, SOVirial, SOMean


@pytest.fixture(scope="function")
def mass():
    return np.logspace(10, 15, 100)


@pytest.mark.parametrize("mdef", [SOMean, SOCritical, SOVirial])
def test_duffy(mass, mdef):
    duffy = cm.Duffy08(sample="full", mdef=mdef())
    duffyc = cm.make_colossus_cm("duffy08")(mdef=mdef())

    assert np.allclose(duffy.cm(mass), duffyc.cm(mass))


def test_ludlow_vs_colossus():
    """Test the Ludlow relation between native and colossus implementations."""
    mf = MassFunction()

    L16Colossus = cm.make_colossus_cm(model="ludlow16")

    l16 = cm.Ludlow16(filter0=mf.normalised_filter)
    l16c = L16Colossus(filter0=mf.normalised_filter)

    m = np.logspace(10, 15, 100)

    # TODO: for masses of ~1e10, halomod gets c(m) = 16, while colossus gets ~12. Need to fix.
    assert np.allclose(l16.cm(m), l16c.cm(m), rtol=0.3)
