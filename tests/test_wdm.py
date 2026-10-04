from __future__ import annotations

import warnings

import numpy as np
import pytest

from halomod import DMHaloModel
from halomod.concentration import CMRelation, Duffy08, Ludlow16
from halomod.wdm import CMRelationWDMRescaled, HaloModelWDM, TracerHaloModelWDM


def test_cmz_wdm():
    wdm = HaloModelWDM(
        hmf_model="SMT",
        z=0,
        hmf_params={"a": 1},
        filter_model="SharpK",
        filter_params={"c": 2.5},
        halo_concentration_model="Duffy08WDM",
        wdm_mass=3.3,
        Mmin=7.0,
        transfer_model="EH",
    )
    cdm = DMHaloModel(
        hmf_model="SMT",
        z=0,
        hmf_params={"a": 1},
        filter_model="SharpK",
        filter_params={"c": 2.5},
        halo_concentration_model="Duffy08",
        Mmin=7.0,
        transfer_model="EH",
    )

    assert np.all(cdm.cmz_relation[cdm.m <= wdm.wdm.m_hm] > wdm.cmz_relation[wdm.m <= wdm.wdm.m_hm])


@pytest.mark.filterwarnings("ignore:Your input mass definition")
def test_ludlow_cmz_wdm():
    """WDM Ludlow16 concentrations lie below the CDM ones below the half-mode mass."""
    # SOCritical(200) is the definition Ludlow16 is calibrated in, but it does not
    # match the SOVirial definition SMT was measured in. We don't care about the
    # mass function here (the c(M) relation does not depend on it), so let hmf
    # convert it rather than error on the mismatch.
    wdm = HaloModelWDM(
        hmf_model="SMT",
        z=0,
        hmf_params={"a": 1},
        filter_model="TopHat",
        mdef_model="SOCritical",
        disable_mass_conversion=False,
        halo_concentration_model="Ludlow16",
        halo_profile_model="Einasto",
        wdm_mass=3.3,
        Mmin=7.0,
        transfer_model="EH",
    )
    cdm = DMHaloModel(
        hmf_model="SMT",
        z=0,
        hmf_params={"a": 1},
        filter_model="TopHat",
        halo_concentration_model="Ludlow16",
        halo_profile_model="Einasto",
        Mmin=7.0,
        mdef_model="SOCritical",
        disable_mass_conversion=False,
        transfer_model="EH",
    )

    assert np.all(cdm.cmz_relation[cdm.m <= wdm.wdm.m_hm] > wdm.cmz_relation[wdm.m <= wdm.wdm.m_hm])


def test_wdm_cm_factory_does_not_mutate_parent_defaults():
    """The WDM rescaling parameters must not leak into the CDM model (#274)."""
    cdm_defaults = dict(Duffy08._defaults)

    wdm_cls = CMRelationWDMRescaled("Duffy08WDM")

    assert Duffy08._defaults == cdm_defaults
    assert wdm_cls._defaults is not Duffy08._defaults
    assert wdm_cls._defaults == {
        **cdm_defaults,
        "g1": 60,
        "g2": 0.17,
        "beta0": 0.026,
        "beta1": 0.04,
    }
    with pytest.raises(ValueError, match="g1"):
        Duffy08(g1=1)


def test_wdm_cm_factory_is_memoised():
    """Repeated calls return one class, registered once in the plugin registry."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        k1 = CMRelationWDMRescaled("Duffy08WDM")
        k2 = CMRelationWDMRescaled("Duffy08WDM")
        k3 = CMRelationWDMRescaled("Duffy08")

    assert k1 is k2
    assert k1 is k3
    assert k1.__name__ == "Duffy08WDM"
    assert issubclass(k1, Duffy08)
    assert CMRelation._plugins["Duffy08WDM"] is k1
    assert CMRelation._plugins["Duffy08"] is Duffy08

    # Two halo models asking for the same WDM model share its class.
    hm1 = HaloModelWDM(halo_concentration_model="Duffy08WDM", transfer_model="EH")
    hm2 = HaloModelWDM(halo_concentration_model="Duffy08WDM", transfer_model="EH")
    assert hm1.halo_concentration_model is hm2.halo_concentration_model is k1


def test_wdm_cm_reduces_to_cdm_for_vanishing_half_mode_mass():
    """With m_hm -> 0 (i.e. CDM) the rescaled c(M) is the CDM one at z=0."""
    m = np.logspace(6, 15, 50)
    cdm = Duffy08().cm(m, z=0)
    wdm = CMRelationWDMRescaled("Duffy08")(m_hm=1e-10).cm(m, z=0)

    np.testing.assert_allclose(wdm, cdm, rtol=1e-6, atol=0)


def test_wdm_cm_suppressed_below_half_mode_mass():
    """WDM suppresses the concentration of haloes below the half-mode mass."""
    m_hm = 1e10
    m = np.logspace(6, 10, 30)
    cdm = Duffy08().cm(m, z=0)
    wdm = CMRelationWDMRescaled("Duffy08")(m_hm=m_hm).cm(m, z=0)

    assert np.all(wdm < cdm)
    assert np.all(wdm > 0)


def test_wdm_cm_redshift_factor():
    """Far above m_hm, only the (1+z)**(beta0*z - beta1) factor differs from CDM."""
    z = 4.0
    m = np.logspace(14, 15, 10)
    cdm = Duffy08().cm(m, z=z)
    wdm_model = CMRelationWDMRescaled("Duffy08")(m_hm=1e3)
    wdm = wdm_model.cm(m, z=z)

    b0 = wdm_model.params["beta0"]
    b1 = wdm_model.params["beta1"]
    expected_ratio = (1 + z) ** (b0 * z - b1)
    assert expected_ratio > 1.05  # the factor is non-trivial at this redshift
    np.testing.assert_allclose(wdm / cdm, expected_ratio, rtol=1e-6, atol=0)


def test_cmz_wdm_values_unchanged():
    """Regression: values for the ``test_cmz_wdm`` setup are unchanged by #274."""
    wdm = HaloModelWDM(
        hmf_model="SMT",
        z=0,
        hmf_params={"a": 1},
        filter_model="SharpK",
        filter_params={"c": 2.5},
        halo_concentration_model="Duffy08WDM",
        wdm_mass=3.3,
        Mmin=7.0,
        transfer_model="EH",
    )
    assert wdm.halo_concentration.m_hm == wdm.wdm.m_hm

    idx = [0, len(wdm.m) // 4, len(wdm.m) // 2, -1]
    np.testing.assert_allclose(
        wdm.m[idx],
        [10000000.0, 5623413251.902732, 3162277660167.5254, 9.772372209552876e17],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        wdm.cmz_relation[idx],
        [8.037842748293494, 12.615424988842374, 8.8503029823488, 2.8391896263508802],
        rtol=1e-12,
    )


def test_subclass_of_wdm_cm_relation():
    """A subclass of a generated WDM class can be used without infinite recursion."""
    base = CMRelationWDMRescaled("Duffy08")

    class MyDuffy08WDM(base):
        pass

    m = np.logspace(8, 14, 10)
    sub = MyDuffy08WDM(m_hm=1e10)
    assert sub.m_hm == 1e10
    np.testing.assert_allclose(sub.cm(m, z=0.5), base(m_hm=1e10).cm(m, z=0.5), rtol=1e-12)


@pytest.mark.filterwarnings("ignore:Requested mass definition")
@pytest.mark.parametrize("framework", [HaloModelWDM, TracerHaloModelWDM])
def test_wdm_default_concentration_is_ludlow16(framework):
    """The WDM frameworks default to Ludlow16, not its deprecated alias."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        hm = framework(transfer_model="EH", Mmin=7.0)
        assert hm.halo_concentration_model is Ludlow16
        assert isinstance(hm.halo_concentration, Ludlow16)
