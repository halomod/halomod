"""Integration-style tests of the full HaloModel class."""

import warnings

import numpy as np
import pytest
from hmf.density_field.filters import Filter
from hmf.halos.mass_definitions import MassDefinition

from halomod import DMHaloModel, TracerHaloModel
from halomod.bias import Bias
from halomod.concentration import CMRelation
from halomod.hod import HOD
from halomod.profiles import Profile


@pytest.mark.parametrize("model", [TracerHaloModel, DMHaloModel])
def test_default_actually_inits(model):
    model(transfer_model="EH")


@pytest.fixture(scope="module")
def dmhm():
    return DMHaloModel(transfer_model="EH")


@pytest.fixture(scope="module")
def thm():
    return TracerHaloModel(
        rmin=0.01,
        rmax=50,
        rnum=20,
        transfer_model="EH",
        hc_spectrum="nonlinear",
        bias_model="Mo96",
        hmf_model="PS",
    )


def test_dm_model_instances(dmhm):
    assert isinstance(dmhm.mdef, MassDefinition)
    assert isinstance(dmhm.filter, Filter)
    assert isinstance(dmhm.halo_profile, Profile)
    assert isinstance(dmhm.bias, Bias)
    assert isinstance(dmhm.halo_concentration, CMRelation)


def test_tr_model_instances(thm):
    assert isinstance(thm.mdef, MassDefinition)
    assert isinstance(thm.filter, Filter)
    assert isinstance(thm.halo_profile, Profile)
    assert isinstance(thm.bias, Bias)
    assert isinstance(thm.halo_concentration, CMRelation)
    assert isinstance(thm.hod, HOD)


@pytest.mark.filterwarnings(
    "ignore:Using halofit for tracer stats is only valid up to quasi-linear scales"
)
@pytest.mark.parametrize(
    "quantity",
    [
        "corr_linear_mm",
        "corr_halofit_mm",
        "corr_1h_auto_matter",
        "corr_2h_auto_matter",
        "corr_auto_matter",
        "corr_1h_ss_auto_tracer",
        "corr_1h_cs_auto_tracer",
        "corr_1h_auto_tracer",
        "corr_auto_tracer",
        "corr_1h_cross_tracer_matter",
        "corr_2h_cross_tracer_matter",
        "corr_cross_tracer_matter",
        "corr_2h_auto_tracer",
        # 'halo_profile_rho', 'halo_profile_lam', 'tracer_profile_rho', 'tracer_profile_lam')
    ],
)
def test_monotonic_dec(thm: TracerHaloModel, quantity):
    # Ensure it's going down (or potentially 1e-5 level numerical noise going up)
    assert np.all(np.diff(getattr(thm, quantity)) <= 1e-5)


def test_halo_power():
    """Tests the halo centre power spectrum."""
    hm = TracerHaloModel(bias_model="UnityBias", transfer_model="EH")
    assert np.allclose(hm.power_hh(hm.k_hm[:10]), hm.power_2h_auto_matter[:10], rtol=1e-2)


def test_setting_default_tracers_conc():
    """Tests setting default tracer parameters based on halo parameters."""
    hm = TracerHaloModel(
        halo_profile_model="NFW",
        tracer_profile_model="CoredNFW",
        halo_concentration_model="Ludlow16",
        tracer_concentration_model="Duffy08",
        halo_concentration_params={
            "f": 0.02,
            "C": 650,
        },
        transfer_model="EH",
    )

    assert hm.tracer_concentration.params == hm.tracer_concentration._defaults


def test_setting_default_tracers_conc_set_params():
    """Tests setting default tracer parameters based on halo parameters."""
    hm = TracerHaloModel(
        halo_profile_model="NFW",
        tracer_profile_model="NFW",
        halo_concentration_model="Ludlow16",
        tracer_concentration_model="Ludlow16",
        tracer_concentration_params={
            "f": 0.03,
            "C": 657,
        },
        transfer_model="EH",
        mdef_model="SOCritical",
    )

    assert hm.tracer_concentration.params["f"] == 0.03
    assert hm.tracer_concentration.params["C"] == 657


def test_setting_default_tracers_prof():
    """Tests setting default tracer parameters based on halo parameters."""
    hm = TracerHaloModel(
        halo_profile_model="GeneralizedNFW",
        tracer_profile_model="NFW",
        halo_concentration_model="Ludlow16",
        tracer_concentration_model="Duffy08",
        halo_profile_params={"alpha": 1.1},
        transfer_model="EH",
    )

    assert hm.tracer_profile.params == hm.tracer_profile._defaults


def test_setting_default_tracers_same_model():
    hm = TracerHaloModel(
        halo_profile_model="NFW",
        tracer_profile_model="NFW",
        halo_concentration_model="Ludlow16",
        tracer_concentration_model="Ludlow16",
        transfer_model="EH",
        mdef_model="SOCritical",
    )

    assert hm.tracer_profile.params == hm.halo_profile.params
    assert hm.halo_concentration.params == hm.tracer_concentration.params


@pytest.mark.parametrize(
    "attr",
    [
        ("halo_concentration_model"),
        ("bias_model"),
        ("hc_spectrum"),
        ("halo_profile_model"),
        ("sd_bias_model"),
        ("hod_model"),
        ("tracer_profile_model"),
        ("tracer_concentration_model"),
    ],
)
def test_raiseerror(thm: TracerHaloModel, attr):
    fakemodel = 1
    with pytest.raises(ValueError):
        setattr(thm, attr, fakemodel)


def test_large_scale_bias(dmhm):
    # First do the easiest case of a peak-background split
    dm2 = dmhm.clone(
        hc_spectrum="linear",
        force_unity_dm_bias=True,
        exclusion_model="NoExclusion",
        bias_model="Tinker10PBSplit",
        hmf_model="Tinker10",
    )

    print(dm2.halo_profile.u(dm2.k_hm[0], dm2.m, c=dm2.cmz_relation))
    assert np.isclose(dm2.power_2h_auto_matter[0], dm2.linear_power_fnc(dm2.k_hm[0]), rtol=1e-4)

    # Now do a non-pb split
    dm2.update(bias_model="Tinker10")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", "You are using an un-normalized mass function and bias function"
        )
        assert np.isclose(dm2.power_2h_auto_matter[0], dm2.linear_power_fnc(dm2.k_hm[0]), rtol=1e-4)
