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
    return DMHaloModel(
        transfer_model="EH",
        bias_model="Tinker10PBSplit",
        hmf_model="Tinker10",
    )


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


def test_passing_r_array(dmhm):
    rr = dmhm.r.copy()
    dmhm2 = dmhm.clone(rmin=rr)
    assert np.allclose(dmhm.r, dmhm2.r)
    assert np.allclose(dmhm.corr_auto_matter, dmhm2.corr_auto_matter)


def test_2h_tracer_smooth_mmin():
    """The 2-halo tracer power spectrum must vary smoothly as Mmin is swept.

    Previously, the lower mass bound was applied as a discrete grid mask (_tm),
    causing step-wise jumps whenever Mmin crossed a mass grid point.  Now it is
    handled via spline integration so the result changes continuously.
    """
    # Use a coarse mass grid (dlog10m=0.05) so grid crossings are easy to detect.
    base_kw = {
        "transfer_model": "EH",
        "hod_model": "Tinker05",
        "dlog10m": 0.05,
        "rmin": 1.0,
        "rmax": 50.0,
        "rnum": 5,
        "dlnk": 0.2,
    }

    # Sweep M_min across a full grid spacing (0.05 dex) spanning two grid points.
    mmin_values = np.linspace(11.55, 11.65, 12)
    k_idx = 5  # a mid-range k value

    powers = []
    for mmin in mmin_values:
        thm = TracerHaloModel(**base_kw, hod_params={"M_min": mmin})
        powers.append(thm.power_2h_auto_tracer[k_idx])

    powers = np.asarray(powers)

    # The power should vary monotonically (or very nearly so) as Mmin increases.
    # With the old discrete mask, large discrete jumps occurred at grid crossings.
    # With spline integration the second differences must be small.
    second_diff = np.abs(np.diff(powers, n=2))
    # 1 % of the mean power is a generous tolerance for smooth variation; a
    # grid-boundary discontinuity in the old code would produce jumps of order
    # ~10 % or more.
    assert np.all(second_diff < 0.01 * np.abs(powers).mean()), (
        f"2-halo tracer power is not smooth w.r.t. Mmin. "
        f"Max |Δ²P| = {second_diff.max():.3e}, "
        f"mean |P| = {np.abs(powers).mean():.3e}"
    )
    
    
@pytest.mark.filterwarnings("ignore:You are setting hod_params directly.")
def test_no_parameter_sharing_between_tracer_instances():
    """Regression test for https://github.com/halomod/halomod/issues/202.

    Two independently-created TracerHaloModel instances must not share parameter
    dicts (hod_params, halo_profile_params, etc.).
    """
    hm1 = TracerHaloModel(transfer_model="EH")
    hm1.hod_params = {"M_1": 12.0}

    hm2 = TracerHaloModel(hod_model="Zheng05", transfer_model="EH")

    # hm2 should have empty hod_params, independent of hm1
    assert hm2.hod_params == {}, (
        f"hm2.hod_params should be empty but got {hm2.hod_params!r}. "
        "Instances appear to be sharing parameter dicts."
    )
    assert hm1.hod_params == {"M_1": 12.0}, (
        f"hm1.hod_params was unexpectedly modified: {hm1.hod_params!r}"
    )

    # Modifying hm2 should not affect hm1
    hm2.hod_params = {"M_1": 13.0}
    assert hm1.hod_params == {"M_1": 12.0}, (
        f"Modifying hm2 affected hm1.hod_params: {hm1.hod_params!r}"
    )


@pytest.mark.filterwarnings("ignore:You are setting halo_profile_params directly.")
def test_no_parameter_sharing_between_dm_instances():
    """Regression test: DMHaloModel instances must not share parameter dicts."""
    hm1 = DMHaloModel(transfer_model="EH")
    hm1.halo_profile_params = {"truncate": False}

    hm2 = DMHaloModel(transfer_model="EH")

    assert hm2.halo_profile_params == {}, (
        f"hm2.halo_profile_params should be empty but got {hm2.halo_profile_params!r}."
    )
    assert hm1.halo_profile_params == {"truncate": False}, (
        f"hm1.halo_profile_params was unexpectedly modified: {hm1.halo_profile_params!r}"
    )


@pytest.fixture(scope="module")
def thm_centrals_only():
    """TracerHaloModel with effectively no satellites (M_1 >> any halo mass)."""
    return TracerHaloModel(
        transfer_model="EH",
        bias_model="Mo96",
        hmf_model="PS",
        hod_model="Zehavi05",
        hod_params={"M_1": 20.0, "alpha": 1.0},  # M_1 = 10^20 Msun => N_s ~ 0
    )


def test_1h_cross_tracer_matter_centrals_only_independent_of_tracer_profile(thm_centrals_only):
    """1-halo cross power must not depend on tracer profile when N_s=0.

    Central galaxies sit at the halo centre and carry no profile factor u_t(k|M).
    With a centrals-only HOD, changing the tracer concentration model must leave
    power_1h_cross_tracer_matter unchanged (Cacciato+2009, eq. 13).
    """
    thm_alt = thm_centrals_only.clone(tracer_concentration_model="Maccio07")
    assert np.allclose(
        thm_centrals_only.power_1h_cross_tracer_matter,
        thm_alt.power_1h_cross_tracer_matter,
        rtol=1e-5,
    )


def test_2h_cross_tracer_matter_centrals_only_independent_of_tracer_profile(thm_centrals_only):
    """2-halo cross power must not depend on tracer profile when N_s=0.

    The tracer-side bias integral bt = integral dndm * b * (N_c + N_s*u_t).
    With N_s=0 the u_t factor drops out entirely, so changing the tracer
    concentration model must leave power_2h_cross_tracer_matter unchanged
    (Cacciato+2009, eq. 20-21).
    """
    thm_alt = thm_centrals_only.clone(tracer_concentration_model="Maccio07")
    assert np.allclose(
        thm_centrals_only.power_2h_cross_tracer_matter,
        thm_alt.power_2h_cross_tracer_matter,
        rtol=1e-5,
    )


@pytest.mark.filterwarnings("ignore:You are using an un-normalized mass function")
@pytest.mark.parametrize("model", [TracerHaloModel, DMHaloModel])
def test_pickle_before_any_computation(model):
    """Models must be pickleable even before any quantities are computed (for MCMC)."""
    import pickle

    m = model(transfer_model="EH")
    p = pickle.dumps(m)
    m2 = pickle.loads(p)
    assert type(m2) is type(m)
    # Verify the unpickled model can still compute quantities
    assert np.allclose(m.corr_auto_matter, m2.corr_auto_matter)


@pytest.mark.filterwarnings(
    "ignore:Using halofit for tracer stats is only valid up to quasi-linear scales"
)
@pytest.mark.filterwarnings("ignore:You are using an un-normalized mass function")
def test_pickle_after_computation(thm):
    """Models must be pickleable after computing cached quantities (for MCMC)."""
    import pickle

    # Trigger computation of several cached quantities
    _ = thm.corr_auto_tracer
    _ = thm.corr_auto_matter

    p = pickle.dumps(thm)
    thm2 = pickle.loads(p)

    # Verify that the unpickled model produces the same results
    assert np.allclose(thm.corr_auto_tracer, thm2.corr_auto_tracer)
    assert np.allclose(thm.corr_auto_matter, thm2.corr_auto_matter)
