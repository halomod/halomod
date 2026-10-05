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


# On hmf < 3.7, get_all_parameter_names() instantiates a default (CAMB) model, which
# warns that extrapolate_with_eh was not set; that is irrelevant to this check.
@pytest.mark.filterwarnings("ignore:'extrapolate_with_eh' was not set")
def test_force_unity_dm_bias_is_parameter():
    """force_unity_dm_bias must be a declared parameter (so update/clone/CLI accept it)."""
    assert "force_unity_dm_bias" in DMHaloModel.get_all_parameter_names()
    assert "force_unity_dm_bias" in TracerHaloModel.get_all_parameter_names()


@pytest.mark.filterwarnings("ignore:You are using an un-normalized mass function")
def test_force_unity_dm_bias_update_invalidates_2h():
    """Updating force_unity_dm_bias must recompute the cached 2-halo matter power.

    Regression test for halomod/halomod#264: previously the update left the cached
    2-halo terms stale (hmf <= 3.6) or raised (hmf >= 3.7).
    """
    kw = {"transfer_model": "EH"}
    hm = DMHaloModel(force_unity_dm_bias=True, **kw)
    p_true = hm.power_2h_auto_matter.copy()

    hm.update(force_unity_dm_bias=False)
    assert hm.force_unity_dm_bias is False
    p_updated = hm.power_2h_auto_matter

    p_fresh = DMHaloModel(force_unity_dm_bias=False, **kw).power_2h_auto_matter
    np.testing.assert_allclose(p_updated, p_fresh, rtol=1e-10)

    # The naive effective bias is not unity on the default model, so the results
    # must actually differ (by much more than numerical noise).
    assert not np.allclose(p_updated, p_true, rtol=1e-3)


@pytest.mark.filterwarnings("ignore:You are using an un-normalized mass function")
def test_force_unity_dm_bias_clone():
    """clone(force_unity_dm_bias=...) works and matches a fresh instance."""
    kw = {"transfer_model": "EH"}
    hm = DMHaloModel(force_unity_dm_bias=True, **kw)
    clone = hm.clone(force_unity_dm_bias=False)

    assert hm.force_unity_dm_bias is True
    assert clone.force_unity_dm_bias is False
    np.testing.assert_allclose(
        clone.power_2h_auto_matter,
        DMHaloModel(force_unity_dm_bias=False, **kw).power_2h_auto_matter,
        rtol=1e-10,
    )


@pytest.mark.filterwarnings("ignore:You are using an un-normalized mass function")
def test_force_unity_dm_bias_update_recovers_linear_power():
    """Switching force_unity_dm_bias on via update() recovers linear power at large scales.

    With unit matter bias, no exclusion and a linear halo-centre spectrum, the 2-halo
    matter power on the largest scales (where u(k|m) -> 1) must equal linear power.
    """
    hm = DMHaloModel(
        transfer_model="EH",
        hc_spectrum="linear",
        exclusion_model="NoExclusion",
        bias_model="Tinker10PBSplit",
        hmf_model="Tinker10",
        force_unity_dm_bias=False,
    )
    k0 = hm.k_hm[0]
    # Without the renormalization, the finite-mass-range integral falls short of unity.
    assert not np.isclose(hm.power_2h_auto_matter[0], hm.linear_power_fnc(k0), rtol=1e-2)

    hm.update(force_unity_dm_bias=True)
    assert hm.bias_effective_matter == 1.0
    assert np.isclose(hm.power_2h_auto_matter[0], hm.linear_power_fnc(k0), rtol=1e-4)


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


# ---------------------------------------------------------------------------------------
# Pairing of hmf_model with bias_model (#275)
# ---------------------------------------------------------------------------------------
#: The hmf_model that ``TracerHaloModel(bias_model=X)`` resolved to before #275 was fixed
#: (recorded on the commit before the fix). The fix must not change these.
PAIRED_HMF_BEFORE_275 = {
    "Jing98": "Tinker10",
    "Mandelbaum05": "SMT",
    "Manera10": "Manera",
    "Mo96": "PS",
    "SMT01": "SMT",
    "ST99": "SMT",
    "Tinker05": "SMT",
    "Tinker10": "Tinker10",
    "Tinker10PBSplit": "Tinker10",
    "UnityBias": "PS",
}

BIAS_WITH_PAIR = sorted(name for name, mdl in Bias.get_models().items() if mdl.pair_hmf)


def _assert_same_model(a: TracerHaloModel, b: TracerHaloModel, rtol: float) -> None:
    assert a.hmf_model is b.hmf_model
    assert a.bias_model is b.bias_model
    np.testing.assert_allclose(a.dndm, b.dndm, rtol=rtol, atol=0)
    np.testing.assert_allclose(a.mean_tracer_den, b.mean_tracer_den, rtol=rtol, atol=0)
    np.testing.assert_allclose(a.power_auto_tracer, b.power_auto_tracer, rtol=rtol, atol=0)


def test_bias_models_with_pair_are_covered():
    """The models checked below must include the ones the issue is about."""
    assert {"ST99", "SMT01", "Mo96", "Manera10", "Tinker10PBSplit"} <= set(BIAS_WITH_PAIR)


@pytest.mark.filterwarnings("ignore:You are using an un-normalized mass function")
@pytest.mark.filterwarnings("ignore:Requested mass definition 'FoF")
@pytest.mark.parametrize("bias_model", BIAS_WITH_PAIR)
def test_update_bias_model_matches_constructor(bias_model):
    """Updating bias_model must give the same model as passing it to the constructor."""
    fresh = TracerHaloModel(transfer_model="EH", bias_model=bias_model)
    updated = TracerHaloModel(transfer_model="EH")
    updated.update(bias_model=bias_model)

    assert updated.hmf_model is fresh.bias_model.pair_hmf[0]
    _assert_same_model(updated, fresh, rtol=1e-10)


@pytest.mark.filterwarnings("ignore:Requested mass definition 'FoF")
@pytest.mark.parametrize("bias_model", sorted(PAIRED_HMF_BEFORE_275))
def test_constructor_pairing_unchanged(bias_model):
    """The constructor must pick the same hmf_model as before #275 was fixed."""
    hm = TracerHaloModel(transfer_model="EH", bias_model=bias_model)
    assert hm.hmf_model.__name__ == PAIRED_HMF_BEFORE_275[bias_model]


@pytest.mark.filterwarnings("ignore:You are using an un-normalized mass function")
@pytest.mark.parametrize("bias_model", ["ST99", "Mo96", "Tinker10"])
def test_paired_constructor_equals_explicit_constructor(bias_model):
    """A paired hmf_model gives exactly the model with that hmf_model set explicitly."""
    paired = TracerHaloModel(transfer_model="EH", bias_model=bias_model)
    explicit = TracerHaloModel(
        transfer_model="EH",
        bias_model=bias_model,
        hmf_model=PAIRED_HMF_BEFORE_275[bias_model],
    )
    _assert_same_model(paired, explicit, rtol=1e-12)


def test_update_bias_model_gives_consistent_pbs_pair():
    """After updating to a peak-background-split bias, the HMF/bias pair is consistent.

    Before #275 was fixed, the HMF stayed at Tinker10, which is not a pair of ST99, so
    the matter power spectrum warned that the pair is not normalized.
    """
    hm = DMHaloModel(transfer_model="EH")
    hm.update(bias_model="ST99")
    assert hm.hmf_model in hm.bias_model.pair_hmf

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        assert np.all(np.isfinite(hm.power_auto_matter))


@pytest.mark.filterwarnings(r"ignore:You are using an un-normalized mass function \(Tinker08\)")
def test_explicit_hmf_model_kept_on_bias_update():
    """An explicitly set hmf_model is not changed by updating bias_model."""
    hm = TracerHaloModel(transfer_model="EH", hmf_model="Tinker08")
    dndm = hm.dndm

    hm.update(bias_model="ST99")
    assert hm.hmf_model.__name__ == "Tinker08"
    # The mass function does not depend on the bias, so it is not recomputed.
    assert hm.dndm is dndm
    # Tinker08 is not a pair of ST99, and the existing warning still says so.
    with pytest.warns(UserWarning, match="un-normalized mass function and bias function pair"):
        _ = hm.power_auto_matter

    # The explicit choice also survives an update to the paired default bias.
    hm.update(bias_model="Tinker10PBSplit")
    assert hm.hmf_model.__name__ == "Tinker08"


@pytest.mark.filterwarnings("ignore:You are using an un-normalized mass function")
def test_explicit_hmf_model_reset_to_paired():
    """Setting hmf_model back to None pairs it with bias_model again."""
    hm = TracerHaloModel(transfer_model="EH", hmf_model="Tinker08", bias_model="ST99")
    hm.update(hmf_model=None)
    _assert_same_model(hm, TracerHaloModel(transfer_model="EH", bias_model="ST99"), rtol=1e-10)

    # ...and it follows bias_model from then on.
    hm.update(bias_model="Mo96")
    _assert_same_model(hm, TracerHaloModel(transfer_model="EH", bias_model="Mo96"), rtol=1e-10)


def test_paired_hmf_model_dependency_tracking():
    """dndm is recomputed on a bias update only if hmf_model is paired with the bias."""
    hm = DMHaloModel(transfer_model="EH", bias_model="ST99")
    dndm = hm.dndm

    # Same paired hmf (SMT): the mass function is not recomputed.
    hm.update(bias_model="SMT01")
    assert hm.dndm is dndm

    # A different paired hmf (PS): it is.
    hm.update(bias_model="Mo96")
    assert hm.hmf_model.__name__ == "PS"
    assert hm.dndm is not dndm


@pytest.mark.filterwarnings("ignore:You are using an un-normalized mass function")
@pytest.mark.parametrize("hmf_model", [None, "Tinker08"])
@pytest.mark.parametrize("copier", ["clone", "pickle"])
def test_copy_preserves_hmf_pairing(hmf_model, copier):
    """Copies keep whether hmf_model is paired with bias_model or set explicitly."""
    import pickle

    hm = TracerHaloModel(transfer_model="EH", bias_model="ST99", hmf_model=hmf_model)
    _ = hm.dndm
    copied = hm.clone() if copier == "clone" else pickle.loads(pickle.dumps(hm))
    assert copied.hmf_model is hm.hmf_model

    copied.update(bias_model="Mo96")
    expected = "PS" if hmf_model is None else "Tinker08"
    assert copied.hmf_model.__name__ == expected
    # The original is not affected.
    assert hm.hmf_model.__name__ == ("SMT" if hmf_model is None else "Tinker08")


@pytest.mark.filterwarnings("ignore:You are using an un-normalized mass function")
def test_failed_update_keeps_hmf_pairing():
    """A rolled-back update must not turn a paired hmf_model into an explicit one."""
    hm = TracerHaloModel(transfer_model="EH", bias_model="ST99")
    with pytest.raises(AssertionError, match="hm_logk_min >= hm_logk_max"):
        hm.update(hmf_model="Tinker08", hm_logk_min=3.0, hm_logk_max=1.0)

    if hm.hmf_model.__name__ != "SMT":
        pytest.skip("This version of hmf does not roll back a failed update.")

    hm.update(bias_model="Mo96")
    assert hm.hmf_model.__name__ == "PS"


_COLOSSUS_NU_WARNING = "Astropy cosmology class contains massive neutrinos"


def _build_colossus_cosmo(hm: DMHaloModel):
    """Build ``hm.colossus_cosmo``, turning any DeprecationWarning into an error."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        # COLOSSUS ignores massive neutrinos and says so; that is expected.
        warnings.filterwarnings("ignore", message=_COLOSSUS_NU_WARNING, category=UserWarning)
        return hm.colossus_cosmo


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {
            "cosmo_params": {"H0": 70.0, "Om0": 0.28, "Ob0": 0.045, "m_nu": 0.0},
            "sigma_8": 0.78,
            "n": 0.95,
        },
    ],
)
def test_colossus_cosmo_matches_model(kwargs):
    """The COLOSSUS cosmology is built without deprecated hmf helpers and matches."""
    hm = DMHaloModel(transfer_model="EH", **kwargs)
    cc = _build_colossus_cosmo(hm)

    np.testing.assert_allclose(cc.sigma8, hm.sigma_8, rtol=1e-10)
    np.testing.assert_allclose(cc.ns, hm.n, rtol=1e-10)
    np.testing.assert_allclose(cc.H0, hm.cosmo.H0.value, rtol=1e-10)
    np.testing.assert_allclose(cc.Om0, hm.cosmo.Om0, rtol=1e-10)
    np.testing.assert_allclose(cc.Ob0, hm.cosmo.Ob0, rtol=1e-10)

    if kwargs:
        np.testing.assert_allclose(cc.sigma8, 0.78, rtol=1e-10)
        np.testing.assert_allclose(cc.ns, 0.95, rtol=1e-10)
        np.testing.assert_allclose(cc.H0, 70.0, rtol=1e-10)
        np.testing.assert_allclose(cc.Om0, 0.28, rtol=1e-10)
        np.testing.assert_allclose(cc.Ob0, 0.045, rtol=1e-10)


def test_colossus_cosmo_growth_factor_agrees_with_hmf():
    """COLOSSUS and hmf compute the same linear growth for the same cosmology.

    Massive neutrinos are switched off, since COLOSSUS does not model them.
    """
    hm = DMHaloModel(transfer_model="EH", z=1.0, cosmo_params={"m_nu": 0.0})
    assert not hm.cosmo.has_massive_nu

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cc = hm.colossus_cosmo
        d_colossus = cc.growthFactor(1.0)

    np.testing.assert_allclose(d_colossus, hm.growth_factor, rtol=5e-3)
    # Sanity: growth at z=1 is suppressed relative to z=0, but not absurdly.
    assert 0.5 < d_colossus < 1.0
