"""Integration-style tests of the full HaloModel class."""

import sys
import warnings

import hmf
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


# ---------------------------------------------------------------------------
# Main outputs are cached quantities (issue #267)
# ---------------------------------------------------------------------------
DM_OUTPUTS = [
    "power_auto_matter",
    "power_1h_auto_matter",
    "power_2h_auto_matter",
    "corr_auto_matter",
    "corr_1h_auto_matter",
    "corr_2h_auto_matter",
]

TRACER_OUTPUTS = [
    "power_auto_tracer",
    "power_1h_auto_tracer",
    "power_1h_ss_auto_tracer",
    "power_1h_cs_auto_tracer",
    "power_2h_auto_tracer",
    "corr_auto_tracer",
    "corr_1h_auto_tracer",
    "corr_1h_ss_auto_tracer",
    "corr_1h_cs_auto_tracer",
    "corr_2h_auto_tracer",
    "power_auto_tracer_fnc",
    "corr_auto_tracer_fnc",
    "power_cross_tracer_matter",
    "power_1h_cross_tracer_matter",
    "power_2h_cross_tracer_matter",
    "corr_cross_tracer_matter",
    "corr_1h_cross_tracer_matter",
    "corr_2h_cross_tracer_matter",
    "tracer_mmin",
]

# A small, fast model setup shared by the caching tests below.
FAST_KW = {
    "transfer_model": "EH",
    "hm_logk_min": -2,
    "hm_logk_max": 1,
    "hm_dlog10k": 0.05,
    "rnum": 100,
}


# hmf<3.7 instantiates the class (with its CAMB default) to list its quantities.
@pytest.mark.filterwarnings("ignore:'extrapolate_with_eh' was not set")
@pytest.mark.parametrize(
    ("model", "names"),
    [(DMHaloModel, DM_OUTPUTS), (TracerHaloModel, DM_OUTPUTS + TRACER_OUTPUTS)],
)
def test_main_outputs_in_quantities_available(model, names):
    """The documented outputs must be discoverable via quantities_available()."""
    available = set(model.quantities_available())
    missing = [name for name in names if name not in available]
    assert not missing


@pytest.mark.filterwarnings("ignore:You are using an un-normalized mass function")
@pytest.mark.parametrize(
    "update",
    [{"z": 1.0}, {"hod_params": {"M_min": 12.5}}],
    ids=["z", "M_min"],
)
def test_cached_outputs_invalidate_on_update(update):
    """Cached outputs recomputed after update() must equal those of a fresh model."""
    names = ["power_auto_tracer", "corr_auto_tracer", "power_auto_matter"]

    hm = TracerHaloModel(**FAST_KW)
    before = {name: getattr(hm, name).copy() for name in names}

    hm.update(**update)
    fresh = TracerHaloModel(**FAST_KW, **update)

    for name in names:
        updated = getattr(hm, name)
        np.testing.assert_allclose(updated, getattr(fresh, name), rtol=1e-10, atol=0)

        # The update must actually have changed the output (so that the comparison above
        # is a real test of invalidation), except that the matter power spectrum does
        # not depend on the HOD.
        if name == "power_auto_matter" and "hod_params" in update:
            np.testing.assert_allclose(updated, before[name], rtol=1e-12, atol=0)
        else:
            assert not np.allclose(updated, before[name], rtol=1e-6, atol=0)


@pytest.mark.filterwarnings("ignore:You are using an un-normalized mass function")
def test_successive_updates_match_fresh():
    """Chained updates (z, then HOD) still give the same outputs as a fresh model."""
    names = ["power_auto_tracer", "corr_auto_tracer", "power_auto_matter"]

    hm = TracerHaloModel(**FAST_KW)
    for name in names:
        getattr(hm, name)

    hm.update(z=1.0)
    for name in names:
        getattr(hm, name)
    hm.update(hod_params={"M_min": 12.5})

    fresh = TracerHaloModel(**FAST_KW, z=1.0, hod_params={"M_min": 12.5})
    for name in names:
        np.testing.assert_allclose(getattr(hm, name), getattr(fresh, name), rtol=1e-10, atol=0)


@pytest.mark.filterwarnings("ignore:You are using an un-normalized mass function")
def test_tracer_mmin_follows_hod():
    """tracer_mmin is 10**M_min for a sharp-cut central HOD, and tracks updates to it."""
    # Tinker05 has a sharp cut at M_min and inherently enforces the central condition.
    hm = TracerHaloModel(**FAST_KW, hod_model="Tinker05", hod_params={"M_min": 12.0})
    np.testing.assert_allclose(hm.tracer_mmin, 1e12, rtol=1e-12)

    hm.update(hod_params={"M_min": 12.5})
    np.testing.assert_allclose(hm.tracer_mmin, 10**12.5, rtol=1e-12)

    # Zheng05 has a smooth central occupation, so no lower mass limit is imposed.
    hm.update(hod_model="Zheng05")
    assert hm.tracer_mmin is None


@pytest.mark.filterwarnings("ignore:You are using an un-normalized mass function")
@pytest.mark.parametrize(
    ("name", "fnc", "grid"),
    [
        ("power_auto_matter", "power_auto_matter_fnc", "k_hm"),
        ("corr_auto_matter", "corr_auto_matter_fnc", "r"),
        ("power_auto_tracer", "power_auto_tracer_fnc", "k_hm"),
        ("corr_auto_tracer", "corr_auto_tracer_fnc", "r"),
        ("power_cross_tracer_matter", "power_cross_tracer_matter_fnc", "k_hm"),
        ("corr_cross_tracer_matter", "corr_cross_tracer_matter_fnc", "r"),
    ],
)
def test_cached_output_is_fnc_on_grid(name, fnc, grid):
    """Each array output equals its callable evaluated on the model's grid."""
    hm = TracerHaloModel(**FAST_KW)
    x = getattr(hm, grid)
    np.testing.assert_allclose(getattr(hm, name), getattr(hm, fnc)(x), rtol=1e-12, atol=0)


# Values computed with ``TracerHaloModel(**FAST_KW)`` on the commit preceding the
# conversion of these outputs to cached quantities, at indices ``REF_INDICES``. They
# depend on the hmf version, so are keyed by it.
REF_INDICES = [0, 20, 40, 60, 80]
REF_VALUES = {
    "3.6.0": {
        "power_auto_tracer": [23644.241167825094, 6180.715296805746, 283.45796741827473],
        "corr_auto_tracer": [
            58918.16127459432,
            2128.3029892436443,
            76.72178329069574,
            2.1420495164642377,
            0.10570816666995948,
        ],
        "power_auto_matter": [21729.240353958383, 5680.827166313392, 348.83757677778567],
        "corr_auto_matter": [
            3537.9161172834647,
            1033.0622525949736,
            101.03829706583996,
            2.0126554452660015,
            0.09715792840164839,
        ],
        "power_cross_tracer_matter": [
            19600.535934509215,
            5363.280278626038,
            310.79591275404334,
        ],
    },
    "3.7.1": {
        "power_auto_tracer": [23648.651803224664, 6181.837493342404, 283.26176497701977],
        "corr_auto_tracer": [
            58920.35858834676,
            2128.026396860861,
            76.67483469824671,
            2.142131134471782,
            0.10572785771509996,
        ],
        "power_auto_matter": [21729.2405230342, 5680.792435144981, 348.55014976171657],
        "corr_auto_matter": [
            3537.510012977628,
            1032.7149839436397,
            100.97006738318066,
            2.0122798527980956,
            0.09715789709544165,
        ],
        "power_cross_tracer_matter": [
            19601.657173717285,
            5363.293131430889,
            310.55510401207016,
        ],
    },
}
# Bit-level agreement is only expected on the platform the references were made on.
REF_RTOL = 1e-12 if sys.platform.startswith("linux") else 1e-6


@pytest.mark.filterwarnings("ignore:You are using an un-normalized mass function")
@pytest.mark.skipif(
    hmf.__version__ not in REF_VALUES,
    reason=f"No reference values for hmf {hmf.__version__}",
)
def test_cached_outputs_unchanged():
    """Converting outputs to cached quantities must not change their values."""
    hm = TracerHaloModel(**FAST_KW)
    for name, ref in REF_VALUES[hmf.__version__].items():
        value = getattr(hm, name)
        idx = [i for i in REF_INDICES if i < len(value)]
        np.testing.assert_allclose(value[idx], ref, rtol=REF_RTOL, atol=0, err_msg=name)
