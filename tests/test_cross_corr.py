import sys

import hmf
import numpy as np
import pytest

from halomod.cross_correlations import ConstantCorr, CrossCorrelations


def test_cross_same():
    """Test if using two components that are the same gives the same as an auto corr."""
    cross = CrossCorrelations(
        cross_hod_model=ConstantCorr,
        halo_model_1_params={
            "exclusion_model": "NoExclusion",
            "sd_bias_model": None,
            "transfer_model": "EH",
            "force_1halo_turnover": False,
        },
        halo_model_2_params={
            "exclusion_model": "NoExclusion",
            "sd_bias_model": None,
            "transfer_model": "EH",
            "force_1halo_turnover": False,
        },
    )

    assert np.allclose(cross.power_2h_cross, cross.halo_model_1.power_2h_auto_tracer)
    assert np.allclose(cross.corr_2h_cross, cross.halo_model_1.corr_2h_auto_tracer)

    # This is only close-ish, because cross-pairs are actually different than auto-pairs,
    # since you can count self-correlations.
    assert np.allclose(
        cross.corr_1h_cross,
        cross.halo_model_1.corr_1h_auto_tracer,
        atol=1e-5,
        rtol=1e-1,
    )

    assert np.allclose(
        cross.power_1h_cross,
        cross.halo_model_1.power_1h_auto_tracer,
        atol=1e-6,
        rtol=1e-1,
    )


CROSS_OUTPUTS = ["power_cross", "power_1h_cross", "power_2h_cross", "corr_cross"]

FAST_KW = {"transfer_model": "EH", "hm_logk_min": -2, "hm_logk_max": 1, "hm_dlog10k": 0.05}


def test_cross_outputs_in_quantities_available():
    """The cross-correlation outputs must be discoverable via quantities_available()."""
    available = set(CrossCorrelations.quantities_available())
    missing = [name for name in CROSS_OUTPUTS if name not in available]
    assert not missing


def test_cross_hod_model_default():
    """CrossCorrelations constructs without arguments, defaulting to ConstantCorr."""
    cross = CrossCorrelations()
    assert cross.cross_hod_model is ConstantCorr


def test_cross_get_all_parameter_defaults():
    defaults = CrossCorrelations.get_all_parameter_defaults()
    assert isinstance(defaults, dict)
    assert defaults["cross_hod_model"] is ConstantCorr


# Every cached quantity of CrossCorrelations, intermediates first.
CROSS_QUANTITIES = [
    "cross_hod",
    "power_1h_cross_fnc",
    "power_2h_cross_fnc",
    "corr_1h_cross_fnc",
    "corr_2h_cross_fnc",
    "power_1h_cross",
    "power_2h_cross",
    "corr_1h_cross",
    "corr_2h_cross",
    "power_cross",
    "corr_cross",
]


@pytest.mark.filterwarnings("ignore:You are using an un-normalized mass function")
@pytest.mark.parametrize("via", ["cross", "sub"])
@pytest.mark.parametrize("order", [1, -1], ids=["leaves_first", "outputs_first"])
def test_cross_outputs_invalidate_on_subframework_update(via, order):
    """Cross outputs recomputed after a halo-model update match those of a fresh model.

    This must hold whether the halo model is updated through ``CrossCorrelations.update``
    or directly, and whichever cached intermediates were computed first.
    """
    cross = CrossCorrelations(halo_model_1_params=FAST_KW, halo_model_2_params=FAST_KW)
    for name in CROSS_QUANTITIES[::order]:
        getattr(cross, name)
    before = {name: getattr(cross, name).copy() for name in CROSS_OUTPUTS}

    if via == "cross":
        cross.update(halo_model_1_params={"z": 1.0})
    else:
        cross.halo_model_1.update(z=1.0)

    fresh = CrossCorrelations(
        halo_model_1_params={**FAST_KW, "z": 1.0}, halo_model_2_params=FAST_KW
    )
    for name in CROSS_OUTPUTS + ["corr_1h_cross", "corr_2h_cross"]:
        np.testing.assert_allclose(
            getattr(cross, name), getattr(fresh, name), rtol=1e-10, atol=0, err_msg=name
        )
    for name in CROSS_OUTPUTS:
        assert not np.allclose(getattr(cross, name), before[name], rtol=1e-6, atol=0)

    # A second update, now of the other halo model, is also picked up.
    cross.update(halo_model_2_params={"hod_params": {"M_min": 12.5}})
    fresh.update(halo_model_2_params={"hod_params": {"M_min": 12.5}})
    for name in CROSS_OUTPUTS:
        np.testing.assert_allclose(
            getattr(cross, name), getattr(fresh, name), rtol=1e-10, atol=0, err_msg=name
        )


# Values computed on the commit preceding the conversion of these outputs to cached
# quantities, at indices ``REF_INDICES``. They depend on the hmf version.
REF_INDICES = [0, 20, 40, 60, 80]
REF_VALUES = {
    "3.7.1": {
        "power_cross": [26538.933298118438, 7161.510135575586, 314.2841509389841],
        "corr_cross": [
            53997.115732177466,
            2122.087989595101,
            81.84345250263902,
            3.386187747154623,
            1.1172286021980025,
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
def test_cross_outputs_unchanged():
    """Converting outputs to cached quantities must not change their values."""
    cross = CrossCorrelations(
        cross_hod_model=ConstantCorr,
        halo_model_1_params=FAST_KW,
        halo_model_2_params={**FAST_KW, "z": 0.5},
    )
    for name, ref in REF_VALUES[hmf.__version__].items():
        value = getattr(cross, name)
        idx = [i for i in REF_INDICES if i < len(value)]
        np.testing.assert_allclose(value[idx], ref, rtol=REF_RTOL, atol=0, err_msg=name)
