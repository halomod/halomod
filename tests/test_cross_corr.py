from halomod.cross_correlations import CrossCorrelations, ConstantCorr
import numpy as np


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
