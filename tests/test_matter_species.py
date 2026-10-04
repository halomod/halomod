"""Tests of the matter field (``matter_species``) that halomod asks CAMB for.

With massive neutrinos, CAMB can return the CDM+baryon field (``"cb"``) or the total
matter field (``"tot"``). halomod uses ``"cb"`` unless ``transfer_params`` says
otherwise (see #269).
"""

import warnings

import numpy as np
import pytest
from hmf import MassFunction

from halomod import DMHaloModel
from halomod.halo_model import DEFAULT_MATTER_SPECIES

pytest.importorskip("camb")

# Unrelated hmf warning for CAMB, silenced by setting the parameter explicitly.
CAMB_PARAMS = {"extrapolate_with_eh": True}


def _matter_species_warnings(record: list[warnings.WarningMessage]) -> list[str]:
    """Messages of the recorded warnings that are about ``matter_species``."""
    return [str(w.message) for w in record if "matter_species" in str(w.message)]


@pytest.fixture(scope="module")
def hm_cb() -> DMHaloModel:
    """A default (CAMB, Planck18) halo model, which uses the CDM+baryon field."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        hm = DMHaloModel(transfer_params=CAMB_PARAMS)
        _ = hm.power
    hm._matter_species_warnings = _matter_species_warnings(record)
    return hm


@pytest.fixture(scope="module")
def hm_tot() -> DMHaloModel:
    """A halo model using the total matter field (including massive neutrinos)."""
    hm = DMHaloModel(transfer_params={**CAMB_PARAMS, "matter_species": "tot"})
    _ = hm.power
    return hm


def test_default_is_cb_without_warning(hm_cb: DMHaloModel):
    assert DEFAULT_MATTER_SPECIES == "cb"
    assert hm_cb.cosmo.has_massive_nu
    assert hm_cb.transfer.params["matter_species"] == "cb"
    assert hm_cb._matter_species_warnings == []
    # The user's parameters themselves are left as given.
    assert "matter_species" not in hm_cb.transfer_params


def test_explicit_choice_is_kept(hm_tot: DMHaloModel):
    assert hm_tot.transfer.params["matter_species"] == "tot"


@pytest.mark.parametrize("species", ["cb", "tot"])
def test_power_matches_hmf(species: str, hm_cb: DMHaloModel, hm_tot: DMHaloModel):
    """halomod passes the chosen field through to hmf unchanged."""
    hm = {"cb": hm_cb, "tot": hm_tot}[species]
    mf = MassFunction(transfer_params={**CAMB_PARAMS, "matter_species": species})
    np.testing.assert_allclose(hm.k, mf.k, rtol=1e-12)
    np.testing.assert_allclose(hm.power, mf.power, rtol=1e-6, atol=0)


def test_cb_power_exceeds_total(hm_cb: DMHaloModel, hm_tot: DMHaloModel):
    """Massive neutrinos don't cluster below their free-streaming scale.

    With sigma_8 normalising the total field in both models, the two share the same
    amplitude, so on large scales (where neutrinos cluster like CDM) their power is
    the same, while far below the free-streaming scale delta_nu ~ 0, so that
    delta_tot = (1 - f_nu) delta_cb and P_cb / P_tot -> (1 - f_nu)^-2.
    """
    assert hm_cb.sigma_8 == hm_tot.sigma_8

    def ratio(k: float) -> float:
        lnk = np.log(k)
        return np.interp(lnk, np.log(hm_cb.k), hm_cb.power) / np.interp(
            lnk, np.log(hm_tot.k), hm_tot.power
        )

    # Large scales: identical fields.
    assert abs(ratio(1e-4) - 1) < 1e-5

    # Small scales: cb power is larger by ~1%.
    excess = ratio(1.0) - 1
    assert 0.005 < excess < 0.015

    # ... and matches the analytic small-scale limit. Omega_nu h^2 = sum(m_nu) / 93.14 eV
    # counts only the massive (non-relativistic today) neutrinos.
    cosmo = hm_cb.cosmo
    omega_nu = cosmo.m_nu.value.sum() / 93.14 / cosmo.h**2
    f_nu = omega_nu / (cosmo.Om0 + omega_nu)
    np.testing.assert_allclose(excess, (1 - f_nu) ** -2 - 1, rtol=0.02)
    # It has reached that limit by k = 1 h/Mpc.
    np.testing.assert_allclose(ratio(5.0) - 1, excess, rtol=1e-3)


def test_eh_gets_no_matter_species():
    """Transfer models without a matter_species parameter are not given one."""
    hm = DMHaloModel(transfer_model="EH")
    assert "matter_species" not in hm.transfer.params
    mf = MassFunction(transfer_model="EH")
    np.testing.assert_allclose(hm.power, mf.power, rtol=1e-10, atol=0)


def test_update_to_camb_applies_default():
    hm = DMHaloModel(transfer_model="EH")
    _ = hm.power
    assert "matter_species" not in hm.transfer.params

    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        hm.update(transfer_model="CAMB", transfer_params={"kmax": 10.0, **CAMB_PARAMS})
        assert hm.transfer.params["matter_species"] == "cb"
        assert hm.transfer.params["kmax"] == 10.0
        _ = hm.power
    assert _matter_species_warnings(record) == []

    # And back to EH, which must not receive the CAMB-only parameter.
    hm.update(transfer_model="EH", transfer_params={})
    assert "matter_species" not in hm.transfer.params
    assert np.all(hm.power > 0)
