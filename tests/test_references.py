"""Tests of the references of halomod components and of ``get_acknowledgments``."""

import re

import pytest

from halomod import TracerHaloModel, _references
from halomod.bias import Bias, ScaleDepBias
from halomod.concentration import CMRelation
from halomod.cross_correlations import ConstantCorr, CrossCorrelations, _HODCross
from halomod.halo_exclusion import Exclusion
from halomod.hod import HOD
from halomod.profiles import Profile

COMPONENTS = (Bias, ScaleDepBias, CMRelation, Profile, HOD, Exclusion, _HODCross)

# Models with nothing to cite: toy models, or models whose docstring gives no
# reference. Keep this list short, and remove a model once it gets a reference.
NO_REFERENCES = {
    ("Bias", "UnityBias"),
    ("Profile", "Constant"),
    ("HOD", "ContinuousPowerLaw"),
    ("HOD", "Constant"),
    ("Exclusion", "NoExclusion"),
    ("Exclusion", "Sphere"),
    ("Exclusion", "DblSphere"),
    ("Exclusion", "DblSphere_"),
    ("_HODCross", "ConstantCorr"),
}

# Classes created on the fly by factory functions (e.g. make_colossus_cm), which are
# registered if other tests have called the factories.
FACTORY_MODELS = {"CustomColossusCM", "CustomColossusBias", "InterpConc"}


def _halomod_models():
    for component in COMPONENTS:
        for name, model in component.get_models().items():
            if model.__module__.startswith("halomod.") and name not in FACTORY_MODELS:
                yield component.__name__, name, model


ALL_MODELS = list(_halomod_models())


@pytest.mark.parametrize(
    ("component", "name", "model"), ALL_MODELS, ids=[f"{c}.{n}" for c, n, _ in ALL_MODELS]
)
def test_model_references(component, name, model):
    refs = getattr(model, "references", ())
    assert isinstance(refs, tuple)
    if (component, name) in NO_REFERENCES:
        assert refs == (), f"{name} now has references; remove it from NO_REFERENCES"
        return

    assert refs, f"{component} model {name} has no references"
    for ref in refs:
        assert isinstance(ref, str)
        # "Authors, YEAR. ..." like hmf's references.
        assert re.search(r", (19|20)\d\d\. ", ref), ref


def test_component_counts():
    """Every halomod component registry is covered by the tests above."""
    counts = {}
    for component, _, _ in ALL_MODELS:
        counts[component] = counts.get(component, 0) + 1
    assert counts["Bias"] == 13
    assert counts["CMRelation"] == 9
    assert counts["Profile"] == 12
    assert counts["HOD"] == 11
    assert counts["ScaleDepBias"] >= 1
    assert counts["Exclusion"] >= 5
    assert counts["_HODCross"] >= 1


def test_shared_references_are_reused():
    """Models citing the same paper use the same string, so it is deduplicated."""
    from halomod import bias, halo_exclusion, hod

    assert bias.Tinker05.references == bias.TinkerSD05.references == (_references.TINKER05,)
    assert hod.Tinker05.references == halo_exclusion.DblEllipsoid.references
    # Subclasses inherit their parent's references.
    assert halo_exclusion.NgMatched.references == halo_exclusion.DblEllipsoid.references


def test_acknowledgments_tracer_halo_model():
    hm = TracerHaloModel(transfer_model="EH")
    refs = hm.get_acknowledgments()

    keys = list(refs)
    assert keys[:2] == ["hmf", "halomod"]
    assert refs["halomod"] == (_references.HALOMOD,)
    assert "arXiv:2009.14066" in refs["halomod"][0]

    for key in (
        "bias_model",
        "halo_concentration_model",
        "halo_profile_model",
        "hod_model",
        "exclusion_model",
    ):
        assert key in refs
    for key in ("bias_model", "halo_concentration_model", "halo_profile_model", "hod_model"):
        assert refs[key], key

    assert refs["bias_model"] == hm.bias_model.references
    assert refs["hod_model"] == hm.hod_model.references


def test_acknowledgments_tracer_models():
    """Optional halomod-specific models are included when they are set."""
    hm = TracerHaloModel(
        transfer_model="EH",
        tracer_profile_model="Einasto",
        tracer_concentration_model="Duffy08",
        sd_bias_model="TinkerSD05",
        exclusion_model="DblEllipsoid",
    )
    refs = hm.get_acknowledgments()
    assert refs["tracer_profile_model"] == (_references.EINASTO65,)
    assert refs["tracer_concentration_model"] == (_references.DUFFY08,)
    assert refs["sd_bias_model"] == (_references.TINKER05,)
    assert refs["exclusion_model"] == (_references.TINKER05,)


def test_acknowledgments_flat():
    hm = TracerHaloModel(
        transfer_model="EH", sd_bias_model="TinkerSD05", exclusion_model="DblEllipsoid"
    )
    flat = hm.get_acknowledgments(flat=True)
    assert flat[1] == _references.HALOMOD
    assert len(flat) == len(set(flat))
    assert flat.count(_references.TINKER05) == 1


def test_acknowledgments_cross_correlations():
    cross = CrossCorrelations(
        cross_hod_model=ConstantCorr,
        halo_model_1_params={"transfer_model": "EH"},
        halo_model_2_params={"transfer_model": "EH", "hod_model": "Zheng05"},
    )
    refs = cross.get_acknowledgments()
    assert list(refs)[:2] == ["hmf", "halomod"]
    assert refs["halo_model_2.hod_model"] == (_references.ZHENG05,)
    assert _references.HALOMOD in cross.get_acknowledgments(flat=True)


def test_with_halomod_reference_ordering():
    refs = _references.with_halomod_reference({"a": ("x",), "hmf": ("y",)})
    assert list(refs) == ["hmf", "halomod", "a"]

    refs = _references.with_halomod_reference({"a": ("x",)})
    assert list(refs) == ["halomod", "a"]
