import numpy as np
from halomod.tools import (
    power_to_corr,
    power_to_corr_ogata,
    populate,
    ExtendedSpline,
    hankel_transform,
)
from halomod.profiles import NFW
from halomod.hod import Tinker05, Zehavi05
from halomod.concentration import Bullock01Power
import pytest


def test_ogata_powerlaw():
    k = np.logspace(-4, 2, 100)
    power = k ** -1.5

    r = np.logspace(-1, 1, 5)

    corr_ogata = power_to_corr_ogata(power, k, r)
    corr_simple = power_to_corr(lambda x: np.exp(x) ** -1.5, r)

    assert np.allclose(corr_ogata, corr_simple, rtol=1e-4)


def test_ogata_powerlaw_fnc():
    r = np.logspace(-1, 1, 5)

    corr_ogata = hankel_transform(lambda k: k ** -1.5, r, "r")
    corr_simple = power_to_corr(lambda x: np.exp(x) ** -1.5, r)

    assert np.allclose(corr_ogata, corr_simple, rtol=1e-4)


def test_ogata_powerlaw_trunc():
    "Test that power_to_corr still works on a truncated spectrum"
    k = np.logspace(-1, 1, 100)
    power = k ** -1.5

    r = np.logspace(-1, 1, 5)

    with pytest.warns(UserWarning):
        corr_ogata = power_to_corr_ogata(power, k, r)

    corr_simple = power_to_corr(lambda x: np.exp(x) ** -1.5, r)

    assert np.allclose(corr_ogata, corr_simple, rtol=1e-3)


def test_ogata_powerlaw_upper_trunc():
    "Test that power_to_corr still works on a truncated spectrum"
    k = np.logspace(-1, 2, 1000)
    power = np.where(k < 10, k ** -1.5, 0)

    r = np.logspace(-1, 1, 5)

    with pytest.warns(UserWarning):
        corr_ogata = power_to_corr_ogata(power, k, r, power_pos=(True, False))

    corr_simple = power_to_corr(
        lambda x: np.where(np.exp(x) < 10, np.exp(x) ** -1.5, 0), r
    )

    assert np.allclose(corr_ogata, corr_simple, rtol=1e-1)


def test_ogata_powerlaw_matrix():
    k = np.logspace(-4, 2, 100)
    power = np.array([i * k ** -1.5 for i in range(1, 6)])
    print(power.shape)
    r = np.logspace(-1, 1, 5)

    corr_ogata = power_to_corr_ogata(power, k, r)
    corr_simple = power_to_corr(lambda x: np.exp(x) ** -1.5, r)

    for i, corr in enumerate(corr_ogata):
        assert np.allclose(corr, corr_simple[i] * (i + 1), rtol=1e-3)


def test_populate_runs_with_central_cond():
    np.random.seed(1234)

    nhalos = 100
    boxsize = 100
    centres = boxsize * np.random.random((nhalos, 3))  # centres between 0 - 100 Mpc
    masses = 10 ** (10 + 5 * np.random.random(nhalos))  # masses between 1e10 and 1e15
    hod = Tinker05(central=True)
    profile = NFW(Bullock01Power(ms=1e12))

    pos, halo, ncen = populate(centres, masses, profile=profile, hodmod=hod)

    assert pos.min() >= -5  # Could be in a big halo on the edge.
    assert pos.max() <= boxsize + 5  # Could be in a big halo on the edge.

    assert halo.min() >= 0
    assert halo.max() <= (nhalos - 1)

    assert ncen <= nhalos
    assert ncen <= len(pos)

    # Test that the first ncen galaxies are at halo centres.
    for i in range(ncen):
        assert np.allclose(pos[i], centres[halo[i]])

    # Test the central condition
    for h in halo[ncen:]:
        assert h in halo[:ncen]


def test_populate_runs_without_central_cond():
    np.random.seed(1234)

    nhalos = 100
    boxsize = 100
    centres = boxsize * np.random.random((nhalos, 3))  # centres between 0 - 100 Mpc
    masses = 10 ** (10 + 5 * np.random.random(nhalos))  # masses between 1e10 and 1e15
    hod = Zehavi05(central=False)
    profile = NFW(Bullock01Power(ms=1e12))

    pos, halo, ncen = populate(centres, masses, profile=profile, hodmod=hod)

    assert pos.min() >= -5  # Could be in a big halo on the edge.
    assert pos.max() <= boxsize + 5  # Could be in a big halo on the edge.

    assert halo.min() >= 0
    assert halo.max() <= (nhalos - 1)

    assert ncen <= nhalos
    assert ncen <= len(pos)

    # Test that the first ncen galaxies are at halo centres.
    for i in range(ncen):
        assert np.allclose(pos[i], centres[halo[i]])

    # Test that the central condition is False
    assert not all(h in halo[:ncen] for h in halo[ncen:])


@pytest.fixture(scope="module")
def xy():
    """Simple power-law vector"""
    x = np.logspace(0, 1, 100)
    y = x ** -2
    return x, y


def test_extended_spline_pl_explicit_func(xy):

    es = ExtendedSpline(
        *xy,
        lower_func=lambda xx: xx ** -2,
        upper_func=lambda xx: xx ** -2,
        match_lower=False,
        match_upper=False
    )

    assert np.isclose(es(0.1), 100.0)
    assert np.isclose(es(100.0), 0.0001)
    assert np.isclose(es(1.0), 1)
    assert np.isclose(es(5.0), 1 / 25.0)


def test_extended_spline_pl_power_law(xy):
    es = ExtendedSpline(*xy, lower_func="power_law", upper_func="power_law")

    assert es.lfunc(0.1) == es(0.1)
    assert es.ufunc(100) == es(100)
    assert np.isclose(es(0.1), 100.0)
    assert np.isclose(es(100.0), 0.0001)
    assert np.isclose(es(1.0), 1)
    assert np.isclose(es(5.0), 1 / 25.0)


def test_extended_spline_pl_match(xy):
    es = ExtendedSpline(
        *xy, lower_func=lambda xx: 3 * xx ** -2, upper_func=lambda xx: 3 * xx ** -2
    )

    assert np.isclose(es(0.1), 100.0)
    assert np.isclose(es(100.0), 0.0001)
    assert np.isclose(es(1.0), 1)
    assert np.isclose(es(5.0), 1 / 25.0)


def test_extended_spline_pl_pure(xy):
    es = ExtendedSpline(*xy)

    assert np.isclose(es(0.1), es._spl(0.1))
    assert np.isclose(es(100.0), es._spl(100))
    assert np.isclose(es(1.0), 1)
    assert np.isclose(es(5.0), 1 / 25.0)


def test_extended_spline_pl_noise(xy):
    x, y = xy
    y += np.random.normal(scale=y / 1000)

    es = ExtendedSpline(x, y, lower_func="power_law", upper_func="power_law")

    assert np.isclose(es(0.1), 100.0, rtol=1e-1)
    assert np.isclose(es(100.0), 0.0001, rtol=1e-1)
    assert np.isclose(es(1.0), 1, rtol=1e-2)
    assert np.isclose(es(5.0), 1 / 25.0, rtol=1e-2)
