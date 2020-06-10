import numpy as np
from halomod.tools import power_to_corr, power_to_corr_ogata, populate
from halomod.profiles import NFW
from halomod.hod import Tinker05, Zehavi05
from halomod.concentration import Bullock01Power


def test_ogata_powerlaw():
    k = np.logspace(-4, 2, 1000)
    power = k ** -1.5

    r = np.logspace(-1, 1, 5)

    corr_ogata = power_to_corr_ogata(power, k, r)
    corr_simple = power_to_corr(lambda x: np.exp(x) ** -1.5, r)

    assert np.allclose(corr_ogata, corr_simple, rtol=1e-4)


def test_ogata_powerlaw_matrix():
    k = np.logspace(-4, 2, 1000)
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
