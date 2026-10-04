"""
This module performs some regression tests.

The idea is to produce regression data with the `make_regression_data.py` script,
then produce *exactly the same* data here, and compare.

These tests should be run with an absolutely pinned set of dependencies so that updates
underneath don't break things. Every now and then, one can update the list of
dependencies, and reproduce the data.
"""

from itertools import product

import numpy as np
import pytest
import yaml
from make_regression_data import (
    base_options,
    compress_data,
    datadir,
    get_hash,
    quantities,
    tested_params,
)

from halomod import TracerHaloModel


@pytest.fixture(scope="module")
def tr():
    return TracerHaloModel(**base_options)


test_matrix = product(tested_params, quantities)
test_matrix = [(t[0][0], t[0][1], t[1]) for t in test_matrix]


# Warnings that are expected when computing the (deliberately unusual) regression models.
_expected_warnings = [
    pytest.mark.filterwarnings("ignore:Requested mass definition"),
    pytest.mark.filterwarnings("ignore:You are using an un-normalized mass function"),
    pytest.mark.filterwarnings("ignore:Using halofit for tracer stats"),
    pytest.mark.filterwarnings("ignore:to use a power-law, y must be all positive"),
]


def _ignore_expected_warnings(func):
    for mark in _expected_warnings:
        func = mark(func)
    return func


@_ignore_expected_warnings
@pytest.mark.parametrize(("z", "params", "quantity"), test_matrix)
def test_regression_quantity_tracerhm(tr, z, params, quantity):
    # Since this is a regression test, we don't care that the mass definition
    # doesn't match the Ludlow16 relation.
    print("Testing for params = ", params)
    tr.update(z=z, **params)

    hsh = get_hash(z, params)
    param_dir = datadir / hsh
    this = getattr(tr, quantity)

    if this is None:
        pytest.skip("This quantity doesn't exist for the input parameters.")

    this = compress_data(this)
    data = np.load(param_dir / (quantity + ".npy"))
    assert data.shape == np.shape(this)
    assert np.allclose(data, this, rtol=1e-3, atol=0)


def test_compress_data_keeps_every_fifth_value():
    """compress_data must thin arrays, not empty them (see #266)."""
    np.testing.assert_array_equal(compress_data(np.arange(23)), [0, 5, 10, 15, 20])
    np.testing.assert_array_equal(compress_data(np.arange(5)), [0])

    arr = np.arange(12 * 7).reshape(12, 7)
    out = compress_data(arr)
    np.testing.assert_array_equal(out, arr[::5, ::5])
    assert out.shape == (3, 2)

    assert compress_data(3.5) == 3.5


@_ignore_expected_warnings
@pytest.mark.parametrize(("z", "params"), tested_params)
def test_regression_snapshots_complete_and_nonempty(tr, z, params):
    """Every stored snapshot must hold data, and none may be missing or stale.

    Guards against the stored data silently degenerating (e.g. to empty arrays,
    for which ``np.allclose`` is trivially ``True``).
    """
    param_dir = datadir / get_hash(z, params)
    assert param_dir.is_dir()

    with (param_dir / "params.yaml").open() as fl:
        assert yaml.safe_load(fl) == {**params, "z": z}

    stored = {f.stem for f in param_dir.glob("*.npy")}
    assert stored <= set(quantities), f"stale snapshots: {stored - set(quantities)}"

    for name in stored:
        data = np.load(param_dir / f"{name}.npy")
        if data.ndim > 0:
            assert data.size > 0, f"{name} snapshot is empty"
        assert np.all(np.isfinite(data)), f"{name} snapshot has non-finite values"

    missing = set(quantities) - stored
    if missing:
        tr.update(z=z, **params)
        for name in missing:
            assert getattr(tr, name) is None, f"{name} has no stored snapshot"


def test_no_stale_regression_directories():
    """Every directory of regression data corresponds to a tested parameter set."""
    expected = {get_hash(z, params) for z, params in tested_params}
    present = {d.name for d in datadir.iterdir() if d.is_dir()}
    assert present == expected
