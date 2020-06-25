"""
This module performs some regression tests.

The idea is to produce regression data with the `make_regression_data.py` script,
then produce _exactly the same_ data here, and compare.

These tests should be run with an absolutely pinned set of dependencies so that updates
underneath don't break things. Every now and then, one can update the list of
dependencies, and reproduce the data.
"""

from make_regression_data import (
    tested_params,
    quantities,
    base_options,
    datadir,
    get_hash,
    compress_data,
)
import pytest
from halomod import TracerHaloModel
from itertools import product
import numpy as np


@pytest.fixture(scope="module")
def tr():
    return TracerHaloModel(**base_options)


test_matrix = product(tested_params, quantities)
test_matrix = [(t[0][0], t[0][1], t[1]) for t in test_matrix]


@pytest.mark.parametrize("z,params,quantity", test_matrix)
def test_regression_quantity_tracerhm(tr, z, params, quantity):
    tr.update(z=z, **params)

    hsh = get_hash(z, params)
    param_dir = datadir / hsh
    this = getattr(tr, quantity)

    if this is None:
        pytest.skip("This quantity doesn't exist for the input parameters.")

    this = compress_data(this)
    data = np.load(param_dir / (quantity + ".npy"))
    assert np.allclose(data, this, rtol=1e-2)
