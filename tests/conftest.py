import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def datadir() -> Path:
    """The directory in which the test data resides."""
    return Path(__file__).parent / "data"
