"""W&B-backed data fixtures; allow local CSVs for explicit local verification."""
from pathlib import Path

import pandas as pd
import pytest
import wandb


def pytest_addoption(parser):
    """Register typed thresholds and artifact references."""
    for name in ["csv", "ref", "kl_threshold", "min_price", "max_price",
                 "min_longitude", "max_longitude", "min_latitude", "max_latitude"]:
        parser.addoption(f"--{name}", action="store")


@pytest.fixture(scope="session")
def artifact_run(request):
    """Use one W&B run for both current and reference input lineage."""
    refs = [request.config.getoption("csv"), request.config.getoption("ref")]
    if all(value and Path(value).is_file() for value in refs):
        yield None
        return
    with wandb.init(job_type="data_tests") as run:
        run.config.update({"csv": refs[0], "ref": refs[1]})
        yield run


def read_data(request, option, run):
    """Read a local verification file or record the W&B artifact dependency."""
    value = request.config.getoption(option)
    if value is None:
        pytest.fail(f"Missing --{option}")
    path = value if Path(value).is_file() else run.use_artifact(value).file()
    return pd.read_csv(path)


@pytest.fixture(scope="session")
def data(request, artifact_run):
    """Load the current sample."""
    return read_data(request, "csv", artifact_run)


@pytest.fixture(scope="session")
def ref_data(request, artifact_run):
    """Load the fixed reference sample."""
    return read_data(request, "ref", artifact_run)


def threshold(request, name):
    """Require a numeric configured threshold."""
    value = request.config.getoption(name)
    if value is None:
        pytest.fail(f"Missing --{name}")
    return float(value)


@pytest.fixture(scope="session")
def kl_threshold(request):
    return threshold(request, "kl_threshold")


@pytest.fixture(scope="session")
def min_price(request):
    return threshold(request, "min_price")


@pytest.fixture(scope="session")
def max_price(request):
    return threshold(request, "max_price")


@pytest.fixture(scope="session")
def min_longitude(request):
    return threshold(request, "min_longitude")


@pytest.fixture(scope="session")
def max_longitude(request):
    return threshold(request, "max_longitude")


@pytest.fixture(scope="session")
def min_latitude(request):
    return threshold(request, "min_latitude")


@pytest.fixture(scope="session")
def max_latitude(request):
    return threshold(request, "max_latitude")
