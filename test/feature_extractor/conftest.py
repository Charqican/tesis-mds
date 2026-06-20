import pytest
import torch
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
FIXTURES_DIR = Path(os.environ["FIXTURES_DIR"])


@pytest.fixture
def point_cloud():
    return torch.load(FIXTURES_DIR / "point_cloud.pt")


@pytest.fixture
def camera_R():
    return torch.load(FIXTURES_DIR / "R.pt")


@pytest.fixture
def camera_T():
    return torch.load(FIXTURES_DIR / "T.pt")


@pytest.fixture
def rendered_images():
    return torch.load(FIXTURES_DIR / "rendered_images.pt")


@pytest.fixture
def mappings():
    return torch.load(FIXTURES_DIR / "mappings.pt")


@pytest.fixture
def reference_features():
    return torch.load(FIXTURES_DIR / "features.pt")


@pytest.fixture
def model_outputs():
    return torch.load(FIXTURES_DIR / "model_outputs.pt")
