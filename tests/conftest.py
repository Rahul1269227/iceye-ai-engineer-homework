import sys
from pathlib import Path

import pytest

ICEYE_ROOT = Path("/Users/rahuly@backbase.com/Documents/iceye")
sys.path.insert(0, str(ICEYE_ROOT / "src"))


@pytest.fixture
def iceye_root() -> Path:
    return ICEYE_ROOT


@pytest.fixture
def store_path() -> Path:
    return Path(__file__).with_name("entities.json")


@pytest.fixture
def store(store_path):
    from iceye_homework.ner import EntityStore

    return EntityStore.load(store_path)
