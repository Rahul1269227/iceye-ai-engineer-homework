from pathlib import Path

import pytest

from iceye_homework.ner import EntityStore


@pytest.fixture
def store_path() -> Path:
    return Path(__file__).with_name("entities.json")


@pytest.fixture
def store(store_path) -> EntityStore:
    return EntityStore.load(store_path)
