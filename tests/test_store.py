"""Entity store: normalisation, lookup, search, save/load."""

import pytest
from pydantic import ValidationError

from iceye_homework.ner.models import EntityStoreFile, StoredEntity
from iceye_homework.ner.store import (
    EntityStore,
    clean_surface,
    fold,
    normalize,
    words,
    write_store,
)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  ICEYE's, ", "ICEYE"),
        ("Dr. Aino Virtanen", "Dr. Aino Virtanen"),
        ('"Helsinki"', "Helsinki"),
        ("  synthetic   aperture  radar ", "synthetic aperture radar"),
    ],
)
def test_clean_surface(raw, expected):
    assert clean_surface(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("ICEYE", "iceye"),
        ("Iceye Oy", "iceye oy"),
        ("Dr. Aino Virtanen", "aino virtanen"),
        ("Prof. Rafal Modrzewski", "rafal modrzewski"),
        ("ICEYE's", "iceye"),
    ],
)
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_fold_and_words():
    assert fold("  Hello\tWORLD ") == "hello world"
    assert words("Bank of Finland") == {"bank", "of", "finland"}


def test_lookup_canonical_and_alias(store):
    iceye = store.lookup("ICEYE")
    assert iceye is not None
    assert iceye.count == 7
    assert store.lookup("Iceye Oy") is iceye
    assert store.lookup("iceye") is iceye
    assert store.lookup("NASA") is None


def test_search_substring_and_label(store):
    hits = store.search("fin")
    assert any(e.name == "Finland" for e in hits)

    orgs = store.search("ice", label="organization")
    assert [e.name for e in orgs] == ["ICEYE"]
    assert store.search("ice", label="location") == []


def test_search_empty_query(store):
    assert store.search("   ") == []
    assert store.search("!!!") == []


def test_suggest_and_top(store):
    assert "ICEYE" in store.suggest("Icey")
    top = store.top(limit=3)
    assert [e.name for e in top] == ["ICEYE", "SAR", "Finland"]
    people = store.top(label="person")
    assert [e.name for e in people] == ["Rafal Modrzewski"]


def test_label_summary(store):
    summary = {s.label: (s.entities, s.mentions) for s in store.label_summary()}
    assert summary["organization"] == (1, 7)
    assert summary["location"] == (2, 7)
    assert summary["person"] == (1, 2)


def test_mentions_in_prefers_longer_name(store):
    found = store.mentions_in("Iceye Oy builds SAR in Finland")
    names = [entity.name for _, _, entity in found]
    assert names == ["ICEYE", "SAR", "Finland"]


def test_save_roundtrip(store, tmp_path):
    path = tmp_path / "out.json"
    store.save(path)
    again = EntityStore.load(path)
    assert again.entities == store.entities
    assert again.lookup("Iceye Oy").name == "ICEYE"


def test_write_store_atomic(tmp_path):
    path = tmp_path / "nested" / "entities.json"
    data = EntityStoreFile(
        source="t",
        model="m",
        labels=["organization"],
        entities=[StoredEntity(name="ICEYE", label="organization", count=1)],
    )
    write_store(data, path)
    assert EntityStore.load(path).lookup("ICEYE").count == 1


def test_duplicate_keys_raise():
    data = EntityStoreFile(
        entities=[
            StoredEntity(name="ICEYE", label="organization", count=1),
            StoredEntity(name="Iceye", label="organization", count=2),
        ]
    )
    with pytest.raises(ValueError, match="matches both"):
        EntityStore(data)


def test_stored_entity_rejects_empty_name():
    with pytest.raises(ValidationError):
        StoredEntity(name="", label="organization", count=1)
