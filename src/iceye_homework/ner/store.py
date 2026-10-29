"""Read-only access to the entity store, plus the name normalisation shared with extract."""

from __future__ import annotations

import difflib
import os
import re
import tempfile
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .models import EntityStoreFile, StoredEntity

# Quotes, brackets and sentence punctuation that NER spans and user questions pick up,
# including typographic quotes, guillemets, ellipsis and dashes.
_EDGE_CHARS = " \t\n\"'`.,;:!?()[]{}<>*_/\\|-\u201c\u201d\u2018\u2019\u00ab\u00bb\u2026\u2013\u2014"
_POSSESSIVE = re.compile("['\u2019]s$")
_HONORIFIC = re.compile(r"^(?:dr|prof|mr|mrs|ms|mx|sir|dame)\.?\s+(?=\S)", re.IGNORECASE)
_TOKEN = re.compile(r"\w+")
_MATCH_THRESHOLD = 0.6


def clean_surface(name: str) -> str:
    """A mention as it should be displayed: NFKC, single spaces, no edge punctuation or
    possessive. ``"  ICEYE's, "`` becomes ``"ICEYE"``."""
    text = " ".join(unicodedata.normalize("NFKC", name).split())
    return _POSSESSIVE.sub("", text.strip(_EDGE_CHARS)).strip(_EDGE_CHARS)


def normalize(name: str) -> str:
    """Key used to match names: the cleaned surface form without a leading honorific,
    casefolded, so ``"Dr. Aino Virtanen"`` and ``"aino virtanen"`` share a key."""
    return _HONORIFIC.sub("", clean_surface(name)).casefold()


def fold(text: str) -> str:
    """Running text folded like :func:`normalize` folds a name: NFKC, single spaces,
    casefolded, so normalised names can be found in it."""
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def words(text: str) -> set[str]:
    """The casefolded word tokens of ``text``."""
    return set(_TOKEN.findall(fold(text)))


@dataclass(frozen=True, slots=True)
class LabelSummary:
    label: str
    entities: int
    mentions: int


class EntityStore:
    """In-memory index over an :class:`EntityStoreFile`.

    Every name and alias is indexed by its normalised key, so exact lookups are O(1).
    A key may belong to only one entity.
    """

    def __init__(self, data: EntityStoreFile) -> None:
        self.data = data
        self._by_key: dict[str, StoredEntity] = {}
        for entity in data.entities:
            for surface in (entity.name, *entity.aliases):
                key = normalize(surface)
                if not key:
                    continue
                owner = self._by_key.setdefault(key, entity)
                if owner is not entity:
                    raise ValueError(f"{surface!r} matches both {owner.name!r} and {entity.name!r}")
        # Longest first, so "Aino Virtanen" is found as itself rather than as "Virtanen".
        keys = sorted(self._by_key, key=len, reverse=True)
        alternatives = "|".join(map(re.escape, keys)) or "(?!)"
        self._names = re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)")
        self.name_words = frozenset(word for key in self._by_key for word in _TOKEN.findall(key))

    @classmethod
    def load(cls, path: Path | str) -> EntityStore:
        return cls(EntityStoreFile.model_validate_json(Path(path).read_bytes()))

    def save(self, path: Path | str) -> None:
        write_store(self.data, path)

    @property
    def entities(self) -> list[StoredEntity]:
        return self.data.entities

    def lookup(self, name: str) -> StoredEntity | None:
        """The entity whose name or alias matches ``name`` after normalisation."""
        return self._by_key.get(normalize(name))

    def mentions_in(self, text: str) -> list[tuple[int, int, StoredEntity]]:
        """``(start, end, entity)`` for each name or alias that ``fold(text)`` contains as
        whole words, in order; offsets are into ``fold(text)``."""
        return [
            (match.start(), match.end(), self._by_key[match.group()])
            for match in self._names.finditer(fold(text))
        ]

    def search(self, query: str, label: str | None = None, limit: int = 10) -> list[StoredEntity]:
        """Entities whose name or an alias resembles ``query``: exact, substring, shared
        words, or close spelling. Best matches first, then by mention count."""
        key = normalize(query)
        if not key:
            return []
        scored: dict[str, tuple[float, StoredEntity]] = {}
        for surface_key, entity in self._by_key.items():
            if label is not None and entity.label.casefold() != label.casefold():
                continue
            score = _similarity(key, surface_key)
            if score >= _MATCH_THRESHOLD and score > scored.get(entity.name, (0.0,))[0]:
                scored[entity.name] = (score, entity)
        ranked = sorted(scored.values(), key=lambda item: (-item[0], -item[1].count))
        return [entity for _, entity in ranked[:limit]]

    def suggest(self, name: str, limit: int = 3) -> list[str]:
        """Canonical names spelled like ``name``, for "did you mean" hints."""
        return [entity.name for entity in self.search(name, limit=limit)]

    def top(self, label: str | None = None, limit: int = 10) -> list[StoredEntity]:
        """Most mentioned entities, optionally of one label."""
        pool = [e for e in self.entities if label is None or e.label.casefold() == label.casefold()]
        return sorted(pool, key=lambda e: (-e.count, e.name.casefold()))[:limit]

    def label_summary(self) -> list[LabelSummary]:
        """Per label: how many distinct entities and how many mentions in total."""
        entities: dict[str, int] = defaultdict(int)
        mentions: dict[str, int] = defaultdict(int)
        for entity in self.entities:
            entities[entity.label] += 1
            mentions[entity.label] += entity.count
        summary = [LabelSummary(label, entities[label], mentions[label]) for label in entities]
        return sorted(summary, key=lambda s: (-s.mentions, s.label))


def _similarity(query: str, candidate: str) -> float:
    if query == candidate:
        return 1.0
    if query in candidate or candidate in query:
        return 0.9
    query_tokens, candidate_tokens = set(_TOKEN.findall(query)), set(_TOKEN.findall(candidate))
    overlap = len(query_tokens & candidate_tokens) / max(len(query_tokens | candidate_tokens), 1)
    spelling = difflib.SequenceMatcher(None, query, candidate).ratio()
    return max(0.5 + 0.4 * overlap if overlap else 0.0, spelling)


def write_store(data: EntityStoreFile, path: Path | str) -> None:
    """Write the store atomically: a crash never leaves a truncated file behind."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(data.model_dump_json(indent=2))
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
