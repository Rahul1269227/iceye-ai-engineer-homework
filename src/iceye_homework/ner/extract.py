"""``extract``: document -> text (Kreuzberg) -> mentions (GLinker) -> entity store.

No LLM is involved: entities are detected here, once, and ``ask`` only reads the store.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

from glinker import processor_registry
from glinker.l1 import L1Output
from kreuzberg import ExtractionConfig, OcrConfig, extract_file_sync

from .models import EntityStoreFile, StoredEntity
from .store import clean_surface, normalize

DEFAULT_MODEL = "knowledgator/gliner-bi-edge-v2.0"
DEFAULT_LABELS = ("person", "organization", "location", "date", "event", "product")
DEFAULT_THRESHOLD = 0.4
OCR_BACKENDS = ("paddleocr", "tesseract", "easyocr")
# GLiNER truncates long inputs, so text is fed in sentence-aligned chunks of this many words.
MAX_CHUNK_WORDS = 200

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n\s*\n")
_WORD = re.compile(r"\S+")


@dataclass(frozen=True, slots=True)
class Chunk:
    start: int
    text: str


@dataclass(frozen=True, slots=True)
class Mention:
    text: str
    label: str
    start: int
    end: int


def extract_text(path: Path, ocr_backend: str = "paddleocr", force_ocr: bool = False) -> str:
    """Document text via Kreuzberg; scanned pages and images go through OCR."""
    config = ExtractionConfig(ocr=OcrConfig(backend=ocr_backend), force_ocr=force_ocr)
    return extract_file_sync(path, config=config).content


def chunk_text(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[Chunk]:
    """Pack whole sentences into chunks of at most ``max_words`` words (a longer sentence
    is split between words). Each chunk keeps its offset into ``text``."""
    chunks: list[Chunk] = []
    start = end = words = 0
    for piece_start, piece_end, piece_words in _sentence_pieces(text, max_words):
        if words and words + piece_words > max_words:
            chunks.append(Chunk(start, text[start:end]))
            words = 0
        if not words:
            start = piece_start
        end, words = piece_end, words + piece_words
    if words:
        chunks.append(Chunk(start, text[start:end]))
    return chunks


def _sentence_pieces(text: str, max_words: int) -> Iterator[tuple[int, int, int]]:
    """``(start, end, n_words)`` of each sentence, overlong ones cut into ``max_words``."""
    bounds = [0]
    for match in _SENTENCE_BREAK.finditer(text):
        bounds += [match.start(), match.end()]
    bounds.append(len(text))
    for sentence_start, sentence_end in zip(bounds[::2], bounds[1::2], strict=True):
        spans = [m.span() for m in _WORD.finditer(text, sentence_start, sentence_end)]
        for i in range(0, len(spans), max_words):
            piece = spans[i : i + max_words]
            yield piece[0][0], piece[-1][1], len(piece)


def detect_mentions(
    chunks: Sequence[Chunk],
    labels: Sequence[str],
    threshold: float = DEFAULT_THRESHOLD,
    model: str = DEFAULT_MODEL,
    device: str = "cpu",
) -> list[Mention]:
    """Entity mentions with document offsets, from GLinker's GLiNER layer (L1).

    The bi-encoder model embeds the labels once and reuses them for every chunk.
    """
    if not chunks:
        return []
    processor = processor_registry.get("l1_gliner")(
        config_dict={
            "model": model,
            "labels": list(labels),
            "threshold": threshold,
            "device": device,
            "use_precomputed_embeddings": True,
        }
    )
    output: L1Output = processor(texts=[chunk.text for chunk in chunks])
    return [
        Mention(entity.text, entity.label or "unknown", chunk.start + entity.start,
                chunk.start + entity.end)
        for chunk, entities in zip(chunks, output.entities, strict=True)
        for entity in entities
    ]  # fmt: skip


def aggregate(mentions: Sequence[Mention]) -> list[StoredEntity]:
    """Group mentions by normalised name: the most frequent surface form (the shortest on
    ties, so "Aino Virtanen" beats "Dr. Aino Virtanen") becomes the name, the others
    aliases; the majority label wins (first seen on ties)."""
    surfaces: dict[str, Counter[str]] = defaultdict(Counter)
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    for mention in mentions:
        key = normalize(mention.text)
        if key:
            surfaces[key][clean_surface(mention.text)] += 1
            labels[key][mention.label] += 1
    entities = []
    for key, forms in surfaces.items():
        name, *others = sorted(forms, key=lambda form: (-forms[form], len(form)))
        entities.append(
            StoredEntity(
                name=name,
                label=labels[key].most_common(1)[0][0],
                count=forms.total(),
                aliases=others,
            )
        )
    return sorted(entities, key=lambda e: (-e.count, e.name.casefold()))


def extract_entities(
    path: Path,
    labels: Sequence[str] = DEFAULT_LABELS,
    threshold: float = DEFAULT_THRESHOLD,
    model: str = DEFAULT_MODEL,
    ocr_backend: str = "paddleocr",
    force_ocr: bool = False,
    device: str = "cpu",
) -> EntityStoreFile:
    """Run the whole pipeline on one document and return the store contents."""
    text = extract_text(path, ocr_backend=ocr_backend, force_ocr=force_ocr)
    mentions = detect_mentions(chunk_text(text), labels, threshold, model, device)
    return EntityStoreFile(
        source=path.name, model=model, labels=list(labels), entities=aggregate(mentions)
    )
