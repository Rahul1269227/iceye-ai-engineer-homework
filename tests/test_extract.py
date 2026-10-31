"""chunking and mention aggregation (no GLiNER download)."""

from iceye_homework.ner.extract import Mention, aggregate, chunk_text


def test_chunk_text_keeps_offsets():
    text = "One. " + ("word " * 50) + "Two. Three."
    chunks = chunk_text(text, max_words=20)
    assert chunks
    assert chunks[0].start == 0
    for chunk in chunks:
        assert text[chunk.start : chunk.start + len(chunk.text)] == chunk.text


def test_chunk_text_empty():
    assert chunk_text("") == []


def test_chunk_text_long_sentence_splits():
    text = " ".join(f"w{i}" for i in range(25))
    chunks = chunk_text(text, max_words=10)
    assert len(chunks) == 3
    assert sum(len(c.text.split()) for c in chunks) == 25


def test_aggregate_merges_aliases_and_majority_label():
    mentions = [
        Mention("Dr. Aino Virtanen", "person", 0, 17),
        Mention("Aino Virtanen", "person", 20, 33),
        Mention("Aino Virtanen", "organization", 40, 53),
        Mention("ICEYE", "organization", 60, 65),
        Mention("Iceye", "organization", 70, 75),
        Mention("ICEYE's", "organization", 80, 87),
    ]
    entities = {e.name: e for e in aggregate(mentions)}
    aino = entities["Aino Virtanen"]
    assert aino.count == 3
    assert aino.label == "person"
    assert "Dr. Aino Virtanen" in aino.aliases

    iceye = entities["ICEYE"]
    assert iceye.count == 3
    assert "Iceye" in iceye.aliases


def test_aggregate_empty():
    assert aggregate([]) == []


def test_aggregate_sorted_by_count():
    mentions = [
        Mention("A", "location", 0, 1),
        Mention("B", "location", 2, 3),
        Mention("B", "location", 4, 5),
    ]
    names = [e.name for e in aggregate(mentions)]
    assert names == ["B", "A"]
