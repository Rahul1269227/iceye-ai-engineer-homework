"""offline model name extraction and replies."""

import pytest

from iceye_homework.ner import ask
from iceye_homework.ner.offline import candidate_names


@pytest.mark.parametrize(
    "question, expected",
    [
        ('How often is "Iceye Oy" mentioned?', ["Iceye Oy"]),
        ("How many times are ICEYE and NASA mentioned?", ["ICEYE", "NASA"]),
        ("Is Helsinky mentioned?", ["Helsinky"]),
        ("is helsinki mentioned?", ["helsinki"]),
        ("mentions of aino virtanen", ["aino virtanen"]),
        ("What companies are mentioned most?", []),
        ("Summarise the store", []),
    ],
)
def test_candidate_names(question, expected):
    assert candidate_names(question) == expected


def test_offline_top_entities(store):
    answer = ask("Show most frequent", store, "offline")
    names = [m.name for m in answer.entities]
    assert "ICEYE" in names
    assert "ICEYE" in answer.reply


def test_offline_list_labels(store):
    answer = ask("Summarise the store", store, "offline")
    assert "organization" in answer.reply.casefold() or "entity" in answer.reply.casefold()


def test_offline_people_label(store):
    answer = ask("most frequent people", store, "offline")
    assert any(m.name == "Rafal Modrzewski" for m in answer.entities)


def test_offline_misspelling_suggestion(store):
    answer = ask("Is Icey mentioned?", store, "offline")
    assert any(m.name == "ICEYE" for m in answer.entities)
    assert "closest" in answer.reply.casefold() or "ICEYE" in answer.reply
