"""ask grounding, tools and model resolution."""

import pytest
from pydantic_ai import capture_run_messages, models
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.models.test import TestModel

from iceye_homework.ner import Answer, EntityMention, ask
from iceye_homework.ner.agent import (
    AskDeps,
    ground_mentions,
    quoted_counts,
    resolve_model,
    unknown_names,
    unsupported_counts,
)
from iceye_homework.ner.offline import offline_model
from iceye_homework.ner.store import normalize


@pytest.fixture(autouse=True)
def _no_network_models(monkeypatch):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NER_CHAT_MODEL"):
        monkeypatch.delenv(key, raising=False)


def test_ask_offline_iceye_and_nasa(store):
    answer = ask("How many times are ICEYE and NASA mentioned?", store, "offline")
    by_name = {m.name: m for m in answer.entities}
    assert by_name["ICEYE"] == EntityMention(
        name="ICEYE", label="organization", mention_count=7, in_store=True
    )
    assert by_name["NASA"] == EntityMention(
        name="NASA", label=None, mention_count=0, in_store=False
    )
    assert "7" in answer.reply or "seven" in answer.reply.casefold()


def test_ask_offline_alias(store):
    answer = ask('How often is "Iceye Oy" mentioned?', store, offline_model())
    assert answer.entities[0].name == "ICEYE"
    assert answer.entities[0].mention_count == 7


def test_ask_grounds_bad_model_output(store):
    model = TestModel(
        custom_output_args={
            "reply": "ICEYE is mentioned a lot, NASA is not.",
            "entities": [
                {"name": "Iceye Oy", "label": "person", "mention_count": 99, "in_store": False},
                {"name": "NASA", "label": "organization", "mention_count": 3, "in_store": True},
            ],
        }
    )
    with capture_run_messages() as messages:
        answer = ask("How often are ICEYE and NASA mentioned?", store, model)

    assert answer.entities == [
        EntityMention(name="ICEYE", label="organization", mention_count=7, in_store=True),
        EntityMention(name="NASA", label=None, mention_count=0, in_store=False),
    ]
    tools = {
        part.tool_name
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    }
    assert "count_mentions" in tools
    for message in messages:
        if isinstance(message, ModelRequest):
            for entity in store.entities:
                assert entity.name not in (message.instructions or "")


def test_ask_loads_path(store_path):
    answer = ask("Is SAR mentioned?", store_path, "offline")
    assert any(m.name == "SAR" and m.in_store for m in answer.entities)


def test_resolve_model_requires_name(monkeypatch):
    monkeypatch.delenv("NER_CHAT_MODEL", raising=False)
    with pytest.raises(ValueError, match="no model"):
        resolve_model(None)


def test_resolve_model_env(monkeypatch):
    monkeypatch.setenv("NER_CHAT_MODEL", "offline")
    resolved = resolve_model(None)
    assert getattr(resolved, "model_name", None) == "offline"


def test_ground_mentions_dedupes_and_absent(store):
    grounded = ground_mentions(
        store,
        [
            EntityMention(name="Iceye Oy", label="x", mention_count=1, in_store=False),
            EntityMention(name="ICEYE", label="y", mention_count=2, in_store=True),
            EntityMention(name="NASA", label="organization", mention_count=9, in_store=True),
        ],
    )
    assert grounded == [
        EntityMention(name="ICEYE", label="organization", mention_count=7, in_store=True),
        EntityMention(name="NASA", label=None, mention_count=0, in_store=False),
    ]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("mentioned 7 times", [(10, 7)]),
        ("twice", [(0, 2)]),
        ("once and thrice", [(0, 1), (9, 3)]),
        ("twenty-one mentions", [(0, 21)]),
        ("no mentions", [(0, 0)]),
    ],
)
def test_quoted_counts(text, expected):
    assert quoted_counts(text) == expected


def test_unsupported_counts_flags_wrong_number(store):
    deps = AskDeps(store, "How often is ICEYE mentioned?")
    deps.seen_counts.add(7)
    bad = unsupported_counts("ICEYE is mentioned 99 times.", deps, set())
    assert bad and bad[0][0] == 99


def test_unsupported_counts_accepts_store_count(store):
    deps = AskDeps(store, "How often is ICEYE mentioned?")
    deps.seen_counts.add(7)
    assert unsupported_counts("ICEYE is mentioned 7 times.", deps, set()) == []


def test_unknown_names_flags_outsiders(store):
    deps = AskDeps(store, "How often is ICEYE mentioned?")
    assert "SpaceX" in unknown_names("Yes, SpaceX is huge.", deps)


def test_unknown_names_allows_store_and_question(store):
    deps = AskDeps(store, "Is NASA mentioned?")
    assert unknown_names("NASA is not mentioned. ICEYE is.", deps) == []


def test_ask_returns_answer_model(store):
    answer = ask("List the top entities", store, "offline")
    Answer.model_validate(answer.model_dump())
    assert answer.reply
    assert normalize("iceye") == "iceye"
