"""Part 1: ``ask`` answers from the entity store only, through tools, with no API key."""

from pathlib import Path

from pydantic_ai import capture_run_messages, models
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from pydantic_ai.models.test import TestModel

from iceye_homework.ner import Answer, EntityMention, EntityStore, ask
from iceye_homework.ner.cli import main

STORE = Path(__file__).with_name("entities.json")


def test_ask_grounds_cited_entities_in_the_store(monkeypatch, capsys):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NER_CHAT_MODEL"):
        monkeypatch.delenv(key, raising=False)
    store = EntityStore.load(STORE)

    # The model cites ICEYE by its alias with a wrong label and count, and an absent name.
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

    requests = [message for message in messages if isinstance(message, ModelRequest)]
    tools_used = {
        part.tool_name
        for request in requests
        for part in request.parts
        if isinstance(part, ToolReturnPart)
    }
    assert "count_mentions" in tools_used
    for request in requests:
        for entity in store.entities:
            assert entity.name not in (request.instructions or "")

    argv = ["ask", "Is NASA mentioned?", "--store", str(STORE), "--model", "test", "--json"]
    assert main(argv) == 0
    Answer.model_validate_json(capsys.readouterr().out)
