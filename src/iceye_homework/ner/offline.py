"""``--model offline``: a deterministic, rule-based stand-in for an LLM.

It drives the agent exactly as an LLM would, through tool calls only: it pulls candidate
names out of the question, looks them up, then writes a templated reply from what the
tools returned. Handy for demos and air-gapped runs; a real model understands free-form
questions far better.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from .models import MAX_ANSWER_ENTITIES

_MAX_LOOKUPS = 5
_QUOTED = re.compile(r'"([^"]+)"|\u201c([^\u201d]+)\u201d')
# Runs of capitalised (or numeric) words, allowing lower-case joiners: "Bank of Finland".
_NAME = re.compile(
    r"\b[A-Z0-9][\w&.'-]*(?:(?:\s+(?:of|the|de|du|la|van|von|der))*\s+[A-Z0-9][\w&.'-]*)*"
)
_LEADING_WORDS = frozenset({
    "how", "what", "which", "who", "whom", "whose", "where", "when", "why",
    "is", "are", "was", "were", "do", "does", "did", "can", "could",
    "tell", "list", "show", "give", "name", "please",
    "and", "or", "about", "in", "on", "the", "a", "an", "i", "me",
})  # fmt: skip
_TOP_INTENT = re.compile(r"\b(most|top|frequent|common|main|key|often)\b", re.IGNORECASE)
_LABEL_WORDS = {
    "person": ("person", "persons", "people", "who"),
    "organization": ("organization", "organizations", "organisation", "organisations",
                     "company", "companies"),
    "location": ("location", "locations", "place", "places", "city", "cities", "country",
                 "countries", "where"),
    "date": ("date", "dates", "when"),
    "event": ("event", "events"),
    "product": ("product", "products"),
}  # fmt: skip


def offline_model() -> FunctionModel:
    return FunctionModel(_respond, model_name="offline")


def _respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    returns = [
        part
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, ToolReturnPart)
    ]
    if not returns:
        return ModelResponse(parts=_lookups(_question(messages)))
    output_tool = info.output_tools[0].name
    return ModelResponse(parts=[ToolCallPart(output_tool, _answer(returns))])


def _question(messages: list[ModelMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    return part.content
    return ""


def candidate_names(question: str) -> list[str]:
    """Quoted phrases, else runs of capitalised words minus leading question words."""
    quoted = [a or b for a, b in _QUOTED.findall(question)]
    if quoted:
        return quoted[:_MAX_LOOKUPS]
    names: list[str] = []
    for match in _NAME.finditer(question):
        words = match.group().split()
        while words and words[0].casefold() in _LEADING_WORDS:
            words.pop(0)
        name = " ".join(words)
        if name and name not in names:
            names.append(name)
    return names[:_MAX_LOOKUPS]


def _lookups(question: str) -> list[ToolCallPart]:
    names = candidate_names(question)
    if names:
        return [ToolCallPart("count_mentions", {"name": name}) for name in names]
    words = set(re.findall(r"[a-z]+", question.casefold()))
    label = next((lab for lab, syn in _LABEL_WORDS.items() if words & set(syn)), None)
    if label or _TOP_INTENT.search(question):
        return [ToolCallPart("top_entities", {"label": label, "limit": 5})]
    return [ToolCallPart("list_labels", {})]


def _answer(returns: Sequence[ToolReturnPart]) -> dict[str, Any]:
    sentences: list[str] = []
    entities: list[dict[str, Any]] = []
    for part in returns:
        content = part.content
        match part.tool_name:
            case "count_mentions":
                sentences.append(_count_sentence(content, entities))
            case "top_entities" | "search_entities":
                entities += [_cited(row) for row in content]
                listing = ", ".join(f"{r['name']} ({r['mention_count']})" for r in content)
                sentences.append(
                    f"The most mentioned are {listing}." if content else "I found no matches."
                )
            case "list_labels":
                kinds = ", ".join(
                    f"{_plural(r['entities'], r['label'] + ' entity', r['label'] + ' entities')}"
                    f" ({_plural(r['mentions'], 'mention')})"
                    for r in content
                )
                sentences.append(f"The store holds {kinds}." if content else "The store is empty.")
    reply = "Here is what the entity store says: " + " ".join(sentences)
    return {"reply": reply, "entities": entities[:MAX_ANSWER_ENTITIES]}


def _count_sentence(result: dict[str, Any], entities: list[dict[str, Any]]) -> str:
    if result["found"]:
        entities.append(_cited(result))
        count = result["mention_count"]
        return f"{result['name']} ({result['label']}) is mentioned {_times(count)}."
    entities.append({"name": result["name"], "label": None, "mention_count": 0, "in_store": False})
    sentence = f"{result['name']} is not mentioned."
    if result["did_you_mean"]:
        best = result["did_you_mean"][0]
        entities.append(_cited(best))
        times = _times(best["mention_count"])
        sentence += f" The closest match is {best['name']}, mentioned {times}."
    return sentence


def _cited(row: dict[str, Any]) -> dict[str, Any]:
    return {"name": row["name"], "label": row["label"], "mention_count": row["mention_count"],
            "in_store": True}  # fmt: skip


def _times(count: int) -> str:
    return _plural(count, "time")


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else plural or singular + 's'}"
