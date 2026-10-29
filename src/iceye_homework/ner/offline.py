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

_QUOTED = re.compile(r'"([^"]+)"|\u201c([^\u201d]+)\u201d')
# Runs of capitalised (or numeric) words, allowing lower-case joiners: "Bank of Finland".
_NAME = re.compile(
    r"\b[A-Z0-9][\w&.'-]*(?:(?:\s+(?:of|the|de|du|la|van|von|der))*\s+[A-Z0-9][\w&.'-]*)*"
)
# Capitalised only because they start a sentence: dropped from the front of a name.
_LEADING_WORDS = frozenset({
    "how", "what", "which", "who", "whom", "whose", "where", "when", "why",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "would", "will",
    "has", "have", "had", "tell", "list", "show", "give", "name", "please", "find",
    "count", "check", "say", "print", "dump", "output", "repeat", "ignore", "forget",
    "explain", "describe", "summarize", "summarise", "provide", "return", "write",
    "hi", "hello", "hey", "thanks", "ok", "okay", "yes", "no", "now", "also", "then",
    "and", "or", "about", "in", "on", "for", "from", "with", "by", "to", "of", "if",
    "the", "a", "an", "this", "that", "these", "those", "all", "any", "every", "each",
    "i", "me", "my", "you", "your", "we", "our", "it", "its", "they", "their",
})  # fmt: skip
# Lower-case names in the usual question shapes: "is helsinki mentioned?", "how often do
# nasa and esa appear", "mentions of aino virtanen".
_LOWER_NAMES = re.compile(
    r"\b(?:is|are|was|were|does|do|did)\s+(?:the\s+)?(.+?)\s+"
    r"(?:mentioned|named|referenced|cited|appear|appears|present|in\s+the)\b"
    r"|\b(?:mentions?\s+of|(?:document|text|report|it)\s+mentions?)\s+(.+?)(?:[?.!,;]|$)",
    re.IGNORECASE,
)
_SPLIT_NAMES = re.compile(r"\s*(?:,|\band\b|\bor\b)\s*", re.IGNORECASE)
_ENTITY_WORDS = frozenset({"entity", "entities", "name", "names", "thing", "things", "one", "ones"})
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
_GENERIC_WORDS = (
    _LEADING_WORDS | _ENTITY_WORDS | {word for words in _LABEL_WORDS.values() for word in words}
)


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
    question = _question(messages)
    if not returns:
        return ModelResponse(parts=_lookups(question))
    output_tool = info.output_tools[0].name
    skipped = len(candidate_names(question)) - MAX_ANSWER_ENTITIES
    return ModelResponse(parts=[ToolCallPart(output_tool, _answer(returns, skipped))])


def _question(messages: list[ModelMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart) and isinstance(part.content, str):
                    return part.content
    return ""


def candidate_names(question: str) -> list[str]:
    """Quoted phrases; else runs of capitalised words minus leading question words; else
    lower-case names in common question shapes."""
    quoted = [a or b for a, b in _QUOTED.findall(question)]
    if quoted:
        return quoted
    names: list[str] = []
    for match in _NAME.finditer(question):
        for words in _sentence_parts(match.group().split()):
            while words and words[0].casefold() in _LEADING_WORDS:
                words.pop(0)
            _add(names, " ".join(words))
    if names:
        return names
    for match in _LOWER_NAMES.finditer(question):
        for name in _SPLIT_NAMES.split(match.group(1) or match.group(2)):
            if not set(name.casefold().split()) <= _GENERIC_WORDS:
                _add(names, name)
    return names


def _sentence_parts(words: list[str]) -> list[list[str]]:
    """``words`` split after each word that ends a sentence, with the full stop dropped;
    abbreviations such as "U.S." or "St." do not end one."""
    parts: list[list[str]] = [[]]
    for word in words:
        stem = word.removesuffix(".")
        if stem != word and len(stem) > 3 and "." not in stem:
            parts[-1].append(stem)
            parts.append([])
        else:
            parts[-1].append(word)
    return parts


def _add(names: list[str], name: str) -> None:
    if name and name.casefold() not in _LEADING_WORDS and name not in names:
        names.append(name)


def _lookups(question: str) -> list[ToolCallPart]:
    names = candidate_names(question)[:MAX_ANSWER_ENTITIES]
    if names:
        return [ToolCallPart("count_mentions", {"name": name}) for name in names]
    words = set(re.findall(r"[a-z]+", question.casefold()))
    label = next((lab for lab, syn in _LABEL_WORDS.items() if words & set(syn)), None)
    if label or _TOP_INTENT.search(question):
        return [ToolCallPart("top_entities", {"label": label, "limit": 5})]
    return [ToolCallPart("list_labels", {})]


def _answer(returns: Sequence[ToolReturnPart], skipped: int = 0) -> dict[str, Any]:
    sentences: list[str] = []
    entities: list[dict[str, Any]] = []
    for part in returns:
        content: Any = part.content
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
                sentences.append(_labels_sentence(content))
    if skipped > 0:
        sentences.append(f"I looked up only the first {MAX_ANSWER_ENTITIES} names you gave.")
    reply = "Here is what the entity store says: " + " ".join(sentences)
    return {"reply": reply, "entities": entities[:MAX_ANSWER_ENTITIES]}


def _count_sentence(result: dict[str, Any], entities: list[dict[str, Any]]) -> str:
    if result["found"]:
        entities.append(_cited(result))
        count = result["mention_count"]
        return f"{result['name']} ({result['label']}) is mentioned {_times(count)}."
    entities.append({"name": result["name"], "label": None, "mention_count": 0, "in_store": False})
    if not result["did_you_mean"]:
        return f"{result['name']} is not mentioned."
    best = result["did_you_mean"][0]
    entities.append(_cited(best))
    return (
        f"{result['name']} is not an entity in the store; the closest match is {best['name']} "
        f"({best['label']}), mentioned {_times(best['mention_count'])}."
    )


def _labels_sentence(content: dict[str, Any]) -> str:
    if not content["labels"]:
        return "The store is empty."
    kinds = ", ".join(
        f"{_plural(r['entities'], r['label'] + ' entity', r['label'] + ' entities')}"
        f" ({_plural(r['mentions'], 'mention')})"
        for r in content["labels"]
    )
    total = _plural(content["total_entities"], "entity", "entities")
    return f"The store holds {total} with {_plural(content['total_mentions'], 'mention')}: {kinds}."


def _cited(row: dict[str, Any]) -> dict[str, Any]:
    return {"name": row["name"], "label": row["label"], "mention_count": row["mention_count"],
            "in_store": True}  # fmt: skip


def _times(count: int) -> str:
    return _plural(count, "time")


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count} {singular if count == 1 else plural or singular + 's'}"
