"""``ask``: answer one question about the document's entities from the store, via tools.

The model never sees the document or the entity inventory. It looks entities up through
capped tools. An output validator rewrites every entity it cites with the store's name,
label and count, and sends the reply back for a rewrite if it quotes a count or a name
the store does not support. If the model still fails, the offline model answers instead.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pydantic_ai
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.exceptions import UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.models import Model
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import UsageLimits

from .models import MAX_ANSWER_ENTITIES, Answer, EntityMention, StoredEntity
from .offline import offline_model
from .store import EntityStore, clean_surface, fold, normalize, words

MODEL_ENV = "NER_CHAT_MODEL"
OFFLINE_MODEL = "offline"
MAX_TOOL_RESULTS = 10
_REQUEST_LIMIT = 8

logger = logging.getLogger(__name__)
pydantic_ai.BANNER_ENABLED = False

INSTRUCTIONS = """\
You answer one question about the named entities detected in a document. You cannot see \
the document or its entity list; look everything up with the tools:
- count_mentions(name): use it for every entity the user names, even one you expect to be \
absent.
- search_entities(query, label): partial, misspelled or descriptive names.
- top_entities(label): "most mentioned" questions.
- list_labels(): which entity types exist, how many mentions each has, and the totals.
Reply in one or two friendly sentences that acknowledge the question. Use only names and \
counts returned by the tools, and give each count right after its entity's name. \
In `entities`, list just the entities your reply talks about (at most 10), including any \
the user asked about that are not in the store (mention_count 0).\
"""

_UNITS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
)  # fmt: skip
_TENS = ("twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_NUMBER = rf"\d+|no|(?:{'|'.join(_TENS)})(?:-(?:{'|'.join(_UNITS[1:10])}))?|{'|'.join(_UNITS)}"
_COUNT_NOUN = "times?|mentions?|occurrences?|occasions?|hits?|references?|appearances?|instances?"
# "7 times", "27 entity mentions", "on 9 occasions", "twenty-one hits", "twice".
_QUOTED_COUNT = re.compile(
    rf"\b({_NUMBER})\s+(?:(?!of\b)[^\W\d_]+\s+){{0,2}}?(?:{_COUNT_NOUN})\b"
    r"|\b(once|twice|thrice)\b",
    re.IGNORECASE,
)
_ADVERB_COUNTS = {"once": 1, "twice": 2, "thrice": 3}
_SENTENCE_BREAK = re.compile(r"(?<=[.!?;])\s+|\n+")
_CLAUSE_BREAK = re.compile(r"(?<=[.!?;:])\s+|\n+")
_WORD = re.compile(r"[^\W_]+")


@dataclass
class AskDeps:
    store: EntityStore
    question: str = ""
    # Entity counts the tools returned during this run.
    seen_counts: set[int] = field(default_factory=set)
    # Per-label and overall totals from list_labels: aggregates a reply may also quote.
    totals: set[int] = field(default_factory=set)
    # Normalised names count_mentions looked up and did not find.
    absent: set[str] = field(default_factory=set)

    def rows(self, entities: Iterable[StoredEntity]) -> list[dict[str, object]]:
        out = []
        for entity in entities:
            self.seen_counts.add(entity.count)
            out.append({"name": entity.name, "label": entity.label, "mention_count": entity.count})
        return out


agent = Agent(
    output_type=Answer,
    deps_type=AskDeps,
    instructions=INSTRUCTIONS,
    name="ner-chat",
    retries=2,
)


@agent.tool
def count_mentions(ctx: RunContext[AskDeps], name: str) -> dict[str, object]:
    """Mentions of one entity in the document; 0 (with spelling suggestions) if absent."""
    entity = ctx.deps.store.lookup(name)
    if entity is not None:
        return {"found": True, **ctx.deps.rows([entity])[0]}
    ctx.deps.seen_counts.add(0)
    if key := normalize(name):
        ctx.deps.absent.add(key)
    return {
        "found": False,
        "name": name,
        "mention_count": 0,
        "did_you_mean": ctx.deps.rows(ctx.deps.store.search(name, limit=3)),
    }


@agent.tool
def search_entities(
    ctx: RunContext[AskDeps], query: str, label: str | None = None, limit: int = 5
) -> list[dict[str, object]]:
    """Entities whose name resembles ``query`` (substring, shared words, close spelling),
    optionally of one label. At most 10."""
    return ctx.deps.rows(ctx.deps.store.search(query, label, _clamp(limit)))


@agent.tool
def top_entities(
    ctx: RunContext[AskDeps], label: str | None = None, limit: int = 5
) -> list[dict[str, object]]:
    """The most mentioned entities, optionally of one label. At most 10."""
    return ctx.deps.rows(ctx.deps.store.top(label, _clamp(limit)))


@agent.tool
def list_labels(ctx: RunContext[AskDeps]) -> dict[str, object]:
    """Entity types in the store with how many entities and mentions each has (at most 10
    types, most mentioned first), plus totals over the whole store."""
    summary = ctx.deps.store.label_summary()
    total_entities = sum(s.entities for s in summary)
    total_mentions = sum(s.mentions for s in summary)
    shown = summary[:MAX_TOOL_RESULTS]
    ctx.deps.totals.update(n for s in shown for n in (s.entities, s.mentions))
    ctx.deps.totals.update((total_entities, total_mentions))
    labels = [{"label": s.label, "entities": s.entities, "mentions": s.mentions} for s in shown]
    return {
        "labels": labels,
        "total_entities": total_entities,
        "total_mentions": total_mentions,
    }


@agent.output_validator
def ground_answer(ctx: RunContext[AskDeps], answer: Answer) -> Answer:
    """Replace every cited entity with the store's record (keeping at most 10), and send
    the reply back if it quotes a count or a name the store does not support."""
    deps = ctx.deps
    grounded = ground_mentions(deps.store, answer.entities)[:MAX_ANSWER_ENTITIES]
    absent = deps.absent | {normalize(m.name) for m in grounded if not m.in_store}
    problems = []
    if bad_counts := unsupported_counts(answer.reply, deps, absent):
        facts = "; ".join(f"{e.name}: {e.count}" for e in _named(bad_counts)) or "none named"
        quoted = sorted({count for count, _ in bad_counts})
        problems.append(
            f"it quotes {quoted} mentions, which do not match the store "
            f"(entities named in those sentences: {facts})"
        )
    if unknown := unknown_names(answer.reply, deps):
        problems.append(f"it names {unknown}, which are neither in the store nor the question")
    if problems:
        raise ModelRetry(
            f"Rewrite the reply: {'; '.join(problems)}. Use only names and counts returned "
            "by the tools, and give each count right after its entity's name."
        )
    return answer.model_copy(update={"entities": grounded})


def ground_mentions(store: EntityStore, mentions: Sequence[EntityMention]) -> list[EntityMention]:
    """Store-backed copies of ``mentions``: canonical name, label and count when the store
    has the entity, otherwise label ``None``, count 0 and ``in_store=False``. Duplicates
    (by normalised name) are dropped."""
    grounded: list[EntityMention] = []
    seen: set[str] = set()
    for mention in mentions:
        entity = store.lookup(mention.name)
        if entity is not None:
            key, item = (
                normalize(entity.name),
                EntityMention(
                    name=entity.name, label=entity.label, mention_count=entity.count, in_store=True
                ),
            )
        else:
            name = clean_surface(mention.name)
            key, item = (
                normalize(name),
                EntityMention(name=name, label=None, mention_count=0, in_store=False),
            )
        if key and key not in seen:
            seen.add(key)
            grounded.append(item)
    return grounded


def unsupported_counts(
    reply: str, deps: AskDeps, absent: set[str]
) -> list[tuple[int, list[StoredEntity]]]:
    """Counts quoted in ``reply`` that the store does not back, with the store entities
    named in the same sentence.

    A count belongs to the nearest name before it in its sentence (a store entity, or an
    absent name with count 0) and must be that name's count, or the sum over the store
    entities the sentence names. With no name before it, the count may also be the first
    name's count after it, a store total from ``list_labels`` or, in a sentence that names
    nothing, any count a tool returned.
    """
    bad = []
    for sentence in _SENTENCE_BREAK.split(reply):
        folded = fold(sentence)
        counts = quoted_counts(folded)
        if not counts:
            continue
        found = deps.store.mentions_in(folded)
        entities = list({id(entity): entity for _, _, entity in found}.values())
        names = [(start, entity.count) for start, _, entity in found]
        for key in absent:
            for match in re.finditer(rf"(?<!\w){re.escape(key)}(?!\w)", folded):
                if not any(start <= match.start() < end for start, end, _ in found):
                    names.append((match.start(), 0))
        names.sort()
        together = {sum(entity.count for entity in entities)} if len(entities) > 1 else set()
        for offset, count in counts:
            before = [name_count for start, name_count in names if start < offset]
            if before:
                allowed = {before[-1]} | together
            elif names:
                allowed = {names[0][1]} | together | deps.totals
            else:
                allowed = deps.seen_counts | deps.totals
            if count not in allowed:
                bad.append((count, entities))
    return bad


def unknown_names(reply: str, deps: AskDeps) -> list[str]:
    """Capitalised words in ``reply`` that appear in no store name, label or the question.

    The first word of a sentence (or after a colon) is skipped unless it is an acronym:
    it is capitalised anyway and is usually an ordinary word.
    """
    store = deps.store
    known = store.name_words | words(deps.question) | words(" ".join(store.data.labels))
    known |= words(" ".join(entity.label for entity in store.entities))
    unknown: list[str] = []
    for sentence in _CLAUSE_BREAK.split(reply):
        for index, match in enumerate(_WORD.finditer(sentence)):
            word = match.group()
            if len(word) < 2 or not word[0].isupper() or (index == 0 and not word.isupper()):
                continue
            if word.casefold() not in known and word not in unknown:
                unknown.append(word)
    return unknown


def quoted_counts(text: str) -> list[tuple[int, int]]:
    """``(offset, count)`` for each count ``text`` states as mentions: "7 times", "two
    mentions", "once", ..."""
    counts = []
    for match in _QUOTED_COUNT.finditer(text):
        number, adverb = match.groups()
        count = _ADVERB_COUNTS[adverb.casefold()] if adverb else _number_value(number)
        counts.append((match.start(), count))
    return counts


def _number_value(word: str) -> int:
    if word.isdigit():
        return int(word)
    word = word.casefold()
    if word == "no":
        return 0
    tens, _, unit = word.partition("-")
    if tens in _TENS:
        return 20 + 10 * _TENS.index(tens) + (_UNITS.index(unit) if unit else 0)
    return _UNITS.index(word)


def _named(bad_counts: list[tuple[int, list[StoredEntity]]]) -> list[StoredEntity]:
    found: dict[int, StoredEntity] = {}
    for _, named in bad_counts:
        for entity in named:
            found.setdefault(id(entity), entity)
    return list(found.values())


def resolve_model(model: Model | str | None) -> Model | str:
    """The given model, else ``$NER_CHAT_MODEL``. ``"test"`` is Pydantic AI's TestModel and
    ``"offline"`` the rule-based model; anything else is a Pydantic AI model name."""
    resolved = model or os.environ.get(MODEL_ENV)
    if not resolved:
        raise ValueError(
            f"no model given: pass --model (e.g. 'openai:gpt-4.1-mini', or 'offline' / 'test' "
            f"to run without a key) or set {MODEL_ENV}"
        )
    return offline_model() if resolved == OFFLINE_MODEL else resolved


def ask(question: str, store: EntityStore | Path | str, model: Model | str | None = None) -> Answer:
    """Answer ``question`` from the entity store only. Never opens the source document.

    If the model gives no answer that passes the checks within its retries (or loops on
    tools), the offline model answers from the same tools instead, and a warning is logged.
    """
    if not isinstance(store, EntityStore):
        store = EntityStore.load(store)
    resolved = resolve_model(model)
    try:
        return _run(question, store, resolved)
    except (UnexpectedModelBehavior, UsageLimitExceeded) as exc:
        if isinstance(resolved, FunctionModel) and resolved.model_name == OFFLINE_MODEL:
            raise
        logger.warning("warning: %s; answering with the offline model instead", exc)
        return _run(question, store, offline_model())


def _run(question: str, store: EntityStore, model: Model | str) -> Answer:
    result = agent.run_sync(
        question,
        model=model,
        deps=AskDeps(store, question),
        usage_limits=UsageLimits(request_limit=_REQUEST_LIMIT),
    )
    return result.output


def _clamp(limit: int) -> int:
    return max(1, min(limit, MAX_TOOL_RESULTS))
