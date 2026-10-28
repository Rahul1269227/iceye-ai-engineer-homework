"""``ask``: answer one question about the document's entities from the store, via tools.

The model never sees the document or the entity inventory. It looks entities up through
capped tools, and an output validator rewrites every entity it cites with the store's
name, label and count, so the structured answer is grounded whatever the model says.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from .models import Answer, EntityMention, StoredEntity
from .offline import offline_model
from .store import EntityStore, clean_surface, normalize

MODEL_ENV = "NER_CHAT_MODEL"
OFFLINE_MODEL = "offline"
MAX_TOOL_RESULTS = 10
_REQUEST_LIMIT = 8

INSTRUCTIONS = """\
You answer one question about the named entities detected in a document. You cannot see \
the document or its entity list; look everything up with the tools:
- count_mentions(name): use it for every entity the user names, even one you expect to be \
absent.
- search_entities(query, label): partial, misspelled or descriptive names.
- top_entities(label): "most mentioned" questions.
- list_labels(): which entity types exist and how many mentions each has.
Reply in one or two friendly sentences that acknowledge the question. Use only names and \
counts returned by the tools. In `entities`, list just the entities your reply talks \
about, including any the user asked about that are not in the store (mention_count 0).\
"""

_NUMBER_WORDS = {
    "zero": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}  # fmt: skip
_QUOTED_COUNT = re.compile(
    r"\b(\d+|" + "|".join(_NUMBER_WORDS) + r")\s+(?:times?|mentions?|occurrences?)\b"
    r"|\b(once|twice|thrice)\b",
    re.IGNORECASE,
)
_ADVERB_COUNTS = {"once": 1, "twice": 2, "thrice": 3}


@dataclass
class AskDeps:
    store: EntityStore
    # Every count any tool returned during this run: the only counts a reply may quote.
    seen_counts: set[int] = field(default_factory=set)

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
def list_labels(ctx: RunContext[AskDeps]) -> list[dict[str, object]]:
    """Entity types in the store with how many entities and mentions each has."""
    summary = ctx.deps.store.label_summary()
    ctx.deps.seen_counts.update(n for s in summary for n in (s.entities, s.mentions))
    return [{"label": s.label, "entities": s.entities, "mentions": s.mentions} for s in summary]


@agent.output_validator
def ground_answer(ctx: RunContext[AskDeps], answer: Answer) -> Answer:
    """Replace every cited entity with the store's record; reject replies quoting counts
    that no tool returned, so the model rewrites them from the store."""
    grounded = ground_mentions(ctx.deps.store, answer.entities)
    allowed = ctx.deps.seen_counts | {mention.mention_count for mention in grounded}
    unsupported = sorted(set(quoted_counts(answer.reply)) - allowed)
    if unsupported:
        facts = "; ".join(f"{m.name}: {m.mention_count}" for m in grounded) or "none cited"
        raise ModelRetry(
            f"The reply quotes {unsupported} mentions, but no tool returned those counts. "
            f"Use only counts from the tools (cited entities: {facts})."
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


def quoted_counts(text: str) -> list[int]:
    """Counts a reply states as mentions: "7 times", "two mentions", "once", ..."""
    counts = []
    for match in _QUOTED_COUNT.finditer(text):
        number, adverb = match.groups()
        if adverb:
            counts.append(_ADVERB_COUNTS[adverb.lower()])
        elif number.isdigit():
            counts.append(int(number))
        else:
            counts.append(_NUMBER_WORDS[number.lower()])
    return counts


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
    """Answer ``question`` from the entity store only. Never opens the source document."""
    if not isinstance(store, EntityStore):
        store = EntityStore.load(store)
    result = agent.run_sync(
        question,
        model=resolve_model(model),
        deps=AskDeps(store),
        usage_limits=UsageLimits(request_limit=_REQUEST_LIMIT),
    )
    return result.output


def _clamp(limit: int) -> int:
    return max(1, min(limit, MAX_TOOL_RESULTS))
