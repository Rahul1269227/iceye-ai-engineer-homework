"""Pydantic models: the entity store on disk and the grounded answer returned by ``ask``."""

from __future__ import annotations

from pydantic import BaseModel, Field

MAX_ANSWER_ENTITIES = 10


class StoredEntity(BaseModel):
    """One distinct entity found in the document."""

    name: str = Field(min_length=1, description="Canonical surface form")
    label: str = Field(min_length=1, description="Entity type, e.g. 'organization'")
    count: int = Field(ge=1, description="Number of mentions in the document")
    aliases: list[str] = Field(default_factory=list, description="Other surface forms")


class EntityStoreFile(BaseModel):
    """The JSON document written by ``extract`` and read by ``ask``."""

    source: str | None = None
    model: str | None = None
    labels: list[str] = Field(default_factory=list)
    entities: list[StoredEntity]


class EntityMention(BaseModel):
    """An entity used in a reply, with its label and count as recorded in the store."""

    name: str = Field(description="Entity name; the store's canonical name when found")
    label: str | None = Field(description="Entity type from the store; null when not in it")
    mention_count: int = Field(ge=0, description="Mentions in the document; 0 = not mentioned")
    in_store: bool = Field(description="Whether the entity was found in the store")


class Answer(BaseModel):
    """Structured reply to one question about the document's entities."""

    reply: str = Field(description="Short conversational reply that acknowledges the question")
    # Not a schema limit: a longer list from the model is cut to MAX_ANSWER_ENTITIES by
    # ask's validator, where a validation error would only cost a retry.
    entities: list[EntityMention] = Field(
        default_factory=list,
        description=f"Only the entities the reply talks about (at most {MAX_ANSWER_ENTITIES}), "
        "including names the user asked about that are not in the store (mention_count 0). "
        "Never the full entity list.",
    )
