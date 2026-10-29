"""``ner-chat``: ``extract`` entities from a document into a store, then ``ask`` about them."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from kreuzberg import KreuzbergError
from pydantic_ai.exceptions import AgentRunError, UserError

from .agent import MODEL_ENV, OFFLINE_MODEL, ask
from .extract import (
    DEFAULT_LABELS,
    DEFAULT_MODEL,
    DEFAULT_THRESHOLD,
    OCR_BACKENDS,
    extract_entities,
)
from .models import Answer
from .store import EntityStore, write_store

DEFAULT_STORE = Path("entities.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ner-chat",
        description="Detect named entities in a document once (extract), then answer "
        "questions about them from the entity store (ask).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    extract = commands.add_parser(
        "extract", help="extract text (Kreuzberg), detect entities (GLinker), write the store"
    )
    extract.add_argument("document", type=Path, help="PDF, image, Office file, HTML, ...")
    extract.add_argument("--store", type=Path, default=DEFAULT_STORE, help="output JSON store")
    extract.add_argument(
        "--labels", nargs="+", default=list(DEFAULT_LABELS), help="entity types to detect"
    )
    extract.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    extract.add_argument("--ner-model", default=DEFAULT_MODEL, help="GLiNER model id")
    extract.add_argument("--ocr-backend", choices=OCR_BACKENDS, default="paddleocr")
    extract.add_argument("--force-ocr", action="store_true", help="OCR even text-based PDFs")
    extract.add_argument("--device", default="cpu", help="torch device: cpu, cuda, mps")

    ask_cmd = commands.add_parser("ask", help="answer one question from the store")
    ask_cmd.add_argument("question")
    ask_cmd.add_argument("--store", type=Path, default=DEFAULT_STORE, help="JSON store to read")
    ask_cmd.add_argument(
        "--model",
        help="Pydantic AI model, e.g. openai:gpt-4.1-mini or anthropic:claude-sonnet-4-5; "
        f"'{OFFLINE_MODEL}' (rule-based) and 'test' (Pydantic AI TestModel) need no key. "
        f"Default: ${MODEL_ENV}",
    )
    ask_cmd.add_argument("--json", action="store_true", help="print the structured answer")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(format="%(message)s")
    match args.command:
        case "extract":
            return _extract(args)
        case "ask":
            return _ask(args)
        case _:
            raise AssertionError(f"unknown command {args.command!r}")


def _extract(args: argparse.Namespace) -> int:
    if not args.document.is_file():
        print(f"error: {args.document} is not a file", file=sys.stderr)
        return 2
    start = time.perf_counter()
    try:
        data = extract_entities(
            args.document,
            labels=args.labels,
            threshold=args.threshold,
            model=args.ner_model,
            ocr_backend=args.ocr_backend,
            force_ocr=args.force_ocr,
            device=args.device,
        )
    except KreuzbergError as exc:
        print(f"error: cannot extract text from {args.document}: {exc}", file=sys.stderr)
        return 2
    write_store(data, args.store)
    mentions = sum(entity.count for entity in data.entities)
    print(
        f"{len(data.entities)} entities ({mentions} mentions) from {args.document} "
        f"in {time.perf_counter() - start:.1f}s -> {args.store}",
        file=sys.stderr,
    )
    return 0


def _ask(args: argparse.Namespace) -> int:
    try:
        store = EntityStore.load(args.store)
    except (OSError, ValueError) as exc:
        print(f"error: cannot read store {args.store}: {exc}", file=sys.stderr)
        return 2
    try:
        answer = ask(args.question, store, args.model)
    except (ValueError, UserError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except AgentRunError as exc:
        print(f"error: the model run failed: {exc}", file=sys.stderr)
        return 1
    print(answer.model_dump_json(indent=2) if args.json else _format(answer))
    return 0


def _format(answer: Answer) -> str:
    lines = [answer.reply]
    if answer.entities:
        width = max(len(mention.name) for mention in answer.entities)
        lines.append("")
        for mention in answer.entities:
            if mention.in_store:
                count = mention.mention_count
                detail = f"{mention.label}, {count} mention{'' if count == 1 else 's'}"
            else:
                detail = "not mentioned (0)"
            lines.append(f"  {mention.name:<{width}}  {detail}")
    return "\n".join(lines)
