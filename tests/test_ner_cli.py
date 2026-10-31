"""ner-chat CLI ask path (no extract / no Hub)."""

from pydantic_ai import models

from iceye_homework.ner import Answer
from iceye_homework.ner.cli import main


def test_ask_json(store_path, monkeypatch, capsys):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NER_CHAT_MODEL"):
        monkeypatch.delenv(key, raising=False)

    argv = [
        "ask",
        "How many times are ICEYE and NASA mentioned?",
        "--store",
        str(store_path),
        "--model",
        "offline",
        "--json",
    ]
    assert main(argv) == 0
    answer = Answer.model_validate_json(capsys.readouterr().out)
    assert any(m.name == "ICEYE" and m.mention_count == 7 for m in answer.entities)


def test_ask_test_model_json(store_path, monkeypatch, capsys):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "NER_CHAT_MODEL"):
        monkeypatch.delenv(key, raising=False)

    argv = ["ask", "Is NASA mentioned?", "--store", str(store_path), "--model", "test", "--json"]
    assert main(argv) == 0
    Answer.model_validate_json(capsys.readouterr().out)


def test_ask_human_format(store_path, monkeypatch, capsys):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    argv = ["ask", "Is SAR mentioned?", "--store", str(store_path), "--model", "offline"]
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "SAR" in out


def test_ask_missing_store(tmp_path, capsys):
    assert main(["ask", "hi", "--store", str(tmp_path / "no.json"), "--model", "offline"]) == 2
    assert "error:" in capsys.readouterr().err


def test_ask_missing_model(store_path, monkeypatch, capsys):
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    monkeypatch.delenv("NER_CHAT_MODEL", raising=False)
    assert main(["ask", "Is ICEYE mentioned?", "--store", str(store_path)]) == 2
    assert "error:" in capsys.readouterr().err


def test_extract_missing_file(tmp_path, capsys):
    assert main(["extract", str(tmp_path / "missing.pdf")]) == 2
    assert "not a file" in capsys.readouterr().err
