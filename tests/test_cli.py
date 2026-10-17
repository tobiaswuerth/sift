import io
import json
import runpy
import sqlite3
import sys

import pytest

from sift import cli, store


def run(capsys, *argv):
    cli.main(list(argv))
    return capsys.readouterr()


@pytest.fixture
def cli_corpus(notes, home, fake_models, capsys):
    run(capsys, "add", str(notes))
    run(capsys, "index")
    run(capsys, "embed")
    return notes


def test_add_index_embed(notes, home, fake_models, capsys):
    assert "run `sift index` next" in run(capsys, "add", str(notes), "--name", "n").err
    assert "3 added" in run(capsys, "index").err
    assert "embedded 3 chunks" in run(capsys, "embed").err
    assert "embedded 3 chunks" in run(capsys, "embed", "--reset").err


@pytest.mark.parametrize("command", ["search", "vsearch", "query"])
def test_search_commands(cli_corpus, capsys, command):
    out = run(capsys, command, "reward shaping", "-k", "1").out
    assert "notes/rl.md:1-3  Reward shaping" in out
    hits = json.loads(run(capsys, command, "brush", "--json", "-c", "notes", "-f", "categories=cs.RO").out)
    assert [h["id"] for h in hits] == ["notes/robots/brush.md"]
    hits = json.loads(run(capsys, command, "brush", "--json", "--path", "robots/*", "-f", "year=2024").out)
    assert [h["id"] for h in hits] == ["notes/robots/brush.md"]


def test_get_status_rm(cli_corpus, capsys):
    assert run(capsys, "get", "notes/rl.md").out.startswith("# Reward shaping")
    status = json.loads(run(capsys, "status").out)
    assert status["collections"][0]["embedded"] == 3
    run(capsys, "rm", "notes")
    assert json.loads(run(capsys, "status").out)["collections"] == []


@pytest.mark.parametrize(
    "argv, message",
    [
        (["search", "x", "-f", "novalue"], "KEY=VALUE"),
        (["search", "x", "-c", "missing"], "error: no collection named 'missing'"),
        (["search", "x", "-k", "0"], "error: k must be"),
        (["get", "missing/x.md"], "error: unknown document"),
        (["vsearch", "x"], "error: no embeddings yet"),
        (["rm", "missing"], "no collection named 'missing'"),
    ],
)
def test_errors_are_messages_not_tracebacks(home, capsys, argv, message):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert message in str(exc.value)


def test_database_errors_are_messages(home, monkeypatch):
    def locked(*args):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(store, "index", locked)
    with pytest.raises(SystemExit, match="error: database is locked"):
        cli.main(["index"])


def test_ctrl_c_exits_quietly(home, monkeypatch):
    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(store, "index", interrupted)
    with pytest.raises(SystemExit) as exc:
        cli.main(["index"])
    assert exc.value.code == 130


def test_non_utf8_stdout_is_switched_to_utf8(cli_corpus, monkeypatch):
    class LegacyConsole(io.StringIO):
        encoding = "cp1252"
        reconfigured = None

        def reconfigure(self, **kwargs):
            self.reconfigured = kwargs

    out = LegacyConsole()
    monkeypatch.setattr(sys, "stdout", out)
    cli.main(["status"])
    assert out.reconfigured == {"encoding": "utf-8"}


def test_serve(home, monkeypatch):
    calls = []
    monkeypatch.setattr("sift.server.serve", lambda *args: calls.append(args))
    cli.main(["serve", "--port", "9000", "--stdio"])
    assert calls == [("127.0.0.1", 9000, True)]


def test_version_and_module_entry_point(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["sift", "--version"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("sift", run_name="__main__")
    assert exc.value.code == 0
    assert capsys.readouterr().out.startswith("sift ")
