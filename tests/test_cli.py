"""The `sift` command, run as users run it: every subcommand, its options, and errors as messages."""

import io
import json
import runpy
import sqlite3
import subprocess
import sys

import pytest
from mcp.server.mcpserver import MCPServer

from sift import cli, indexing


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


@pytest.mark.parametrize(
    "argv, target",
    [
        (["embed"], "sift.embed.embed_pending"),
        (["vsearch", "x"], "sift.search.Index.vsearch"),
        (["query", "x"], "sift.search.Index.query"),
        (["similar", "notes/x.md"], "sift.search.Index.similar"),
        (["serve"], "sift.server.serve"),
        (["worker"], "sift.worker.serve"),
    ],
)
def test_slow_commands_say_they_started_before_loading_anything(home, monkeypatch, capsys, argv, target):
    # Loading PyTorch can take long, longer still with Intel's GPU build: without a word first, sift looked hung.
    said = []
    monkeypatch.setattr(target, lambda *args, **kwargs: said.append(capsys.readouterr().err) or [])
    cli.main(argv)
    assert said[0].startswith(f"sift {argv[0]}: starting")


def test_the_cli_starts_with_the_standard_library_alone():
    # CI installs the wheel without dependencies and runs `sift --version`; importing NumPy at the top of the CLI
    # broke that. It also keeps the first line quick.
    code = "import sys; sys.modules.update(numpy=None, torch=None); from sift import cli; cli.main(['--version'])"
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    assert run.stdout.startswith("sift ")


@pytest.mark.parametrize("command", ["vsearch", "query", "similar"])
def test_searching_through_a_worker_says_nothing_of_pytorch(home, monkeypatch, capsys, command, configure):
    configure(remote="https://t.p@workstation:8766")
    monkeypatch.setattr(f"sift.search.Index.{command}", lambda *args, **kwargs: [])
    cli.main([command, "x"])
    assert "PyTorch" not in capsys.readouterr().err  # it doesn't load it


@pytest.mark.parametrize("command", ["search", "vsearch", "query"])
def test_search_commands(cli_corpus, capsys, command):
    out = run(capsys, command, "reward shaping", "-k", "1").out
    assert "notes/rl.md:1-3  Reward shaping" in out
    hits = json.loads(run(capsys, command, "brush", "--json", "-c", "notes", "-f", "categories=cs.RO").out)
    assert [h["id"] for h in hits] == ["notes/robots/brush.md"]
    hits = json.loads(run(capsys, command, "brush", "--json", "--path", "robots/*", "-f", "year=2024").out)
    assert [h["id"] for h in hits] == ["notes/robots/brush.md"]


def test_search_command_fields_and_compact(cli_corpus, capsys):
    out = run(capsys, "search", "brush", "--fields", "year,categories").out
    assert 'year: 2024  categories: ["cs.RO", "cs.GR"]' in out
    hits = json.loads(run(capsys, "query", "brush", "-k", "1", "--json", "--compact", "--fields", "year").out)
    assert [(h["id"], h["fields"], "text" in h) for h in hits] == [("notes/robots/brush.md", {"year": 2024}, False)]
    assert "Kubelka" not in run(capsys, "search", "brush", "--compact").out


@pytest.mark.parametrize("command", ["search", "vsearch", "query", "similar"])
def test_search_commands_pass_every_option_on(cli_corpus, capsys, monkeypatch, command):
    from sift.search import Index

    calls = []
    monkeypatch.setattr(Index, command, lambda self, *args, **kwargs: calls.append((args, kwargs)) or [])
    run(capsys, command, "brush", "-k", "3", "-c", "notes", "--path", "*.md", "-f", "year=2024", "--fields", "year")
    run(capsys, command, "brush", "--compact", "--all-passages", "--sort=-year")
    assert calls == [
        (
            ("brush", 3),
            {
                "collection": "notes",
                "path": "*.md",
                "filters": {"year": 2024},
                "fields": ["year"],
                "compact": False,
                "all_passages": False,
                "sort": None,
            },
        ),
        (
            ("brush", 10),
            {
                "collection": None,
                "path": None,
                "filters": {},
                "fields": None,
                "compact": True,
                "all_passages": True,
                "sort": "-year",
            },
        ),
    ]


def test_similar_command(cli_corpus, capsys):
    out = run(capsys, "similar", "notes/brush.md", "-k", "1").out
    assert out.count("notes/") == 1
    assert "notes/robots/brush.md" not in out


def test_search_command_says_when_nothing_matches(cli_corpus, capsys):
    result = run(capsys, "search", "quantum")
    assert (result.out, result.err.strip()) == ("", "no matches")


@pytest.mark.parametrize(
    "pairs, filters",
    [
        (["year=2024", "tag=a=b"], {"year": 2024, "tag": "a=b"}),
        (["created>=2024-06", "created<2025"], {"created": {">=": "2024-06", "<": 2025}}),
        (["created<=2024", "created>2023-06"], {"created": {"<=": 2024, ">": "2023-06"}}),
        (["year>=2020", "year=2024"], {"year": {">=": 2020, "=": 2024}}),
        (["title=a>b"], {"title": "a>b"}),
        (["categories=cs.RO", "categories=cs.CV"], {"categories": ["cs.RO", "cs.CV"]}),  # any of them
        (["categories!=cs.CV"], {"categories": {"not": "cs.CV"}}),
        (["categories!=cs.CV", "categories!=cs.GR"], {"categories": {"not": ["cs.CV", "cs.GR"]}}),  # none of them
        (["year>=2020", "year=2024", "year=2025"], {"year": {">=": 2020, "=": [2024, 2025]}}),
        (['tags=["a","b"]', "tags=c"], {"tags": ["a", "b", "c"]}),
        (["journal-ref=Nature", "date created>=2024"], {"journal-ref": "Nature", "date created": {">=": 2024}}),
    ],
)
def test_filters_on_the_command_line(pairs, filters):
    assert cli._filters(pairs) == filters


@pytest.mark.parametrize("pair", ["year", "-->", "=1", " =1"])
def test_filters_need_a_field_name_and_an_operator(pair):
    with pytest.raises(SystemExit, match="filter must be"):
        cli._filters([pair])


def test_bench_command(cli_corpus, capsys, tmp_path):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps([{"query": "reward shaping reinforcement", "relevant": ["notes/rl.md"]}]), encoding="utf-8"
    )
    result = run(capsys, "bench", str(path))
    assert "sift bench: starting" in result.err  # it runs vsearch and query: PyTorch may load
    lines = result.out.splitlines()
    assert lines[0].split() == ["mode", "nDCG@10", "MRR@10", "Recall@10", "ms/query"]
    assert [line.split()[:4] for line in lines[1:]] == [
        [mode, "1.000", "1.000", "1.000"] for mode in ("search", "vsearch", "query")
    ]


def test_get_lines_command(cli_corpus, capsys):
    assert run(capsys, "get", "notes/rl.md", "--lines", "1").out == "# Reward shaping\n"


def test_get_with_fields_command(cli_corpus, capsys):
    out = run(capsys, "get", "notes/brush.md", "--lines", "7", "--fields", "year,categories").out
    assert out == 'year: 2024\ncategories: ["cs.RO", "cs.GR"]\n\n# Brush stroke planning\n'


def test_get_status_rm(cli_corpus, capsys):
    assert run(capsys, "get", "notes/rl.md").out.startswith("# Reward shaping")
    status = json.loads(run(capsys, "status").out)
    assert status["collections"][0]["embedded"] == 3
    err = run(capsys, "rm", "notes").err
    assert "removed collection 'notes' and its 3 documents from the index" in err
    assert "files on disk are untouched" in err
    assert json.loads(run(capsys, "status").out)["collections"] == []


def test_rm_reports_a_single_document(tmp_path, home, capsys):
    folder = tmp_path / "one"
    folder.mkdir()
    (folder / "only.md").write_text("# Only\n", encoding="utf-8")
    run(capsys, "add", str(folder))
    run(capsys, "index")
    assert "and its 1 document from" in run(capsys, "rm", "one").err


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
def test_errors_are_messages_not_tracebacks(home, argv, message):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert message in str(exc.value)


def test_database_errors_are_messages(home, monkeypatch):
    def locked(*args):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(indexing, "index", locked)
    with pytest.raises(SystemExit, match="error: database is locked"):
        cli.main(["index"])


def test_ctrl_c_exits_quietly(home, monkeypatch):
    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(indexing, "index", interrupted)
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


@pytest.mark.parametrize(
    "error, code", [(KeyboardInterrupt(), 130), (OSError("address in use"), "error: address in use")]
)
def test_serve_stops_without_a_traceback(cli_corpus, monkeypatch, error, code):
    async def failing(self, **kwargs):
        raise error

    monkeypatch.setattr(MCPServer, "run_stdio_async", failing)
    with pytest.raises(SystemExit) as exc:
        cli.main(["serve", "--stdio"])
    assert exc.value.code == code


def test_unusable_home_is_a_message(tmp_path, configure):
    home = tmp_path / "home"
    home.write_text("a file, not a folder", encoding="utf-8")
    configure(home=home)
    with pytest.raises(SystemExit, match=r"^error: "):
        cli.main(["status"])


def test_add_with_chosen_fields(tmp_path, home, capsys):
    folder = tmp_path / "qa"
    folder.mkdir()
    (folder / "qa.jsonl").write_text(json.dumps({"key": "q1", "q": "Why?", "a": "Because."}) + "\n", encoding="utf-8")
    run(capsys, "add", str(folder), "--id", "key", "--title", "q", "--text", "q,a")
    run(capsys, "index")
    assert run(capsys, "get", "qa/qa/q1").out == "# Why?\n\nWhy?\n\nBecause.\n"  # a final newline for the shell
