from concurrent.futures import ThreadPoolExecutor
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import cast

from sift.api.package import PACKAGE
from sift.interfaces.cli.entries import ENTRIES
from tests.surface.smoke import (
    Answer,
    Done,
    Smoke,
    answer,
    answering,
    coloured,
    finding,
    hits_of,
    ids,
    indexed,
    listed,
    ok,
    unexercised,
)
from tests.surface.smoke_options import SEARCHED, options_run
from tests.surface.smoke_runs import documents_run, imported_run, lent_run, models_run, records_run, served_run
from tests.tiny_model import tiny_model, tiny_reranker


def narrowed_searches_run(smoke: Smoke) -> None:
    options = ("--path", "sub/*", "-f", "year!=1999", "--fields", "*", "--sort=-year", "--json")
    smoke.checked(
        "search",
        "brush OR learning",
        *options,
        holds=lambda done: [(hit["id"], hit["fields"]) for hit in hits_of(done)] == [("notes/sub/brush.md", {})],
    )
    smoke.checked("get", "notes/brush.md", "--fields", "*", holds=lambda done: done.stdout == "# Brush\n")


def _status_table(done: Done) -> tuple[list[str], list[str], list[str]]:
    header, row = done.stdout.split("\n")[:2]
    return header.split()[-2:], row.split()[-6:-4], row.split()[-2:]


def status_run(smoke: Smoke) -> None:
    columns = (["links", "unresolved"], ["3", "3"], ["3", "1"])
    smoke.checked("status", holds=lambda done: ok(done) and _status_table(done) == columns)
    smoke.checked(
        "status", "notes", "--unreadable", "--json", holds=lambda done: '"unreadable_files": []' in done.stdout
    )
    smoke.checked("--json", "status", holds=lambda done: ok(done) and sorted(answer(done)) == ["collections", "home"])


def links_run(smoke: Smoke) -> None:
    lines = [
        "     2  see: brush  ->  notes/sub/brush.md",
        "     6  brush  ->  notes/sub/brush.md",
        "     6  Gone  ->  unresolved",
    ]
    smoke.checked(
        "links", "notes/rl.md", "--direction", "both", holds=lambda done: done.stdout.split("\n")[1:4] == lines
    )
    smoke.checked("links", "notes/sub/brush.md", "--direction", "in", holds=lambda done: "  notes/rl.md" in done.stdout)
    smoke.checked("unresolved", holds=lambda done: done.stdout == "1  Gone  notes/rl.md\n")
    smoke.checked(
        "mentions",
        "notes/sub/brush.md",
        "--json",
        "--compact",
        holds=finding("notes/huge.txt"),
    )
    linking = ("search", "learning OR brush", "--json", "--compact")
    smoke.checked(*linking, "--links-to", "brush", holds=finding("notes/rl.md"))
    smoke.checked(*linking, "--linked-from", "notes/rl.md", holds=finding("notes/sub/brush.md"))
    smoke.checked(*linking, "-f", "see=[[sub/brush]]", holds=finding("notes/rl.md"))


def notes_in(notes: Path) -> Path:
    (notes / "sub").mkdir(parents=True)
    (notes / "rl.md").write_bytes(
        b'---\nsee: "[[brush]]"\n---\n# Reward shaping\n\nSpeeds up learning. See [[brush]], [[Gone]].\n'
    )
    (notes / "sub" / "brush.md").write_bytes(b"# Brush\n")
    (notes / "huge.txt").write_bytes(b"A brush stroke.")
    return notes


def _named(done: Done) -> object:
    return cast("list[Answer]", cast("list[Answer]", answer(done)["collections"])[0]["named"])[0]["state"]


def files_run(smoke: Smoke, notes: Path) -> None:
    smoke.checked("index", "notes", "--json", holds=lambda done: done.stdout.count('"added": 0') == 1)
    smoke.checked(
        "index",
        "--file",
        str(notes / "rl.md"),
        "--file",
        "notes/gone.md",
        holds=lambda done: (
            (done.returncode, done.stderr.splitlines()[-2:])
            == (0, [" file  notes/rl.md: unchanged", " file  notes/gone.md: missing"])
        ),
    )
    smoke.checked("index", "--file", "notes/sub/brush.md", "--json", holds=lambda done: _named(done) == "unchanged")
    smoke.checked(
        "index",
        "--file",
        "brush.md",
        "--file",
        "../gone.md",
        cwd=notes / "sub",
        holds=lambda done: (
            done.stderr.splitlines()[-2:] == [" file  notes/sub/brush.md: unchanged", " file  notes/gone.md: missing"]
        ),
    )


def plain_run(smoke: Smoke, notes: Path) -> None:
    indexed(smoke, notes)
    files_run(smoke, notes)
    status_run(smoke)
    smoke.checked(
        "get", "notes/brush.md", holds=lambda done: (done.returncode, done.stdout, done.stderr) == (0, "# Brush\n", "")
    )
    smoke.checked(
        "get",
        str(notes / "rl.md"),
        "notes/sub/brush.md",
        "--lines",
        "1",
        holds=lambda done: done.stdout == "==> notes/rl.md <==\n---\n\n==> notes/sub/brush.md <==\n# Brush\n",
    )
    smoke.checked(
        "search",
        "learning",
        "-k",
        "1",
        holds=lambda done: (
            (done.returncode, done.stdout.split("\n")[1]) == (0, " " * 9 + "notes/rl.md:4-6")
            and done.stderr.startswith("1 result  \N{MIDDLE DOT}  keyword  \N{MIDDLE DOT}  1 collection")
        ),
    )
    smoke.checked("search", '"speeds up"', "--json", holds=finding("notes/rl.md"))
    smoke.checked(
        "search",
        "speed*",
        "--budget",
        "5",
        "--json",
        holds=lambda done: (answer(done)["partial"], ids(done)) == (False, ["notes/rl.md"]),
    )
    narrowed_searches_run(smoke)
    links_run(smoke)
    smoke.checked(
        "get",
        "notes/gone.md",
        holds=lambda done: (
            (done.returncode, done.stdout) == (1, "")
            and done.stderr.startswith("error  no note or document 'notes/gone.md'")
        ),
    )
    assert smoke.ran("frobnicate").returncode == 2
    smoke.checked("--help", holds=lambda done: done.stdout.startswith("usage: sift"))
    smoke.checked("search", "--help", holds=lambda done: done.stdout.startswith("usage: sift search"))
    smoke.checked("--version", holds=lambda done: done.stdout == f"sift {version(PACKAGE)}\n")
    smoke.checked(
        "links",
        "notes/rl.md",
        "--direction",
        "out",
        "--depth",
        "2",
        "--field",
        "see",
        "-k",
        "5",
        "--json",
        holds=lambda done: len(listed(done, "out")) == 1,
    )
    smoke.checked(
        "unresolved", "-c", "notes", "-k", "5", "--json", holds=lambda done: len(listed(done, "targets")) == 1
    )
    smoke.checked("index", "notes", "--color", "always", holds=lambda done: "\x1b[" in done.stderr)
    smoke.checked(
        "index", "notes", "--color", "never", holds=lambda done: done.returncode == 0 and "\x1b[" not in done.stderr
    )
    catalog_run(smoke, notes)


def catalog_run(smoke: Smoke, notes: Path) -> None:
    home = smoke.home
    smoke.checked(
        "add",
        str(notes),
        "--title-from",
        "name",
        "--no-agents",
        "--no-follow-links",
        "--json",
        holds=lambda done: (done.returncode, answer(done)["state"], answer(done)["retitled"]) == (0, "changed", 2),
    )
    smoke.checked(
        "add",
        str(notes),
        "--agents",
        holds=lambda done: (done.returncode, "agent  notes: seen by agents" in done.stderr) == (0, True),
    )
    smoke.checked("deactivate", "notes", "--json", holds=answering("changed", ["notes"]))
    refused = smoke.ran("search", "learning")
    assert (refused.returncode, refused.stderr.split("\n")[:2]) == (
        2,
        ["error  no collection is active", "  fix  activate a collection by its name"],
    )
    smoke.checked("activate", "notes", "--json", holds=answering("changed", ["notes"]))
    kept = str((home / "kept" / "notes.sqlite").resolve())
    smoke.checked(
        "move",
        "notes",
        str(home / "kept"),
        "--json",
        holds=lambda done: (done.returncode, [store["new"] for store in listed(done, "moved")]) == (0, [kept]),
    )
    back = str((home / "back" / "notes.sqlite").resolve())
    smoke.checked(
        "move",
        str(home / "back"),
        "--json",
        holds=lambda done: [store["new"] for store in listed(done, "moved")] == [back],
    )
    smoke.checked("search", "learning", "--json", holds=finding("notes/rl.md"))
    smoke.checked("rm", "notes", "--json", holds=lambda done: (done.returncode, answer(done)["documents"]) == (0, 3))
    assert (list((home / "kept").iterdir()), list((home / "back").iterdir())) == ([], [])
    logs = sorted((home / "logs").iterdir())
    assert {log.name.split("-20", 1)[0] for log in logs} == {"activate", "add", "deactivate", "index", "move", "rm"}
    (rm,) = (log for log in logs if log.name.startswith("rm-"))
    assert rm.read_text(encoding="utf-8").endswith(" INFO sift: ended with exit code 0\n")


def test_every_command_runs_as_a_process_from_a_folder_to_its_documents(tmp_path: Path) -> None:
    notes = notes_in(tmp_path / "Notizen Müller")
    model, reranker = str(tiny_model(tmp_path / "models")), str(tiny_reranker(tmp_path / "models"))
    (tmp_path / "more").mkdir()
    smokes = {
        "plain": Smoke(tmp_path / "plain"),
        "models": Smoke(tmp_path / "models-home", model, reranker),
        "records": Smoke(tmp_path / "records", model),
        "documents": Smoke(tmp_path / "documents"),
        "served": Smoke(tmp_path / "served"),
        "lent": Smoke(tmp_path / "lent"),
        "imported": Smoke(tmp_path / "imported"),
        "worker": Smoke(tmp_path / "worker", model, reranker),
    }
    runs = (
        partial(plain_run, smokes["plain"], notes),
        partial(models_run, smokes["models"], notes),
        partial(records_run, smokes["records"], notes_in(tmp_path / "Papiere")),
        partial(documents_run, smokes["documents"], tmp_path / "Dokumente"),
        partial(served_run, smokes["served"], notes, tmp_path / "more"),
        partial(lent_run, smokes["lent"], smokes["worker"], notes),
        partial(imported_run, smokes["imported"], notes_in(tmp_path / "Importiert")),
    )
    with ThreadPoolExecutor(len(runs)) as pool:
        for running in [pool.submit(run) for run in runs]:
            running.result()
    ran = [argv for smoke in smokes.values() for argv in smoke.exercised]
    assert (unexercised(ran, ENTRIES.keys() - SEARCHED.keys()), coloured(ran)) == ([], True)


def test_every_flag_of_a_search_has_its_effect(tmp_path: Path) -> None:
    models = tmp_path / "models"
    smoke = Smoke(tmp_path / "home", str(tiny_model(models)), str(tiny_reranker(models)))
    options_run(smoke, tmp_path / "Optionen")
    assert unexercised(smoke.exercised, SEARCHED) == []
