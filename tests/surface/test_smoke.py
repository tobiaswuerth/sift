import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import cast

from sift.interfaces.cli.arguments import Argument, How, arguments
from sift.interfaces.cli.main import ENTRIES
from sift.interfaces.server.endpoint import found
from tests.mails import WORD, mail_bytes, message, sample
from tests.surface import process
from tests.tiny_model import tiny_model, tiny_reranker

RAN: list[tuple[str, ...]] = []
FLAGGED = (How.OPTION, How.SWITCH, How.TOGGLE, How.READ)
LISTED = (How.ANY, How.SOME)


def sift(
    home: Path, *args: str, model: str = "", reranker: str = "", cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    RAN.append(args)
    return process.sift(home, *args, model=model, reranker=reranker, cwd=cwd)


def _forms(argument: Argument) -> frozenset[str]:
    return frozenset(name for name in (argument.flag.long, argument.flag.short) if name is not None)


def _given(forms: frozenset[str], argvs: list[tuple[str, ...]]) -> bool:
    return any(word.split("=", 1)[0] in forms for argv in argvs for word in argv)


def unexercised(ran: list[tuple[str, ...]]) -> list[str]:
    missing: list[str] = []
    for name, entry in ENTRIES.items():
        argvs = [argv for argv in ran if argv[:1] == (name,)]
        for argument in arguments(entry.request):
            if argument.how in FLAGGED and not _given(_forms(argument), argvs):
                missing.append(f"{name} {min(_forms(argument), key=len)}")
            if argument.how in LISTED and not any(len(argv) > 1 and not argv[1].startswith("-") for argv in argvs):
                missing.append(f"{name} {argument.name.upper()}")
        if not _given(frozenset({"--json"}), argvs):
            missing.append(f"{name} --json")
    return missing if _given(frozenset({"--color"}), ran) else [*missing, "--color"]


type Answer = dict[str, object]


def answer(ran: subprocess.CompletedProcess[str]) -> Answer:
    return cast("Answer", json.loads(ran.stdout))


def listed(ran: subprocess.CompletedProcess[str], key: str) -> list[Answer]:
    return cast("list[Answer]", answer(ran)[key])


def hits_of(ran: subprocess.CompletedProcess[str]) -> list[dict[str, object]]:
    return cast("dict[str, list[dict[str, object]]]", json.loads(ran.stdout))["hits"]


def narrowed_searches_run(home: Path) -> None:
    options = ("--path", "sub/*", "-f", "year!=1999", "--fields", "*", "--sort=-year", "--json")
    narrowed = hits_of(sift(home, "search", "brush OR learning", *options))
    assert [(hit["id"], hit["fields"]) for hit in narrowed] == [("notes/sub/brush.md", {})]
    assert sift(home, "get", "notes/brush.md", "--fields", "*").stdout == "# Brush\n"


def status_run(home: Path) -> None:
    status = sift(home, "status")
    assert (status.returncode, status.stderr) == (0, "")
    header, row = status.stdout.split("\n")[:2]
    assert (header.split()[-2:], row.split()[-6:-4], row.split()[-2:]) == (
        ["links", "unresolved"],
        ["3", "3"],
        ["3", "1"],
    )
    listed = sift(home, "status", "notes", "--unreadable", "--json")
    assert '"unreadable_files": []' in listed.stdout
    assert json.loads(listed.stdout)


def similar_run(home: Path) -> None:
    alike = sift(home, "similar", "notes/rl.md", "--json", "--compact")
    assert (alike.returncode, alike.stderr) == (0, "")
    assert sorted(str(hit["id"]) for hit in hits_of(alike)) == ["notes/huge.txt", "notes/sub/brush.md"]


def links_run(home: Path) -> None:
    linked = sift(home, "links", "notes/rl.md", "--direction", "both")
    assert linked.stdout.split("\n")[1:4] == [
        "     2  see: brush  ->  notes/sub/brush.md",
        "     6  brush  ->  notes/sub/brush.md",
        "     6  Gone  ->  unresolved",
    ]
    assert sift(home, "unresolved").stdout == "1  Gone  notes/rl.md\n"
    mentioned = sift(home, "mentions", "notes/sub/brush.md", "--json", "--compact")
    assert [hit["id"] for hit in hits_of(mentioned)] == ["notes/huge.txt"]
    linking = ("search", "learning OR brush", "--json", "--compact")
    assert [hit["id"] for hit in hits_of(sift(home, *linking, "--links-to", "brush"))] == ["notes/rl.md"]
    assert [hit["id"] for hit in hits_of(sift(home, *linking, "--linked-from", "notes/rl.md"))] == [
        "notes/sub/brush.md"
    ]
    assert [hit["id"] for hit in hits_of(sift(home, *linking, "-f", "see=[[sub/brush]]"))] == ["notes/rl.md"]


def notes_in(notes: Path) -> Path:
    (notes / "sub").mkdir(parents=True)
    (notes / "rl.md").write_bytes(
        b'---\nsee: "[[brush]]"\n---\n# Reward shaping\n\nSpeeds up learning. See [[brush]], [[Gone]].\n'
    )
    (notes / "sub" / "brush.md").write_bytes(b"# Brush\n")
    (notes / "huge.txt").write_bytes(b"A brush stroke.")
    return notes


def indexed(home: Path, notes: Path) -> None:
    added = sift(home, "add", str(notes), "--name", "notes")
    assert (added.returncode, added.stdout) == (0, "")
    assert added.stderr.split("\n")[0] == f"added  notes  {notes.resolve()}"
    done = sift(home, "index")
    assert (done.returncode, done.stdout) == (0, "")
    assert "  done  +3  ~0  -0  " in done.stderr


def plain_run(home: Path, notes: Path) -> None:
    indexed(home, notes)
    assert sift(home, "index", "notes", "--json").stdout.count('"added": 0') == 1
    named = sift(home, "index", "--file", str(notes / "rl.md"), "--file", "notes/gone.md")
    assert (named.returncode, named.stderr.splitlines()[-2:]) == (
        0,
        ["notes/rl.md: unchanged", "notes/gone.md: missing"],
    )
    asked = answer(sift(home, "index", "--file", "notes/sub/brush.md", "--json"))
    assert cast("list[Answer]", cast("list[Answer]", asked["collections"])[0]["named"])[0]["state"] == "unchanged"
    here = sift(home, "index", "--file", "brush.md", "--file", "../gone.md", cwd=notes / "sub")
    assert here.stderr.splitlines()[-2:] == ["notes/sub/brush.md: unchanged", "notes/gone.md: missing"]
    status_run(home)
    one = sift(home, "get", "notes/brush.md")
    assert (one.returncode, one.stdout, one.stderr) == (0, "# Brush\n", "")
    lines = sift(home, "get", str(notes / "rl.md"), "notes/sub/brush.md", "--lines", "1")
    assert lines.stdout == "==> notes/rl.md <==\n---\n\n==> notes/sub/brush.md <==\n# Brush\n"
    found = sift(home, "search", "learning", "-k", "1")
    assert (found.returncode, found.stdout.split("\n")[1]) == (0, " " * 9 + "notes/rl.md:4-6")
    assert found.stderr.startswith("1 result  \N{MIDDLE DOT}  keyword  \N{MIDDLE DOT}  1 collection")
    assert json.loads(sift(home, "search", '"speeds up"', "--json").stdout)["hits"][0]["id"] == "notes/rl.md"
    timed = cast("dict[str, object]", json.loads(sift(home, "search", "speed*", "--budget", "5", "--json").stdout))
    assert timed["partial"] is False
    assert cast("list[dict[str, object]]", timed["hits"])[0]["id"] == "notes/rl.md"
    narrowed_searches_run(home)
    links_run(home)
    missing = sift(home, "get", "notes/gone.md")
    assert (missing.returncode, missing.stdout) == (1, "")
    assert missing.stderr.startswith("error  notes/gone.md: unknown document")
    assert sift(home, "frobnicate").returncode == 2
    assert sift(home, "--help").stdout.startswith("usage: sift")
    linking = sift(
        home, "links", "notes/rl.md", "--direction", "out", "--depth", "2", "--field", "see", "-k", "5", "--json"
    )
    assert len(listed(linking, "out")) == 1
    unresolved = sift(home, "unresolved", "-c", "notes", "-k", "5", "--json")
    assert len(listed(unresolved, "targets")) == 1
    coloured = sift(home, "index", "notes", "--color", "always")
    assert "\x1b[" in coloured.stderr
    catalog_run(home, notes)


def catalog_run(home: Path, notes: Path) -> None:
    again = sift(home, "add", str(notes), "--title-from", "name", "--no-agents", "--no-follow-links", "--json")
    assert (again.returncode, answer(again)["state"], answer(again)["retitled"]) == (0, "changed", 2)
    seen = sift(home, "add", str(notes), "--agents")
    assert (seen.returncode, "agent  notes: seen by agents" in seen.stderr) == (0, True)
    assert answer(sift(home, "deactivate", "notes", "--json"))["changed"] == ["notes"]
    refused = sift(home, "search", "learning")
    assert (refused.returncode, refused.stderr.split("\n")[:2]) == (
        2,
        ["error  no collection is active", "  fix  run sift activate with a collection's name"],
    )
    assert answer(sift(home, "activate", "notes", "--json"))["changed"] == ["notes"]
    moved = sift(home, "move", "notes", str(home / "kept"), "--json")
    (store,) = cast("list[Answer]", answer(moved)["moved"])
    assert (moved.returncode, store["new"]) == (0, str((home / "kept" / "notes.sqlite").resolve()))
    assert sift(home, "search", "learning", "--json").returncode == 0
    removed = sift(home, "rm", "notes", "--json")
    assert (removed.returncode, answer(removed)["documents"]) == (0, 3)
    assert not list((home / "kept").iterdir())
    logs = sorted((home / "logs").iterdir())
    assert {log.name.split("-20", 1)[0] for log in logs} == {"activate", "add", "deactivate", "index", "move", "rm"}
    (rm,) = (log for log in logs if log.name.startswith("rm-"))
    assert rm.read_text(encoding="utf-8").endswith(" INFO sift: ended with exit code 0\n")


def models_run(home: Path, notes: Path, model: str, reranker: str) -> None:
    indexed(home, notes)
    embedded = sift(home, "embed", model=model)
    assert (embedded.returncode, embedded.stdout) == (0, "")
    assert "  embed          3 chunks" in embedded.stderr
    assert "  load           1 model " in embedded.stderr
    assert ("%|" in embedded.stderr, "it/s]" in embedded.stderr) == (False, False)
    near = sift(home, "vsearch", "learning", "-k", "1", "--json", model=model)
    assert len(cast("dict[str, list[object]]", json.loads(near.stdout))["hits"]) == 1
    assert "  load           1 model " in near.stderr
    similar_run(home)
    kept = sift(home, "vsearch", "learning", "--path", "sub/*", "--json", model=model)
    assert [hit["id"] for hit in hits_of(kept)] == ["notes/sub/brush.md"]
    asked = sift(home, "query", "reward shaping", "-k", "1", model=model, reranker=reranker)
    assert (asked.returncode, asked.stdout.split("\n")[1].strip().startswith("notes/")) == (0, True)
    assert asked.stderr.startswith("  load           2 models ")
    assert asked.stderr.splitlines()[-1].startswith("1 result  \N{MIDDLE DOT}  hybrid  \N{MIDDLE DOT}  1 collection")


def options_run(home: Path, notes: Path, model: str, reranker: str) -> None:
    indexed(home, notes)
    assert sift(home, "embed", model=model).returncode == 0
    narrowed = ("-k", "2", "-c", "notes", "--path", "*", "-f", "see!=x", "--fields", "*", "--sort=-see")
    linked = (
        "--links-to",
        "brush",
        "--linked-from",
        "notes/rl.md",
        "--all-passages",
        "--compact",
        "--adaptive",
        "--json",
    )
    for asked in (
        ("search", "brush OR learning", *narrowed, *linked, "--budget", "5"),
        ("vsearch", "learning", *narrowed, *linked, "--budget", "5"),
        ("query", "learning", *narrowed, *linked, "--candidates", "5", "--budget", "5"),
        ("similar", "notes/rl.md", *narrowed, *linked, "--budget", "5"),
        ("mentions", "notes/sub/brush.md", *narrowed, *linked, "--also", "Pinsel", "--budget", "5"),
    ):
        ran = sift(home, *asked, model=model, reranker=reranker)
        assert (asked[0], ran.returncode, ran.stderr) == (asked[0], 0, "")
        answered = cast("dict[str, int]", answer(ran))
        assert answered["searched"] + answered["ruled_out"] == 1
    for asked in (
        ("search", "brush OR learning"),
        ("vsearch", "learning"),
        ("query", "learning"),
        ("similar", "notes/rl.md"),
        ("mentions", "notes/sub/brush.md"),
    ):
        first, second = (
            sift(home, *asked, "-k", "1", "--offset", at, "--json", model=model, reranker=reranker) for at in "01"
        )
        assert (asked[0], first.returncode, second.returncode) == (asked[0], 0, 0)
        assert hits_of(first) != hits_of(second) or not hits_of(second)


def records_run(home: Path, notes: Path, model: str) -> None:
    indexed(home, notes)
    papers = notes / "papers"
    papers.mkdir()
    (papers / "p.jsonl").write_bytes(b'{"key": "p1", "headline": "Brush strokes", "body": "learning to paint"}\n')
    chosen = ("--id-field", "key", "--title-field", "headline", "--text-fields", "body", "--title-from", "heading")
    store = ("--store", str(home / "kept" / "papers.sqlite"), "--nested", "--follow-links")
    added = sift(home, "add", str(papers), "--name", "papers", *store, *chosen, "--json")
    assert (added.returncode, answer(added)["name"]) == (0, "papers")
    assert sift(home, "index", "papers", "--json").returncode == 0
    (record,) = listed(sift(home, "get", "papers/p.jsonl#p1", "--json"), "found")
    assert (record["title"], record["text"]) == ("Brush strokes", "# Brush strokes\n\nlearning to paint")
    embedded = sift(home, "embed", "papers", "--reset", "--json", model=model)
    refused = cast("Answer", answer(embedded)["error"])
    assert (embedded.returncode, refused["field"]) == (2, "reset")
    assert sift(home, "embed", "--reset", "--json", model=model).returncode == 0


def documents_run(home: Path, folder: Path) -> None:
    folder.mkdir()
    shutil.copy(sample(".pdf"), folder / "pigments.pdf")
    attached = [("contract.docx", sample(".docx").read_bytes(), WORD)]
    (folder / "offer.eml").write_bytes(mail_bytes(message(headers={"Subject": "Offer"}, attachments=attached)))
    (folder / "broken.pdf").write_bytes(b"%PDF-1.4 the rest is missing")
    assert sift(home, "add", str(folder), "--name", "docs").returncode == 0
    done = sift(home, "index")
    assert (done.returncode, "  done  +3  ~0  -0  " in done.stderr) == (0, True)
    found = [str(hit["id"]) for hit in hits_of(sift(home, "search", "Kubelka-Munk", "--json"))]
    assert sorted(found) == ["docs/offer.eml#contract.docx", "docs/pigments.pdf"]
    assert sift(home, "get", "docs/offer.eml#contract.docx").stdout.startswith("# Kubelka-Munk theory\n")
    listing = sift(home, "status", "docs", "--unreadable", "--json").stdout
    assert '"path": "broken.pdf"' in listing


def said(ran: subprocess.CompletedProcess[str]) -> tuple[int, str, str]:
    return ran.returncode, ran.stdout, ran.stderr


def served_run(home: Path, notes: Path) -> None:
    indexed(home, notes)
    asked = (("search", "brush", "--json"), ("status", "--json"), ("get", "notes/none.md"))
    alone = [said(process.sift(home, *argv)) for argv in asked]
    RAN.append(("serve", "--port", "0", "--json"))
    with process.served(home) as running:
        assert [said(process.sift(home, *argv)) for argv in asked] == alone
        again = process.sift(home, "index")
        assert (again.returncode, "  done  +0  ~0  -0  " in again.stderr) == (0, True)
        jobs = sift(home, "jobs", "--json")
        assert (jobs.returncode, json.loads(jobs.stdout)) == (0, {"jobs": [], "stopped": None})
        unknown = sift(home, "jobs", "--stop", "99")
        assert (unknown.returncode, "there is no job 99" in unknown.stderr) == (1, True)
    assert (running.poll() is not None, found(home)) == (True, None)
    (log,) = (home / "logs").glob("serve-*.log")
    logged = log.read_text(encoding="utf-8")
    answered = ("search answered", "status answered", "get answered", "index answered")
    assert all(f" INFO sift.server: {name} " in logged for name in answered)


def test_every_command_runs_as_a_process_from_a_folder_to_its_documents(tmp_path: Path) -> None:
    notes = notes_in(tmp_path / "Notizen Müller")
    model, reranker = str(tiny_model(tmp_path / "models")), str(tiny_reranker(tmp_path / "models"))
    runs = (
        partial(plain_run, tmp_path / "plain", notes),
        partial(models_run, tmp_path / "models-home", notes, model, reranker),
        partial(options_run, tmp_path / "options", notes, model, reranker),
        partial(records_run, tmp_path / "records", notes_in(tmp_path / "Papiere"), model),
        partial(documents_run, tmp_path / "documents", tmp_path / "Dokumente"),
        partial(served_run, tmp_path / "served", notes),
    )
    with ThreadPoolExecutor(len(runs)) as pool:
        for running in [pool.submit(run) for run in runs]:
            running.result()
    assert unexercised(RAN) == []
