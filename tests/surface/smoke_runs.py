import shutil
from pathlib import Path
from typing import cast

from sift.interfaces.server.endpoint import found
from tests.mails import WORD, mail_bytes, message, sample
from tests.surface import process
from tests.surface.smoke import Answer, Done, Smoke, answer, finding, ids, indexed, listed, ok


def similar_run(smoke: Smoke) -> None:
    smoke.checked(
        "similar",
        "notes/rl.md",
        "--json",
        "--compact",
        holds=lambda done: ok(done) and sorted(ids(done)) == ["notes/huge.txt", "notes/sub/brush.md"],
    )


def models_run(smoke: Smoke, notes: Path) -> None:
    indexed(smoke, notes)
    smoke.checked(
        "embed",
        holds=lambda done: (
            (done.returncode, done.stdout) == (0, "")
            and "  embed          3 passages" in done.stderr
            and "  load           1 model " in done.stderr
            and ("%|" in done.stderr, "it/s]" in done.stderr) == (False, False)
        ),
    )
    smoke.checked(
        "vsearch",
        "learning",
        "-k",
        "1",
        "--json",
        holds=lambda done: len(ids(done)) == 1 and "  load           1 model " in done.stderr,
    )
    similar_run(smoke)
    smoke.checked("vsearch", "learning", "--path", "sub/*", "--json", holds=finding("notes/sub/brush.md"))
    smoke.checked(
        "query",
        "reward shaping",
        "-k",
        "1",
        holds=lambda done: (
            (done.returncode, done.stdout.split("\n")[1].strip().startswith("notes/")) == (0, True)
            and done.stderr.startswith("  load           2 models ")
            and done.stderr.splitlines()[-1].startswith(
                "1 result, more follow  \N{MIDDLE DOT}  hybrid  \N{MIDDLE DOT}  1 collection"
            )
        ),
    )


def records_run(smoke: Smoke, notes: Path) -> None:
    home = smoke.home
    indexed(smoke, notes)
    papers = notes / "papers"
    papers.mkdir()
    (papers / "p.jsonl").write_bytes(b'{"key": "p1", "headline": "Brush strokes", "body": "learning to paint"}\n')
    chosen = ("--id-field", "key", "--title-field", "headline", "--text-fields", "body", "--title-from", "heading")
    store = ("--store", str(home / "kept" / "papers.sqlite"), "--nested", "--follow-links")
    smoke.checked(
        "add",
        str(papers),
        "--name",
        "papers",
        *store,
        *chosen,
        "--json",
        holds=lambda done: (done.returncode, answer(done)["name"]) == (0, "papers"),
    )
    smoke.checked("index", "papers", "--json", holds=lambda done: done.returncode == 0)
    smoke.checked(
        "get",
        "papers/p.jsonl#p1",
        "--json",
        holds=lambda done: (
            [(record["title"], record["text"]) for record in listed(done, "found")]
            == [("Brush strokes", "# Brush strokes\n\nlearning to paint")]
        ),
    )
    smoke.checked(
        "embed",
        "papers",
        "--reset",
        "--json",
        holds=lambda done: (done.returncode, cast("Answer", answer(done)["error"])["field"]) == (2, "reset"),
    )
    smoke.checked("embed", "--reset", "--json", holds=lambda done: done.returncode == 0)


def documents_run(smoke: Smoke, folder: Path) -> None:
    folder.mkdir()
    shutil.copy(sample(".pdf"), folder / "pigments.pdf")
    attached = [("contract.docx", sample(".docx").read_bytes(), WORD)]
    (folder / "offer.eml").write_bytes(mail_bytes(message(headers={"Subject": "Offer"}, attachments=attached)))
    (folder / "broken.pdf").write_bytes(b"%PDF-1.4 the rest is missing")
    smoke.checked("add", str(folder), "--name", "docs", holds=lambda done: done.returncode == 0)
    smoke.checked("index", holds=lambda done: (done.returncode, "  done  +3  ~0  -0  " in done.stderr) == (0, True))
    smoke.checked(
        "search",
        "Kubelka-Munk",
        "--json",
        holds=lambda done: sorted(ids(done)) == ["docs/offer.eml#contract.docx", "docs/pigments.pdf"],
    )
    smoke.checked(
        "get", "docs/offer.eml#contract.docx", holds=lambda done: done.stdout.startswith("# Kubelka-Munk theory\n")
    )
    smoke.checked("status", "docs", "--unreadable", "--json", holds=lambda done: '"path": "broken.pdf"' in done.stdout)


def said(done: Done) -> tuple[int, str, str]:
    return done.returncode, done.stdout, done.stderr


def served_run(smoke: Smoke, notes: Path, more: Path) -> None:
    home = smoke.home
    indexed(smoke, notes)
    asked = (("search", "brush", "--json"), ("status", "--json"), ("get", "notes/none.md"))
    alone = [said(smoke.ran(*argv)) for argv in asked]
    with process.served(home) as running:
        assert [said(smoke.ran(*argv)) for argv in asked] == alone
        smoke.exercised.append(process.SERVE)
        again = smoke.ran("index")
        assert (again.returncode, "  done  +0  ~0  -0  " in again.stderr) == (0, True)
        smoke.checked(
            "jobs", "--json", holds=lambda done: (done.returncode, answer(done)) == (0, {"jobs": [], "stopped": None})
        )
        smoke.checked(
            "jobs",
            "--stop",
            "99",
            holds=lambda done: (done.returncode, "there is no job 99" in done.stderr) == (1, True),
        )
        smoke.checked(
            "add",
            str(more),
            "--name",
            "more",
            "--json",
            holds=lambda done: (done.returncode, answer(done)["name"]) == (0, "more"),
        )
        smoke.checked("rm", "more", "--json", holds=lambda done: (done.returncode, answer(done)["documents"]) == (0, 0))
    assert (running.poll() is not None, found(process.settings_of(home))) == (True, None)
    (log,) = (home / "logs").glob("serve-*.log")
    logged = log.read_text(encoding="utf-8")
    answered = ("search", "status", "get", "index", "add", "rm")
    assert [name for name in answered if f" INFO sift.server: {name} answered " not in logged] == []
