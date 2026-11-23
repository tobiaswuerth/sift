import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

from tests.tiny_model import tiny_model, tiny_reranker


def sift(home: Path, *args: str, model: str = "", reranker: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "sift", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=180,
        env={
            **os.environ,
            "SIFT_HOME": str(home),
            "PYTHONIOENCODING": "utf-8",
            "NO_COLOR": "1",
            "SIFT_EMBED_MODEL": model,
            "SIFT_RERANK_MODEL": reranker,
        },
    )


def hits_of(ran: subprocess.CompletedProcess[str]) -> list[dict[str, object]]:
    return cast("dict[str, list[dict[str, object]]]", json.loads(ran.stdout))["hits"]


def narrowed_searches_run(home: Path) -> None:
    options = ("--path", "sub/*", "-f", "year!=1999", "--fields", "*", "--sort=-year", "--json")
    narrowed = hits_of(sift(home, "search", "brush OR learning", *options))
    assert [(hit["id"], hit["fields"]) for hit in narrowed] == [("notes/sub/brush.md", {})]
    assert sift(home, "get", "notes/brush.md", "--fields", "*").stdout == "# Brush\n"


def links_run(home: Path) -> None:
    linked = sift(home, "links", "notes/rl.md", "--direction", "both")
    assert linked.stdout.split("\n")[1:4] == [
        "     2  see: brush  ->  notes/sub/brush.md",
        "     6  brush  ->  notes/sub/brush.md",
        "     6  Gone  ->  unresolved",
    ]
    assert sift(home, "unresolved").stdout == "1  Gone  notes/rl.md\n"
    linking = ("search", "learning OR brush", "--json", "--compact")
    assert [hit["id"] for hit in hits_of(sift(home, *linking, "--links-to", "brush"))] == ["notes/rl.md"]
    assert [hit["id"] for hit in hits_of(sift(home, *linking, "--linked-from", "notes/rl.md"))] == [
        "notes/sub/brush.md"
    ]
    assert [hit["id"] for hit in hits_of(sift(home, *linking, "-f", "see=[[sub/brush]]"))] == ["notes/rl.md"]


def test_every_command_runs_as_a_process_from_a_folder_to_its_documents(tmp_path: Path) -> None:
    home, notes = tmp_path / "home", tmp_path / "Notizen Müller"
    (notes / "sub").mkdir(parents=True)
    (notes / "rl.md").write_bytes(
        b'---\nsee: "[[brush]]"\n---\n# Reward shaping\n\nSpeeds up learning. See [[brush]], [[Gone]].\n'
    )
    (notes / "sub" / "brush.md").write_bytes(b"# Brush\n")
    (notes / "huge.txt").write_bytes(b"x")

    added = sift(home, "add", str(notes), "--name", "notes")
    assert (added.returncode, added.stdout) == (0, "")
    assert added.stderr.split("\n")[0] == f"added  notes  {notes.resolve()}"

    indexed = sift(home, "index")
    assert (indexed.returncode, indexed.stdout) == (0, "")
    assert "  done  +3  ~0  -0  " in indexed.stderr

    assert sift(home, "index", "notes", "--json").stdout.count('"added": 0') == 1

    status = sift(home, "status")
    assert (status.returncode, status.stderr) == (0, "")
    assert status.stdout.split("\n")[1].split()[-4:-2] == ["3", "3"]
    listed = sift(home, "status", "notes", "--unreadable", "--json")
    assert '"unreadable_files": []' in listed.stdout
    assert json.loads(listed.stdout)

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

    reranker = str(tiny_reranker(tmp_path / "models"))
    model = str(tiny_model(tmp_path / "models"))
    embedded = sift(home, "embed", model=model)
    assert (embedded.returncode, embedded.stdout) == (0, "")
    assert "  embed          3 chunks" in embedded.stderr
    assert "Loading" not in embedded.stderr
    near = sift(home, "vsearch", "learning", "-k", "1", "--json", model=model)
    assert len(cast("dict[str, list[object]]", json.loads(near.stdout))["hits"]) == 1
    kept = sift(home, "vsearch", "learning", "--path", "sub/*", "--json", model=model)
    assert [hit["id"] for hit in hits_of(kept)] == ["notes/sub/brush.md"]
    asked = sift(home, "query", "reward shaping", "-k", "1", model=model, reranker=reranker)
    assert (asked.returncode, asked.stdout.split("\n")[1].strip().startswith("notes/")) == (0, True)
    assert asked.stderr.startswith("1 result  \N{MIDDLE DOT}  hybrid  \N{MIDDLE DOT}  1 collection")

    missing = sift(home, "get", "notes/gone.md")
    assert (missing.returncode, missing.stdout) == (1, "")
    assert missing.stderr.startswith("error  notes/gone.md: unknown document")
    assert sift(home, "frobnicate").returncode == 2
    assert sift(home, "--help").stdout.startswith("usage: sift")
