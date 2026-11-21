"""The smoke run: every command of sift as a user runs it, each in a process of its own, on a small folder of notes
with a SIFT_HOME of its own, both temporary: `python scripts/smoke.py`.

Without options it runs what needs no model: adding, indexing, keyword search with its options, `get`, links, and
moving, deactivating and removing a collection. With --models also what loads PyTorch and the models (on the GPU, if
there is one; they are downloaded the first time): `embed`, `vsearch`, `query`, `similar`, `bench`, and `sift serve`
with its page and its MCP endpoint. `sift ui` opens a browser and `sift worker` serves other machines: neither is run.

Each step says what it ran; the first that fails ends the run, with what the command said."""

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

NOTES = {
    "Acme AG.md": "---\naliases: [Acme]\nyear: 2024\ntags: [org, client]\n---\n# Acme AG\n\n"
    "Acme builds rockets with [[Hans Example]] and [[A note not written]].\n",
    "People/Hans Example.md": '---\norg: "[[Acme AG]]"\nyear: "2023"\n---\n'
    "Hans Example plans first-stage engines for the rockets.\n",
    "papers.jsonl": '{"id": "p1", "title": "Brush strokes", "year": 2024, '
    '"text": "A painting robot plans brush strokes."}\n',
    "notes.txt": "Plain text about engines, written without a heading.\n",
}


class Smoke:
    """The run: its folders, and the commands so far."""

    def __init__(self, folder):
        self.vault, self.home, self.steps = folder / "vault", folder / "home", 0
        self.environment = os.environ | {"SIFT_HOME": str(self.home), "SIFT_UPDATE_CHECK": "off"}
        for name, text in NOTES.items():
            (self.vault / name).parent.mkdir(parents=True, exist_ok=True)
            (self.vault / name).write_text(text, encoding="utf-8")

    def sift(self, *args, fails=False, says=None):
        """Run `sift ARGS`; it must succeed (or, with `fails`, fail), and say `says`, if given. Returns its results."""
        self.steps += 1
        command = [sys.executable, "-m", "sift", *map(str, args)]
        ran = subprocess.run(command, capture_output=True, encoding="utf-8", env=self.environment, timeout=600)
        said = ran.stdout + ran.stderr
        if bool(ran.returncode) != fails or (says is not None and says not in said):
            wanted = f"to {'fail' if fails else 'succeed'}" + (f" and say {says!r}" if says else "")
            raise SystemExit(
                f"smoke: `sift {' '.join(map(str, args))}` was {wanted}; it ended with {ran.returncode}:\n{said}"
            )
        print(f"  sift {' '.join(map(str, args))}", flush=True)
        return ran.stdout

    def found(self, *args):
        """The ids a search finds, in order."""
        return [hit["id"] for hit in json.loads(self.sift(*args, "--json"))]


def without_models(run):
    sift, found = run.sift, run.found
    sift("add", run.vault, "--name", "notes", says="added collection 'notes'")
    sift("index", says="4 files, 4 added")
    status = json.loads(sift("status"))["collections"][0]
    assert (status["docs"], status["keywords"], status["unresolved"]) == (4, "ready", 1), status
    assert found("search", "rockets") == ["notes/People/Hans Example.md", "notes/Acme AG.md"]
    assert found("search", '"first-stage engines"', "-c", "notes", "--compact") == ["notes/People/Hans Example.md"]
    assert found("search", "rock*", "--path", "People/*", "--all-passages") == ["notes/People/Hans Example.md"]
    assert set(found("search", "rockets OR brush", "-f", "year=2024", "--sort=-year")) == {
        "notes/Acme AG.md",
        "notes/papers/p1",
    }
    assert found("search", "rockets", "-f", "year!=2024", "-k", "5", "--adaptive") == ["notes/People/Hans Example.md"]
    assert found("search", "rockets", "-f", "year>=2024", "--fields", "year,tags", "--budget", "5") == [
        "notes/Acme AG.md"
    ]
    assert found("search", "engines", "--links-to", "Acme AG") == ["notes/People/Hans Example.md"]
    assert found("search", "engines", "--linked-from", "notes/Acme AG.md") == ["notes/People/Hans Example.md"]
    sift("search", "rockets", says="notes/Acme AG.md:6-8")
    sift("get", "notes/Acme AG.md", "--lines", "6", "--fields", "*", says="# Acme AG")
    sift("get", "notes/Hans Example.md", "notes/papers/p1", says="==> notes/papers/p1 <==")
    sift("links", "Acme AG", "--direction", "both", "--depth", "2", says="->  notes/People/Hans Example.md")
    sift("links", "Acme", "--direction", "in", "--field", "org", "--json", says='"field": "org"')
    sift("mentions", "Acme AG", "--also", "aliases", says="no matches")  # the one note naming it links it
    sift("unresolved", says="A note not written")
    sift("add", run.vault, "--follow-links", says="reads files linked from outside its folder now")
    sift("add", run.vault, "--no-follow-links", says="leaves out files linked from outside its folder now")
    (run.vault / "notes.txt").write_text("Plain text, changed: now about turbines.\n", encoding="utf-8")
    sift("index", "--files", run.vault / "notes.txt", "notes/gone.md", says="notes/notes.txt: changed")
    assert found("search", "turbines") == ["notes/notes.txt"]
    sift("deactivate", "notes", says="inactive")
    sift("search", "rockets", fails=True, says="no collection is active")
    sift("activate", "notes", says="active")
    sift("move", "notes", run.home / "elsewhere", says="moved the index of notes")
    assert found("search", "rockets"), "nothing found after the move"
    sift("search", "rockets", "-c", "nope", fails=True, says="no collection named 'nope'")


def with_models(run):
    sift, found = run.sift, run.found
    sift("embed", says="embedded 4 chunks")
    assert found("vsearch", "spacecraft propulsion", "-k", "2")[0] in (
        "notes/Acme AG.md",
        "notes/People/Hans Example.md",
    )
    assert found("query", "who plans the engines?", "-k", "1") == ["notes/People/Hans Example.md"]
    assert found("similar", "notes/Acme AG.md", "-k", "1") == ["notes/People/Hans Example.md"]
    bench = run.home / "bench.json"
    bench.write_text(json.dumps([{"query": "painting robot", "relevant": ["notes/papers/p1"]}]), encoding="utf-8")
    sift("bench", bench, says="nDCG@10")
    serve(run)


def serve(run):
    """`sift serve` on a free port: its page, a search on it, and its MCP endpoint; then stopped."""
    with socket.socket() as free:
        free.bind(("127.0.0.1", 0))
        port = free.getsockname()[1]
    command = [sys.executable, "-m", "sift", "serve", "--port", str(port), "--allow-host", "localhost"]
    with subprocess.Popen(command, env=run.environment, stderr=subprocess.DEVNULL) as server:
        try:
            page = _answer(f"http://127.0.0.1:{port}/ui/?q=rockets", server)
            assert "notes/Acme AG.md" in page, "the page lists no result"
            hello = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}}
            hello["params"] |= {"capabilities": {}, "clientInfo": {"name": "smoke", "version": "0"}}
            headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
            asked = urllib.request.Request(f"http://127.0.0.1:{port}/mcp", json.dumps(hello).encode(), headers)
            assert '"name":"sift"' in _answer(asked, server).replace(" ", ""), "the MCP endpoint didn't name itself"
        finally:
            server.terminate()
    run.steps += 1
    print(f"  sift serve --port {port}: its page, a search on it, its MCP endpoint", flush=True)


def _answer(request, server, wait=600):
    """What the server answers, once it listens: its models load first."""
    directly = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + wait
    while True:
        try:
            with directly.open(request, timeout=60) as answer:
                return answer.read().decode()
        except ConnectionError:
            if server.poll() is not None or time.monotonic() > deadline:
                raise SystemExit("smoke: `sift serve` didn't come up") from None
            time.sleep(0.5)
        except urllib.error.URLError as refused:
            if (
                not isinstance(refused.reason, ConnectionError)
                or server.poll() is not None
                or time.monotonic() > deadline
            ):
                raise SystemExit(f"smoke: `sift serve` didn't answer: {refused}") from None
            time.sleep(0.5)


def main(argv=None):
    options = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    options.add_argument("--models", action="store_true", help="also run what loads PyTorch and the models")
    chosen = options.parse_args(argv)
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="sift-smoke-", ignore_cleanup_errors=True) as folder:
        run = Smoke(Path(folder))
        print("smoke: the commands that need no model", flush=True)
        without_models(run)
        if chosen.models:
            print("smoke: the commands that load the models", flush=True)
            with_models(run)
        run.sift("rm", "notes", says="removed collection 'notes'")
        assert json.loads(run.sift("status"))["collections"] == []
    print(f"smoke: {run.steps} commands as a user runs them, all as expected ({time.perf_counter() - start:.0f}s)")


if __name__ == "__main__":
    main()
