"""Command-line interface."""

import argparse
import contextlib
import json
import re
import sqlite3
import sys
import textwrap

from . import __version__, console, settings, store

SLOW = ("embed", "vsearch", "query", "serve", "worker")  # the commands that load PyTorch, unless a worker does it
THROUGH_WORKER = ("vsearch", "query")  # with SIFT_EMBED_REMOTE, these leave PyTorch to the worker


def _value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def _filters(pairs):
    """-f KEY=VALUE and ranges such as -f created>=2024-06 -f created<2025, as search filters."""
    out = {}
    for pair in pairs:
        m = re.fullmatch(r"([^=<>]+)(>=|<=|=|<|>)(.*)", pair)
        if not m:
            raise SystemExit(f"filter must be KEY=VALUE, or KEY>=VALUE and the like for ranges, got {pair!r}")
        key, op, value = m[1], m[2], _value(m[3])
        if op == "=" and key not in out:
            out[key] = value
        else:
            conditions = out.get(key, {})
            out[key] = (conditions if isinstance(conditions, dict) else {"=": conditions}) | {op: value}
    return out


def _shown(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def _print_hits(hits, as_json):
    if as_json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return
    if not hits:
        console.say("no matches")
    for h in hits:
        print(f"{h['score']:>8.4f}  {h['id']}:{h['lines']}  {h['title']}")
        if "fields" in h:
            print(" " * 10 + "  ".join(f"{k}: {_shown(v)}" for k, v in h["fields"].items()))
        if "text" in h:
            print(textwrap.indent(textwrap.shorten(h["text"], 300), " " * 10))


def main(argv=None):
    p = argparse.ArgumentParser(prog="sift", description="Fast local search over your documents and notes.")
    p.add_argument("--version", action="version", version=f"sift {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("add", help="register a folder as a collection")
    s.add_argument("path")
    s.add_argument("--name", help="collection name (default: folder name)")
    s.add_argument("--id", help="JSON Lines: the field with each record's id (default: id or _id, else the line)")
    s.add_argument("--title", help="JSON Lines: the title field (default: title, name, headline or subject)")
    s.add_argument(
        "--text",
        help="JSON Lines: the text field(s), comma-separated "
        "(default: text, content, body, abstract, description or summary, else the longest string)",
    )
    s = sub.add_parser("rm", help="remove a collection from the index (its files stay on disk)")
    s.add_argument("name")
    s = sub.add_parser("index", help="scan collections and update the keyword index")
    s.add_argument("names", nargs="*", help="collections to scan (default: all)")
    s = sub.add_parser("embed", help="embed all chunks that have no vector yet")
    s.add_argument("--reset", action="store_true", help="drop all vectors first (after changing the model)")
    for name, desc in (
        ("search", "exact keyword search (BM25)"),
        ("vsearch", "semantic search"),
        ("query", "hybrid search with reranking"),
    ):
        s = sub.add_parser(name, help=desc)
        s.add_argument("query")
        s.add_argument("-k", type=int, default=10, help="number of results (default: 10)")
        s.add_argument("-c", "--collection")
        s.add_argument("--path", help='path glob within the collection, e.g. "2024/*"')
        s.add_argument(
            "-f", "--filter", action="append", default=[], metavar="KEY=VALUE", help="front-matter filter, repeatable"
        )
        s.add_argument("--fields", help='front-matter fields to show, comma-separated (e.g. "url,created"; "*": all)')
        s.add_argument("--compact", action="store_true", help="leave out the text")
        s.add_argument("--json", action="store_true", help="print results as JSON")
    s = sub.add_parser("get", help="print a document by its id (collection/path)")
    s.add_argument("id")
    s.add_argument("--lines", help='only these lines, e.g. "17-26"')
    s.add_argument("--fields", help='front-matter fields to show first, comma-separated (e.g. "url,created"; "*": all)')
    sub.add_parser("status", help="show collections and index statistics")
    s = sub.add_parser("serve", help="run the MCP server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--stdio", action="store_true", help="serve over stdio instead of HTTP")
    s = sub.add_parser("worker", help="embed for sift on other machines, which set SIFT_EMBED_REMOTE to this one")
    s.add_argument("--host", default="0.0.0.0", help="address to listen on (default: all of this machine's)")
    s.add_argument("--port", type=int, default=8766)

    a = p.parse_args(argv)
    if (sys.stdout.encoding or "").lower() != "utf-8":  # e.g. a Windows pipe: titles may be any script
        sys.stdout.reconfigure(encoding="utf-8")
    if a.cmd in SLOW:  # loading PyTorch alone can take long: say first that sift runs
        loading = "" if settings.current().remote and a.cmd in THROUGH_WORKER else ", loading PyTorch"
        console.say(f"sift {a.cmd}: starting{loading} ...")
    try:
        if a.cmd == "worker":  # needs no index
            from .worker import serve

            serve(a.host, a.port)
            return
        with contextlib.closing(store.connect()) as db:
            _run(a, db)
    except (LookupError, ValueError, OSError, sqlite3.Error) as e:
        raise SystemExit(f"error: {e}") from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None


def _run(a, db):
    if a.cmd == "add":
        chosen = {"id": a.id, "title": a.title, "text": a.text and [f.strip() for f in a.text.split(",")]}
        name = store.add_collection(db, a.path, a.name, {k: v for k, v in chosen.items() if v})
        console.say(f"added collection {name!r}; run `sift index` next")
    elif a.cmd == "rm":
        n = store.remove_collection(db, a.name)
        console.say(
            f"removed collection {a.name!r} and its {console.count(n, 'document')} from the index; "
            "the files on disk are untouched"
        )
    elif a.cmd == "index":
        from .indexing import index

        index(db, a.names)
    elif a.cmd == "embed":
        from .embed import embed_pending

        embed_pending(db, a.reset)
    elif a.cmd in ("search", "vsearch", "query"):
        from .search import Index

        fields = a.fields and [f.strip() for f in a.fields.split(",")]
        options = {"collection": a.collection, "path": a.path, "filters": _filters(a.filter), "fields": fields}
        _print_hits(getattr(Index(db), a.cmd)(a.query, a.k, compact=a.compact, **options), a.json)
    elif a.cmd == "get":
        from .search import Index

        document = Index(db).document(a.id, a.lines, a.fields and [f.strip() for f in a.fields.split(",")])
        for key, value in document.get("fields", {}).items():
            print(f"{key}: {_shown(value)}")
        if "fields" in document:
            print()
        text = document["text"]
        sys.stdout.write(text if text.endswith("\n") or not text else text + "\n")  # JSON records end without one
    elif a.cmd == "status":
        from .search import Index

        print(json.dumps(Index(db).status(), indent=2))
    else:  # serve
        from .server import serve

        serve(a.host, a.port, a.stdio)
