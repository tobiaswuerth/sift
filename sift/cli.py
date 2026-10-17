"""Command-line interface."""

import argparse
import json
import sqlite3
import sys
import textwrap

from . import __version__, store


def _filters(pairs):
    out = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise SystemExit(f"filter must be KEY=VALUE, got {pair!r}")
        try:
            out[key] = json.loads(value)
        except ValueError:
            out[key] = value
    return out


def _print_hits(hits, as_json):
    if as_json:
        print(json.dumps(hits, ensure_ascii=False, indent=2))
        return
    for h in hits:
        print(f"{h['score']:>8.4f}  {h['id']}:{h['lines']}  {h['title']}")
        print(textwrap.indent(textwrap.shorten(h["text"], 300), " " * 10))


def main(argv=None):
    p = argparse.ArgumentParser(prog="sift", description="Fast local search over text and Markdown files.")
    p.add_argument("--version", action="version", version=f"sift {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("add", help="register a folder as a collection")
    s.add_argument("path")
    s.add_argument("--name", help="collection name (default: folder name)")
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
        s.add_argument("--json", action="store_true", help="print results as JSON")
    s = sub.add_parser("get", help="print a document by its id (collection/path)")
    s.add_argument("id")
    sub.add_parser("status", help="show collections and index statistics")
    s = sub.add_parser("serve", help="run the MCP server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--stdio", action="store_true", help="serve over stdio instead of HTTP")

    a = p.parse_args(argv)
    if (sys.stdout.encoding or "").lower() != "utf-8":  # e.g. a Windows pipe: titles may be any script
        sys.stdout.reconfigure(encoding="utf-8")
    db = store.connect()
    try:
        if a.cmd == "add":
            name = store.add_collection(db, a.path, a.name)
            print(f"added collection {name!r}; run `sift index` next", file=sys.stderr)
        elif a.cmd == "rm":
            n = store.remove_collection(db, a.name)
            documents = "document" if n == 1 else "documents"
            print(f"removed collection {a.name!r} and its {n} {documents} from the index;", end=" ", file=sys.stderr)
            print("the files on disk are untouched", file=sys.stderr)
        elif a.cmd == "index":
            store.index(db, a.names)
        elif a.cmd == "embed":
            from .embed import embed_pending

            embed_pending(db, a.reset)
        elif a.cmd in ("search", "vsearch", "query"):
            from .search import Index

            fn = getattr(Index(db), a.cmd)
            _print_hits(fn(a.query, a.k, a.collection, a.path, _filters(a.filter)), a.json)
        elif a.cmd == "get":
            from .search import Index

            sys.stdout.write(Index(db).get(a.id))
        elif a.cmd == "status":
            from .search import Index

            print(json.dumps(Index(db).status(), indent=2))
        else:  # serve
            from .server import serve

            serve(a.host, a.port, a.stdio)
    except (LookupError, ValueError, sqlite3.Error) as e:
        raise SystemExit(f"error: {e}") from None
    except KeyboardInterrupt:
        raise SystemExit(130) from None
