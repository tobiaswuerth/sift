"""Command-line interface."""

import argparse
import contextlib
import json
import os
import re
import sqlite3
import sys
import textwrap
from pathlib import Path

from . import __version__, console, readers, settings, store

# The commands that load PyTorch, unless a worker does it for them; with SIFT_EMBED_REMOTE, THROUGH_WORKER do.
SLOW = ("embed", "vsearch", "query", "similar", "bench", "serve", "worker")
THROUGH_WORKER = ("vsearch", "query", "similar", "bench")
SEARCHES = ("search", "vsearch", "query", "similar")


def _value(text):
    try:
        return json.loads(text)
    except ValueError:
        return text


def _filters(pairs):
    """-f KEY=VALUE, KEY!=VALUE and ranges such as -f created>=2024-06 -f created<2025, as search filters. Repeated,
    KEY=VALUE stands for any of the values, and KEY!=VALUE for none of them."""
    out = {}
    for pair in pairs:
        m = re.fullmatch(r"(\w[^=<>!]*)(>=|<=|!=|=|<|>)(.*)", pair)  # a field name: "journal-ref", "date created"
        if not m:
            raise SystemExit(
                f"filter must be KEY=VALUE, KEY!=VALUE, or KEY>=VALUE and the like for ranges, got {pair!r}"
            )
        key, op, value = m[1].rstrip(), "not" if m[2] == "!=" else m[2], _value(m[3])
        conditions = out.setdefault(key, {})
        if op in ("=", "not") and op in conditions:
            value = _listed(conditions[op]) + _listed(value)
        conditions[op] = value
    return {key: c["="] if list(c) == ["="] else c for key, c in out.items()}


def _fields(text):
    """--fields url,created as a list of field names."""
    return text and [field.strip() for field in text.split(",")]


def _listed(value):
    return value if isinstance(value, list) else [value]


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


def _print_links(found, as_json):
    """`sift links`: each link out with where it leads, and each document linking in with its lines."""
    if as_json:
        print(json.dumps(found, ensure_ascii=False, indent=2))
        return
    note = f"{found['title']} ({found['id']})"
    for way, heading in (("out", f"{note} links to"), ("in", f"Linking to {note}")):
        if way not in found:
            continue
        print(f"{heading}:" if found[way] else f"{heading}: nothing")
        for link in found[way]:
            through = f"  (through {link['through']})" if "through" in link else ""
            if way == "in":
                print(f"  {link['id']}  {link['title']}{through}")
                for line in link["links"]:
                    print(f"{line['line']:>6}  {line['text']}")
                continue
            written = (f"{link['field']}: " if "field" in link else "") + link["target"]
            written += f"#{link['fragment']}" if "fragment" in link else ""
            if "id" in link:
                to = link["id"] + (" (alias)" if link["via"] == "alias" else "")
            else:
                to = "ambiguous: " + ", ".join(link["ambiguous"]) if "ambiguous" in link else "unresolved"
            print(f"{link['line']:>6}  {written}  ->  {to}{through}")


def _print_unresolved(targets, as_json):
    """`sift unresolved`: each target with how many documents link it, and a few of them."""
    if as_json:
        print(json.dumps(targets, ensure_ascii=False, indent=2))
        return
    if not targets:
        console.say("no unresolved links")
    for target in targets:
        more = ", ..." if target["docs"] > len(target["ids"]) else ""
        print(f"{target['docs']:>6}  {target['target']}  {', '.join(target['ids'])}{more}")


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
    s.add_argument(
        "--title-from",
        choices=readers.TITLE_FROM,
        help="Markdown and text files: a file's title, unless its front matter sets one, is its first heading "
        "(default) or its file name, as in an Obsidian vault. Given for a folder that is a collection already, it "
        "changes that collection",
    )
    s.add_argument(
        "--nested",
        action="store_true",
        help="add a folder inside a collection, or one that holds a collection, though the files of both are then "
        "indexed twice",
    )
    s = sub.add_parser("rm", help="remove a collection from the index (its files stay on disk)")
    s.add_argument("name")
    s = sub.add_parser("index", help="scan collections and update the keyword index")
    s.add_argument("names", nargs="*", help="collections to scan (default: all)")
    s.add_argument(
        "--files",
        nargs="+",
        metavar="PATH",
        help="read only these files now, or drop them if gone: paths in a collection, or ids (collection/path); "
        "`sift embed` embeds them",
    )
    s = sub.add_parser("embed", help="embed all chunks that have no vector yet")
    s.add_argument("--reset", action="store_true", help="drop all vectors first (after changing the model)")
    for name, desc, subject in (
        ("search", "exact keyword search (BM25)", "query"),
        ("vsearch", "semantic search", "query"),
        ("query", "hybrid search with reranking", "query"),
        ("similar", "documents like a given one", "id"),
    ):
        s = sub.add_parser(name, help=desc)
        s.add_argument("subject", metavar=subject)
        s.add_argument("-k", type=int, default=10, help="number of results (default: 10)")
        s.add_argument("-c", "--collection")
        s.add_argument("--path", help='path glob within the collection, e.g. "2024/*"')
        s.add_argument(
            "-f",
            "--filter",
            action="append",
            default=[],
            metavar="KEY=VALUE",
            help="front-matter filter, repeatable: KEY=VALUE (repeated: any of them), KEY!=VALUE, KEY>=VALUE, ...",
        )
        s.add_argument("--fields", help='front-matter fields to show, comma-separated (e.g. "url,created"; "*": all)')
        s.add_argument("--compact", action="store_true", help="leave out the text")
        s.add_argument(
            "--all-passages", action="store_true", help="every matching passage, not just each document's best"
        )
        s.add_argument(
            "--sort",
            metavar="FIELD",
            help="order by a front-matter field instead of score; --sort=-created: newest first",
        )
        s.add_argument(
            "--links-to",
            action="append",
            metavar="NOTE",
            help="only documents linking to this note, by id or as links name it, e.g. 'Acme AG'; repeatable: any",
        )
        s.add_argument("--linked-from", action="append", metavar="NOTE", help="only documents this note links to")
        s.add_argument("--json", action="store_true", help="print results as JSON")
    s = sub.add_parser("get", help="print a document by its id (collection/path)")
    s.add_argument("id")
    s.add_argument("--lines", help='only these lines, e.g. "17-26"')
    s.add_argument("--fields", help='front-matter fields to show first, comma-separated (e.g. "url,created"; "*": all)')
    s = sub.add_parser("status", help="show collections, their folders, and index statistics, as JSON")
    s.add_argument("--json", action="store_true", help="as other commands take it: status prints JSON anyway")
    s = sub.add_parser("links", help="the links of a Markdown note: those it holds, and those into it")
    s.add_argument("id")
    s.add_argument("--direction", choices=["out", "in", "both"], default="out", help="out (default), in, or both")
    s.add_argument("--depth", type=int, choices=[1, 2], default=1, help="2: also the notes one link further")
    s.add_argument("--field", help="only links in this front-matter field")
    s.add_argument("-k", type=int, default=50, help="at most this many links each way (default: 50)")
    s.add_argument("--json", action="store_true", help="print the links as JSON")
    s = sub.add_parser("mentions", help="documents that name a note without linking to it")
    s.add_argument("id")
    s.add_argument("-k", type=int, default=10, help="number of results (default: 10)")
    s.add_argument("-c", "--collection")
    s.add_argument("--also", help='front-matter fields whose values are names of the note too, e.g. "emails"')
    s.add_argument("--fields", help='front-matter fields to show, comma-separated (e.g. "url,created"; "*": all)')
    s.add_argument("--compact", action="store_true", help="leave out the text")
    s.add_argument("--json", action="store_true", help="print results as JSON")
    s = sub.add_parser("unresolved", help="link targets no note answers to yet, most linked first")
    s.add_argument("-c", "--collection")
    s.add_argument("-k", type=int, default=50, help="number of targets (default: 50)")
    s.add_argument("--json", action="store_true", help="print the targets as JSON")
    s = sub.add_parser("bench", help="measure each search mode on questions whose answers you know")
    s.add_argument("file", help='JSON: [{"query": "...", "relevant": ["collection/path", ...]}, ...]')
    s.add_argument("-c", "--collection", help="search only this collection")
    s = sub.add_parser("serve", help="run the MCP server")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--stdio", action="store_true", help="serve over stdio instead of HTTP")
    s = sub.add_parser("ui", help="search, browse and read the index in a web browser (starts `sift serve` if needed)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
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


def _add(a, db):
    """`sift add`: register a folder; or, for a folder registered already, change where its titles come from."""
    chosen = {"id": a.id, "title": a.title, "text": a.text and [f.strip() for f in a.text.split(",")]}
    chosen = {k: v for k, v in chosen.items() if v} | ({"title_from": a.title_from} if a.title_from else {})
    existing = store.collection_at(db, a.path)
    if existing and list(chosen) == ["title_from"] and a.name in (None, existing):
        from .indexing import retitle

        changed, dropped = retitle(db, existing, a.title_from)
        source = "file names" if a.title_from == "name" else "headings"
        console.say(
            f"collection {existing!r} takes titles from {source} now: {console.count(changed, 'title')} changed"
            + (
                f"; run `sift embed` to embed {console.count(dropped, 'passage')} again, which carries its title"
                if dropped
                else ""
            )
        )
        return
    name = store.add_collection(db, a.path, a.name, chosen, nested=a.nested)
    console.say(f"added collection {name!r}; run `sift index` next")
    if not a.title_from and (Path(a.path).expanduser() / ".obsidian").is_dir():
        console.say(
            f"{a.path} is an Obsidian vault: to title its notes by their file names, as Obsidian does, run "
            f"`{console.command('sift', 'add', a.path, '--title-from', 'name')}`"
        )


def _run(a, db):
    if a.cmd == "add":
        _add(a, db)
    elif a.cmd == "rm":
        n = store.remove_collection(db, a.name)
        console.say(
            f"removed collection {a.name!r} and its {console.count(n, 'document')} from the index; "
            "the files on disk are untouched"
        )
    elif a.cmd == "index" and a.files:
        from .indexing import reindex

        if a.names:
            raise SystemExit("error: give collections or --files, not both")
        # a path from the current folder, as the shell completes it, or ./ or ../ before a file gone; else an id
        relative = tuple(dots + sep for dots in (".", "..") for sep in {"/", os.sep})
        paths = [os.path.abspath(p) if os.path.exists(p) or p.startswith(relative) else p for p in a.files]
        for entry, _ in reindex(db, paths).files:
            console.say(f"{entry['id']}: {entry['status']}" + (f" ({entry['note']})" if "note" in entry else ""))
    elif a.cmd == "index":
        from .indexing import index

        index(db, a.names)
    elif a.cmd == "embed":
        from .embed import embed_pending

        embed_pending(db, a.reset)
    elif a.cmd in SEARCHES:
        from .search import Index

        hits = getattr(Index(db), a.cmd)(
            a.subject,
            a.k,
            collection=a.collection,
            path=a.path,
            filters=_filters(a.filter),
            fields=_fields(a.fields),
            compact=a.compact,
            all_passages=a.all_passages,
            sort=a.sort,
            links_to=a.links_to,
            linked_from=a.linked_from,
        )
        _print_hits(hits, a.json)
    elif a.cmd == "get":
        from .search import Index

        document = Index(db).document(a.id, a.lines, _fields(a.fields))
        for key, value in document.get("fields", {}).items():
            print(f"{key}: {_shown(value)}")
        if "fields" in document:
            print()
        text = document["text"]
        sys.stdout.write(text if text.endswith("\n") or not text else text + "\n")  # JSON records end without one
    elif a.cmd == "bench":
        from . import bench
        from .search import Index

        index = Index(db)
        results = bench.run(index, bench.load(a.file, index), collection=a.collection)
        print(f"{'mode':10}{'nDCG@10':>10}{'MRR@10':>10}{'Recall@10':>11}{'ms/query':>10}")
        for mode, r in results.items():
            print(f"{mode:10}{r['nDCG@10']:>10.3f}{r['MRR@10']:>10.3f}{r['Recall@10']:>11.3f}{r['ms']:>10.0f}")
    elif a.cmd == "status":
        from .search import Index

        print(json.dumps(Index(db).status(), indent=2))
    elif a.cmd == "links":
        from .search import Index

        _print_links(Index(db).links(a.id, a.direction, a.depth, a.field, a.k), a.json)
    elif a.cmd == "mentions":
        from .search import Index

        also, fields = _fields(a.also), _fields(a.fields)
        hits = Index(db).mentions(a.id, a.k, collection=a.collection, also=also, fields=fields, compact=a.compact)
        _print_hits(hits, a.json)
    elif a.cmd == "unresolved":
        from .search import Index

        _print_unresolved(Index(db).unresolved(a.collection, a.k), a.json)
    elif a.cmd == "ui":
        from .ui import open_in_browser

        open_in_browser(a.host, a.port)
    else:  # serve
        from .server import serve

        serve(a.host, a.port, a.stdio)
