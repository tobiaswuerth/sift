from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tests.surface.smoke import Done, Smoke, answer, hits_of, ids

LONG = "\n\n".join(["Brush strokes again and again, " * 40] * 6)
OPTIONS = {
    "rl.md": (
        '---\nsee: "[[brush]]"\nrank: 1\n---\n# Reward shaping\n\nSpeeds up learning with a brush. See [[brush]].\n'
    ),
    "sub/brush.md": "---\ntool: Pinsel\nrank: 2\n---\n# Brush\n\nA brush stroke.\n",
    "painter.md": "---\nrank: 3\n---\n# Painter\n\nHer Pinsel is worn.\n",
    "long.md": f"# Long\n\n{LONG}\n",
    **{f"weak/w{n}.md": f"# Weak {n}\n\n{'Words without the tool. ' * 30}One brush.\n" for n in range(8)},
}
MENTIONED = "opts/sub/brush.md"
SEARCHED = {"search": "brush", "vsearch": "brush", "query": "brush", "similar": "opts/rl.md", "mentions": MENTIONED}
NARROWED = ("-k", "2", "-c", "opts", "--path", "*", "-f", "rank>=2", "--fields", "rank", "--sort=-rank")
BOTH_RANKED = [("opts/painter.md", {"rank": 3}), ("opts/sub/brush.md", {"rank": 2})]
RANKED = {
    "search": [("opts/sub/brush.md", {"rank": 2})],
    "vsearch": BOTH_RANKED,
    "query": BOTH_RANKED,
    "similar": BOTH_RANKED,
    "mentions": [("opts/painter.md", {"rank": 3})],
}
LINKING = {"search": ["opts/rl.md"], "vsearch": ["opts/rl.md"], "query": ["opts/rl.md"], "similar": [], "mentions": []}
LINKED = {
    "search": ["opts/sub/brush.md"],
    "vsearch": ["opts/sub/brush.md"],
    "query": ["opts/sub/brush.md"],
    "similar": ["opts/sub/brush.md"],
    "mentions": [],
}
ALSO = {"mentions": ("--also", "tool")}
NO_DROP = frozenset({"similar"})


def _texts(done: Done) -> list[object]:
    return [hit["text"] for hit in hits_of(done)]


def _cut_short(done: Done, found: list[str], *, may_stay: bool) -> bool:
    kept = ids(done)
    return kept == found[: len(kept)] and (len(kept) < len(found) or may_stay)


def searched(smoke: Smoke, command: str, target: str) -> None:
    asked = (command, target)
    plain = smoke.checked(
        *asked, "-k", "10", "--json", holds=lambda done: all(isinstance(text, str) for text in _texts(done))
    )
    found = ids(plain)
    assert (len(found) >= 2, len(set(found))) == (True, len(found)), found
    smoke.checked(*asked, "-k", "10", "--compact", "--json", holds=lambda done: _texts(done) == [None] * len(found))
    cut = command in NO_DROP
    smoke.checked(*asked, "-k", "10", "--adaptive", "--json", holds=lambda done: _cut_short(done, found, may_stay=cut))
    smoke.checked(
        *asked, "-k", "10", "--all-passages", "--json", holds=lambda done: len(set(ids(done))) < len(ids(done))
    )
    smoke.checked(
        *asked,
        *NARROWED,
        *ALSO.get(command, ()),
        "--json",
        holds=lambda done: [(hit["id"], hit["fields"]) for hit in hits_of(done)] == RANKED[command],
    )
    smoke.checked(*asked, "--links-to", "brush", "--json", holds=lambda done: ids(done) == LINKING[command])
    smoke.checked(*asked, "--linked-from", "opts/rl.md", "--json", holds=lambda done: ids(done) == LINKED[command])
    smoke.checked(
        *asked,
        "-k",
        "10",
        "--budget",
        "5",
        "--json",
        holds=lambda done: (answer(done)["partial"], ids(done)) == (False, found),
    )
    smoke.checked(*asked, "-k", "1", "--offset", "1", "--json", holds=lambda done: ids(done) == found[1:2])


def options_run(smoke: Smoke, folder: Path) -> None:
    for path, text in OPTIONS.items():
        (folder / path).parent.mkdir(parents=True, exist_ok=True)
        (folder / path).write_text(text, encoding="utf-8")
    smoke.checked("add", str(folder), "--name", "opts", holds=lambda done: done.returncode == 0)
    smoke.checked("index", holds=lambda done: "  done  +12  ~0  -0  " in done.stderr)
    smoke.checked("embed", holds=lambda done: done.returncode == 0)
    each = {command: Smoke(smoke.home, smoke.model, smoke.reranker) for command in SEARCHED}
    with ThreadPoolExecutor(len(SEARCHED)) as pool:
        for running in [pool.submit(searched, each[command], command, target) for command, target in SEARCHED.items()]:
            running.result()
    smoke.exercised.extend(argv for alone in each.values() for argv in alone.exercised)
    smoke.checked("query", "brush", "-k", "2", "--candidates", "1", "--json", holds=lambda done: len(ids(done)) == 2)
    without = ids(smoke.ran("mentions", MENTIONED, "--json"))
    assert "opts/painter.md" not in without, without
    smoke.checked("mentions", MENTIONED, "--also", "tool", "--json", holds=lambda done: "opts/painter.md" in ids(done))
