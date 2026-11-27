import email.utils
import http.client
import json
import re
import sys
import time
import tomllib
import urllib.parse
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from types import MappingProxyType
from typing import NamedTuple, cast

from tools.findings import Finding, read, report

WORKFLOW = ".github/workflows/wheels.yml"
TAGS = "https://quay.io/api/v1/repository/{repository}/tag/?limit=100&onlyActiveTags=true"
STABLE = "https://static.rust-lang.org/rustup/release-stable.toml"
CHECKSUM = "https://static.rust-lang.org/rustup/archive/{version}/{target}/rustup-init.sha256"
IMAGE = re.compile(r"image: quay\.io/(?P<repository>[\w./-]+)@(?P<digest>sha256:[0-9a-f]{64})")
RUSTUP = re.compile(r"rustup/archive/(?P<version>[0-9.]+)/")
TARGET = re.compile(r"rustup: (?P<target>[\w-]+)\s+rustup-sha256: (?P<checksum>[0-9a-f]{64})")
RELEASE = re.compile(r"[0-9]{4}\.[0-9]{2}\.[0-9]{2}-[0-9]+")
DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
VERSION = re.compile(r"[0-9]+(?:\.[0-9]+)+")
SUM = re.compile(rb"[0-9a-f]{64}")
SETTLED = 7 * 24 * 3600
TIMEOUT = 30.0
MOST_BYTES = 1 << 20
CONNECTIONS = MappingProxyType({"https": http.client.HTTPSConnection, "http": http.client.HTTPConnection})
USAGE = "usage: python -m tools pins"
REMEDY = (
    "Set what is named in .github/workflows/wheels.yml: nothing else moves these pins, and a release a week old is "
    "what Dependabot waits for with the others."
)


class PinError(Exception):
    pass


class Answer(NamedTuple):
    body: bytes
    written: float | None


type Fetch = Callable[[str], Answer]


def fetched(url: str) -> Answer:
    parts = urllib.parse.urlsplit(url)
    connection = CONNECTIONS[parts.scheme](parts.netloc, timeout=TIMEOUT)
    try:
        connection.request("GET", urllib.parse.urlunsplit(("", "", parts.path, parts.query, "")))
        response = connection.getresponse()
        body = response.read(MOST_BYTES + 1)
        written = response.getheader("Last-Modified")
    finally:
        connection.close()
    if response.status != http.client.OK or len(body) > MOST_BYTES:
        raise PinError(f"{url} answered {response.status} {response.reason}, {len(body):,} bytes")
    return Answer(body, None if written is None else email.utils.parsedate_to_datetime(written).timestamp())


def _line(text: str, at: int) -> int:
    return text.count("\n", 0, at) + 1


def _releases(answer: Answer, url: str) -> Iterator[tuple[int, str, str]]:
    read = cast("object", json.loads(answer.body))
    tags = cast("dict[str, object]", read).get("tags") if isinstance(read, dict) else None
    if not isinstance(tags, list):
        raise PinError(f"{url} answered no list of tags")
    for tag in cast("list[object]", tags):
        held = cast("dict[str, object]", tag) if isinstance(tag, dict) else {}
        name, digest, since = held.get("name"), held.get("manifest_digest"), held.get("start_ts")
        named = isinstance(name, str) and RELEASE.fullmatch(name) is not None
        if named and isinstance(digest, str) and DIGEST.fullmatch(digest) and isinstance(since, int):
            yield since, cast("str", name), digest


def _images(text: str, fetch: Fetch, now: float) -> Iterator[Finding]:
    for pinned in IMAGE.finditer(text):
        repository, digest = pinned["repository"], pinned["digest"]
        url = TAGS.format(repository=repository)
        settled = [each for each in _releases(fetch(url), url) if each[0] <= now - SETTLED]
        if not settled:
            raise PinError(f"{url} names no release a week old")
        _, name, newest = max(settled)
        if newest != digest:
            said = f"{repository} is pinned at {digest}; its newest release a week old, {name}, is {newest}"
            yield Finding(WORKFLOW, _line(text, pinned.start()), said)


def _checksum(fetch: Fetch, version: str, target: str) -> tuple[str, float | None]:
    url = CHECKSUM.format(version=version, target=target)
    answer = fetch(url)
    found = SUM.match(answer.body)
    if found is None:
        raise PinError(f"what {url} answered is no checksum")
    return found[0].decode(), answer.written


def _stable(fetch: Fetch) -> str:
    version = tomllib.loads(fetch(STABLE).body.decode()).get("version")
    if not isinstance(version, str) or VERSION.fullmatch(version) is None:
        raise PinError(f"{STABLE} names no version")
    return version


def _rustup(text: str, fetch: Fetch, now: float) -> Iterator[Finding]:
    pinned = RUSTUP.search(text)
    if pinned is None:
        raise PinError(f"{WORKFLOW} pins no rustup: this check reads what it no longer writes")
    version, stable = pinned["version"], _stable(fetch)
    newer: list[tuple[str, float | None]] = []
    for target in TARGET.finditer(text):
        built = target["target"]
        published, _ = _checksum(fetch, version, built)
        if published != target["checksum"]:
            said = f"rustup {version} for {built} is published with the checksum {published}, not the one pinned"
            yield Finding(WORKFLOW, _line(text, target.start("checksum")), said)
        if stable != version:
            checksum, written = _checksum(fetch, stable, built)
            newer.append((f"{checksum} for {built}", written))
    if newer and all(written is not None and written <= now - SETTLED for _, written in newer):
        sums = " and ".join(named for named, _ in newer)
        said = f"rustup {version} is pinned; {stable}, a week old, has the checksum {sums}"
        yield Finding(WORKFLOW, _line(text, pinned.start()), said)


def behind(text: str, fetch: Fetch, now: float) -> Iterator[Finding]:
    yield from _images(text, fetch, now)
    yield from _rustup(text, fetch, now)


def pinned() -> str:
    return read(WORKFLOW)


@dataclass(frozen=True, slots=True)
class Asking:
    fetch: Fetch = fetched
    now: Callable[[], float] = time.time
    pinned: Callable[[], str] = pinned


ASKING = Asking()


def main(argv: list[str], asking: Asking = ASKING) -> int:
    if argv:
        sys.stderr.write(f"{USAGE}\n")
        return 2
    return report(behind(asking.pinned(), asking.fetch, asking.now()), REMEDY)
