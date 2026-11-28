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
RUSTUP = re.compile(r"rustup/archive/(?P<version>[0-9]+(?:\.[0-9]+)+)/")
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


def _written(url: str, header: str | None) -> float | None:
    if header is None:
        return None
    try:
        return email.utils.parsedate_to_datetime(header).timestamp()
    except ValueError:
        raise PinError(f"{url} answered a Last-Modified that is no date: {header!r}") from None


def fetched(url: str) -> Answer:
    parts = urllib.parse.urlsplit(url)
    connection = CONNECTIONS[parts.scheme](parts.netloc, timeout=TIMEOUT)
    try:
        connection.request("GET", urllib.parse.urlunsplit(("", "", parts.path, parts.query, "")))
        response = connection.getresponse()
        body = response.read(MOST_BYTES + 1)
        written = response.getheader("Last-Modified")
    except (OSError, http.client.HTTPException) as error:
        raise PinError(f"{url} could not be fetched: {error}") from None
    finally:
        connection.close()
    if response.status != http.client.OK or len(body) > MOST_BYTES:
        raise PinError(f"{url} answered {response.status} {response.reason}, {len(body):,} bytes")
    return Answer(body, _written(url, written))


def _line(text: str, at: int) -> int:
    return text.count("\n", 0, at) + 1


def _releases(answer: Answer, url: str) -> Iterator[tuple[int, str, str]]:
    try:
        read = cast("object", json.loads(answer.body))
    except ValueError:
        read = None
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
        releases = list(_releases(fetch(url), url))
        settled = [each for each in releases if each[0] <= now - SETTLED]
        if not settled:
            raise PinError(f"{url} names no release a week old")
        since, name, newest = max(settled)
        if not any(held == digest and released >= since for released, _, held in releases):
            said = f"{repository} is pinned at {digest}; its newest release a week old, {name}, is {newest}"
            yield Finding(WORKFLOW, _line(text, pinned.start()), said)


def _checksum(fetch: Fetch, version: str, target: str) -> tuple[str, float | None]:
    url = CHECKSUM.format(version=version, target=target)
    answer = fetch(url)
    found = SUM.match(answer.body)
    if found is None:
        raise PinError(f"what {url} answered is no checksum")
    return found[0].decode(), answer.written


def _numbered(version: str) -> tuple[int, ...]:
    return tuple(int(part) for part in version.split("."))


def _stable(fetch: Fetch) -> str:
    body = fetch(STABLE).body
    try:
        version = tomllib.loads(body.decode()).get("version")
    except ValueError:
        version = None
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
        if _numbered(stable) > _numbered(version):
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
    try:
        return report(behind(asking.pinned(), asking.fetch, asking.now()), REMEDY)
    except PinError as error:
        sys.stderr.write(f"error  {error}\n")
        return 1
