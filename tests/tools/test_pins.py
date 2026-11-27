import http.server
import json
import threading
from collections.abc import Callable, Iterator, Mapping
from typing import override

import pytest

from tests.waits import joined
from tools.findings import read
from tools.pins import CHECKSUM, STABLE, TAGS, WORKFLOW, Answer, Asking, PinError, behind, fetched, main

NOW = 1_800_000_000.0
DAY = 24 * 3600
OLD, NEW, OTHER = "sha256:" + "a" * 64, "sha256:" + "b" * 64, "sha256:" + "c" * 64
PINNED, PUBLISHED = "1" * 64, "2" * 64
PINS = f"""
          - platform: linux-x86_64
            image: quay.io/pypa/manylinux_2_28_x86_64@{OLD}
            rustup: x86_64-unknown-linux-gnu
            rustup-sha256: {PINNED}
          RUSTUP: https://static.rust-lang.org/rustup/archive/1.29.3/${{{{ matrix.rustup }}}}/rustup-init
"""
REPOSITORY, TARGET = "pypa/manylinux_2_28_x86_64", "x86_64-unknown-linux-gnu"


def tags(*released: tuple[str, str, float]) -> Answer:
    held = [{"name": name, "manifest_digest": digest, "start_ts": int(NOW - age)} for name, digest, age in released]
    return Answer(json.dumps({"tags": held}).encode(), None)


def answering(answers: Mapping[str, Answer]) -> Callable[[str], Answer]:
    return lambda url: answers[url]


def current(**changed: Answer) -> dict[str, Answer]:
    return {
        TAGS.format(repository=REPOSITORY): tags(("2026.11.14-1", OLD, 13 * DAY)),
        STABLE: Answer(b"schema-version = '1'\nversion = '1.29.3'\n", None),
        CHECKSUM.format(version="1.29.3", target=TARGET): Answer(f"{PINNED} *./rustup-init\n".encode(), NOW - 30 * DAY),
    } | {
        {"tags": TAGS.format(repository=REPOSITORY), "stable": STABLE}.get(name, name): answer
        for name, answer in changed.items()
    }


def told(answers: Mapping[str, Answer]) -> list[tuple[int, str]]:
    return [(finding.line, finding.problem) for finding in behind(PINS, answering(answers), NOW)]


def test_pins_that_are_the_newest_releases_a_week_old_are_behind_nothing() -> None:
    assert told(current()) == []
    younger = tags(("latest", NEW, 6 * DAY), ("2026.11.21-1", NEW, 6 * DAY), ("2026.11.14-1", OLD, 13 * DAY))
    assert told(current(tags=younger)) == []


def test_an_image_whose_newest_release_a_week_old_is_another_is_told_with_the_digest_to_set() -> None:
    released = tags(
        ("latest", OTHER, 1 * DAY),
        ("2026.11.26-1", OTHER, 1 * DAY),
        ("2026.11.14-1", OLD, 13 * DAY),
        ("2026.11.20-1", NEW, 7 * DAY),
        ("nightly", OTHER, 9 * DAY),
    )
    assert told(current(tags=released)) == [
        (3, f"{REPOSITORY} is pinned at {OLD}; its newest release a week old, 2026.11.20-1, is {NEW}")
    ]


def test_a_newer_rustup_is_told_with_its_checksums_once_it_is_a_week_old() -> None:
    newer = Answer(b"version = '1.30.0'\n", None)
    published = CHECKSUM.format(version="1.30.0", target=TARGET)
    fresh = current(stable=newer) | {published: Answer(f"{PUBLISHED} *./rustup-init\n".encode(), NOW - 2 * DAY)}
    assert told(fresh) == []
    undated = fresh | {published: Answer(f"{PUBLISHED} *./rustup-init\n".encode(), None)}
    assert told(undated) == []
    settled = fresh | {published: Answer(f"{PUBLISHED} *./rustup-init\n".encode(), NOW - 8 * DAY)}
    assert told(settled) == [
        (6, f"rustup 1.29.3 is pinned; 1.30.0, a week old, has the checksum {PUBLISHED} for {TARGET}")
    ]


def test_a_checksum_other_than_the_one_published_for_the_pinned_rustup_is_told() -> None:
    other = {CHECKSUM.format(version="1.29.3", target=TARGET): Answer(f"{PUBLISHED}  rustup-init\n".encode(), NOW)}
    assert told(current() | other) == [
        (5, f"rustup 1.29.3 for {TARGET} is published with the checksum {PUBLISHED}, not the one pinned")
    ]


@pytest.mark.parametrize(
    "changed, refusal",
    [
        ({"tags": Answer(b"[]", None)}, "no list of tags"),
        ({"tags": Answer(b'{"tags": 7}', None)}, "no list of tags"),
        ({"tags": Answer(b'{"tags": [7]}', None)}, "no release a week old"),
        (
            {
                "tags": Answer(
                    b'{"tags": [{"name": "2026.11.14-1", "manifest_digest": "sha256:zz", "start_ts": 5}]}', None
                )
            },
            "no release a week old",
        ),
        (
            {"tags": Answer(b'{"tags": [{"name": "2026.11.14-1", "manifest_digest": 5, "start_ts": 5}]}', None)},
            "no release a week old",
        ),
        (
            {
                "tags": Answer(
                    b'{"tags": [{"name": "2026.11.14-1", "manifest_digest": "sha256:'
                    + b"a" * 64
                    + b'", "start_ts": "5"}]}',
                    None,
                )
            },
            "no release a week old",
        ),
        ({"stable": Answer(b"version = 7\n", None)}, "names no version"),
        ({"stable": Answer(b"version = '1.30; rm'\n", None)}, "names no version"),
        ({CHECKSUM.format(version="1.29.3", target=TARGET): Answer(b"not a checksum", None)}, "is no checksum"),
    ],
)
def test_an_answer_that_is_not_what_was_asked_for_is_refused(changed: dict[str, Answer], refusal: str) -> None:
    with pytest.raises(PinError, match=refusal):
        told(current(**changed))


def test_a_workflow_that_pins_no_rustup_is_refused_and_one_without_images_has_none_behind() -> None:
    with pytest.raises(PinError, match="pins no rustup"):
        list(behind("jobs: {}", answering({}), NOW))
    only = "RUSTUP: https://static.rust-lang.org/rustup/archive/1.29.3/x/rustup-init\n"
    assert list(behind(only, answering({STABLE: Answer(b"version = '1.29.3'", None)}), NOW)) == []


def test_every_pin_of_the_wheels_workflow_is_asked_about() -> None:
    asked: list[str] = []

    def answered(url: str) -> Answer:
        asked.append(url)
        if url == STABLE:
            return Answer(b"version = '0.0.0'\n", None)
        if "rustup" in url:
            return Answer(b"0" * 64 + b" *./rustup-init\n", None)
        return tags(("2000.01.01-1", OLD, 30 * DAY))

    found = list(behind(read(WORKFLOW), answered, NOW))
    assert sum("quay.io" in url for url in asked) == 2
    assert sum("rustup-init.sha256" in url for url in asked) == 4
    assert [finding.path for finding in found] == [WORKFLOW] * 4
    assert all(finding.line > 1 for finding in found)


class Answering(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        found = self.path == "/held?as=asked"
        body = b"held" * (1 if found else 1 << 19)
        self.send_response(200 if found or self.path == "/long" else 404)
        self.send_header("Content-Length", str(len(body)))
        if found:
            self.send_header("Last-Modified", "Tue, 03 Nov 2026 14:16:33 GMT")
        self.end_headers()
        self.wfile.write(body)

    @override
    def log_message(self, format: str, *args: object) -> None:
        return None


@pytest.fixture
def served() -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Answering)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        joined(thread)
        server.server_close()


def test_an_address_is_fetched_with_when_it_was_written_and_a_refusal_or_too_long_an_answer_is_an_error(
    served: str,
) -> None:
    assert fetched(f"{served}/held?as=asked") == Answer(b"held", 1793715393.0)
    with pytest.raises(PinError, match="answered 404"):
        fetched(f"{served}/missing")
    with pytest.raises(PinError, match="1,048,577 bytes"):
        fetched(f"{served}/long")


def asking(answers: Mapping[str, Answer]) -> Asking:
    return Asking(answering(answers), lambda: NOW, lambda: PINS)


def test_the_check_reports_what_is_behind_and_reads_the_wheels_workflow_unless_told_another(
    capsys: pytest.CaptureFixture[str],
) -> None:
    newer = tags(("2026.11.20-1", NEW, 7 * DAY))
    assert main([], asking(current())) == 0
    assert main([], asking(current(tags=newer))) == 1
    assert "2026.11.20-1" in capsys.readouterr().out
    assert main(["more"], asking(current())) == 2
    assert "usage: python -m tools pins" in capsys.readouterr().err
    assert Asking().pinned() == read(WORKFLOW)
