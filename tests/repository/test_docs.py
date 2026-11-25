import argparse
import re
import shlex
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from sift.app.importing import NEW_STORE
from sift.inference.accelerator import CUDA_FIX
from sift.inference.remote import MOST_TEXTS
from sift.interfaces.cli.entries import ENTRIES
from sift.interfaces.cli.logs import FOLDER as LOG_FOLDER
from sift.interfaces.cli.main import parser
from sift.interfaces.cli.updates import STATE
from sift.interfaces.cli.usage import UsageError
from sift.interfaces.server.endpoint import SERVER, SERVERS
from sift.interfaces.worker.app import MOST_BODY
from sift.interfaces.worker.keys import ADMISSION_FILE, CERTIFICATE, PRIVATE_KEY
from sift.interfaces.worker.keys import FOLDER as WORKER_FOLDER
from sift.settings import GPUS_VARIABLE, NO_COLOR, OFFLINE_VARIABLE, TERM, VARIABLES, Settings
from sift.storage.catalog import STORE_SUFFIX
from sift.storage.store_files import CATALOG, FILE_SUFFIXES, WRITER_SUFFIX
from tests.surface.commands import ran
from tests.tools.repositories import git_path

ROOT = Path(__file__).parents[2]
GUIDES = (ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")))
MARKDOWN = (*GUIDES, ROOT / "CHANGELOG.md")
OWN_PAGES = re.compile(r"https://github\.com/tobiaswuerth/sift/(?:blob|tree)/main/([^#\s]+)(#\S+)?")
FENCED = re.compile(r"^[ \t]*```(\w*)[^\n]*\n(.*?)^[ \t]*```", re.MULTILINE | re.DOTALL)
SHELLS = frozenset({"bash", "sh", "shell", "powershell", "pwsh"})
AGENT_FILES = frozenset({"agents.md", "claude.md", "gemini.md"})
EXITING = frozenset({"--help", "-h", "--version"})
SPANNED = re.compile(r"`([^`\n]+)`")
LINKED = re.compile(r"!?\[[^\]]*\]\(([^)\s]+)\)")
HEADING = re.compile(r"^#{1,6} +(.+?) *$", re.MULTILINE)
SAID = re.compile(r"^\s*(?:\w+=\S*\s+)*(?:\S*python\S* -m\s+)?sift(?=\s|$)(.*)")
FLAG = re.compile(r"^(--?[a-zA-Z][\w-]*)")
PLACEHOLDER = re.compile(r"^(?:<[^>]+>|[A-Z]+)$")
APART = re.compile(r"\|\||&&|[|;]")
READ_ALSO = (GPUS_VARIABLE, OFFLINE_VARIABLE, NO_COLOR, TERM)
SQLITE_FILES = ("", "-wal", "-shm")


def named(path: Path) -> str:
    return path.name


def found_in(pattern: re.Pattern[str], text: str) -> list[str]:
    found: list[str] = pattern.findall(text)
    return found


def blocks(text: str) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = FENCED.findall(text)
    return found


def code_of(text: str) -> Iterator[str]:
    yield from (line for _, block in blocks(text) for line in block.splitlines())
    yield from found_in(SPANNED, FENCED.sub("", text))


def examples(text: str) -> Iterator[str]:
    yield from (line for shell, block in blocks(text) if shell in SHELLS for line in block.splitlines())


def said(code: str) -> Iterator[str]:
    for part in APART.split(re.split(r"\s#\s", code)[0]):
        for head in (part, *part.split(" -- ")[1:]):
            found = SAID.match(head)
            if found is not None:
                yield found.group(1)


def words_of(rest: str) -> list[str]:
    return [each.strip("[]()<>,.:'\"").split("=", 1)[0] for each in rest.split()]


def offered(usage: str) -> frozenset[str]:
    return frozenset(re.findall(r"(?<![\w-])(--?[a-zA-Z][\w-]*)", usage.split("\n\n", 1)[0]))


@pytest.fixture(scope="module")
def options(tmp_path_factory: pytest.TempPathFactory) -> dict[str, frozenset[str]]:
    root = tmp_path_factory.mktemp("help")
    for name in ("user", "work"):
        (root / name).mkdir()
    settings = Settings(home=root / "home", user=root / "user", cwd=root / "work", state=root / "state")
    told = {name: ran(settings, name, "--help")[1] for name in ENTRIES}
    whole = offered(ran(settings, "--help")[1]) | {"--help"}
    return {"": whole} | {name: offered(usage) | whole - {"--version"} for name, usage in told.items()}


def unknown(words: list[str], options: dict[str, frozenset[str]]) -> list[str]:
    command = next((word for word in words if not word.startswith("-")), "")
    known = options.get("" if PLACEHOLDER.match(command) else command)
    if known is None:
        return [f"sift {command}"]
    flags = (FLAG.match(word) for word in words)
    return [found.group(1) for found in flags if found is not None and found.group(1) not in known]


def test_unknown_commands_and_flags_are_found_in_code_and_placeholders_pass() -> None:
    shell = "SIFT_HOME=/x sift serve --host h | jq\nsift nope  # sift --gone\nclaude mcp add sift -s user -- sift ui -x"
    text = f"Run `sift index --files a.md`, `cd sift`, then\n\n```sh\n{shell}\n```\n"
    known = {"": frozenset({"--help", "--json"}), "index": frozenset({"--file"}), "serve": frozenset({"--port"})}
    found = [unknown(words_of(rest), known) for code in code_of(text) for rest in said(code)]
    assert found == [["--host"], ["sift nope"], ["sift ui"], ["--files"]]
    assert unknown(["COMMAND", "--help"], known) == unknown(["<command>", "--json"], known) == []


@pytest.mark.parametrize("guide", GUIDES, ids=named)
def test_every_command_and_flag_a_guide_names_exists(guide: Path, options: dict[str, frozenset[str]]) -> None:
    code = code_of(guide.read_text(encoding="utf-8"))
    assert [name for line in code for rest in said(line) for name in unknown(words_of(rest), options)] == []


def refused(built: argparse.ArgumentParser, rest: str) -> str | None:
    words = shlex.split(rest, comments=True)
    if EXITING & set(words):
        return None
    try:
        built.parse_args(words)
    except UsageError as error:
        return str(error)
    return None


def test_every_example_a_guide_shows_is_taken_as_written(tmp_path: Path) -> None:
    built = parser(tmp_path / "user", tmp_path / "work")
    shown = [
        (guide.name, rest) for guide in GUIDES for line in examples(guide.read_text("utf-8")) for rest in said(line)
    ]
    told = {f"{name}: sift{rest}": refused(built, rest) for name, rest in shown}
    assert (len(shown) > 30, {example: why for example, why in told.items() if why is not None}) == (True, {})


def test_an_example_with_an_argument_sift_refuses_is_found(tmp_path: Path) -> None:
    built = parser(tmp_path / "user", tmp_path / "work")
    text = (
        "```bash\nsift search notes -k many\nsift get a.md --lines 3-4\nsift --version\n```\n```text\nsift nope\n```\n"
    )
    said_here = [rest for line in examples(text) for rest in said(line)]
    assert [refused(built, rest) is None for rest in said_here] == [False, True, True]


def test_the_readme_installs_cuda_pytorch_as_the_fix_sift_prints_does() -> None:
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    assert (CUDA_FIX in readme, re.findall(r"https://download\.pytorch\.org/whl/\w+", readme)) == (True, [])


def test_no_page_is_named_as_an_agent_s_instructions_are() -> None:
    listed = subprocess.run([git_path(), "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True, timeout=60)
    named = sorted(name for name in listed.stdout.decode().split("\0") if Path(name).name.lower() in AGENT_FILES)
    moved = (ROOT / "docs" / "agents.md").read_text(encoding="utf-8").splitlines()
    assert (named, len(moved) <= 6) == (["docs/agents.md"], True)


def anchors(path: Path) -> frozenset[str]:
    seen: dict[str, int] = {}
    for heading in found_in(HEADING, FENCED.sub("", path.read_text(encoding="utf-8"))):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        seen[slug if slug not in seen else f"{slug}-{seen[slug]}"] = 0
        seen[slug] = seen.get(slug, 0) + 1
    return frozenset(seen)


def broken(path: Path) -> Iterator[str]:
    prose = SPANNED.sub("", FENCED.sub("", path.read_text(encoding="utf-8")))
    for target in found_in(LINKED, prose):
        own = OWN_PAGES.fullmatch(target)
        if own is not None:
            place, anchor = ROOT / own.group(1), (own.group(2) or "")[1:]
        elif re.match(r"^[a-z]+:", target):
            continue
        else:
            written, _, anchor = target.partition("#")
            place = path if not written else path.parent / written
        if not place.exists() or (anchor and anchor not in anchors(place)):
            yield target


def test_a_link_is_broken_when_its_file_or_its_heading_is_missing(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("# Où `sift` lit\n\n## Où `sift` lit\n", encoding="utf-8")
    links = "[a](a.md#où-sift-lit-1) [b](a.md#gone) [c](gone.md) [d](#here) [e](https://example.org) `[f](x.md)`"
    (tmp_path / "b.md").write_text(f"# Here\n\n{links}\n\n```\n[g](y.md)\n```\n", encoding="utf-8")
    assert list(broken(tmp_path / "b.md")) == ["a.md#gone", "gone.md"]


@pytest.mark.parametrize("page", MARKDOWN, ids=named)
def test_every_link_of_a_page_leads_to_a_file_and_heading_that_exist(page: Path) -> None:
    assert list(broken(page)) == []


def test_the_readme_names_every_setting_sift_reads() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert [name for name in (*VARIABLES, *READ_ALSO) if f"`{name}`" not in readme] == []


def section(path: Path, heading: str) -> str:
    return path.read_text(encoding="utf-8").split(f"{heading}\n", 1)[1].split("\n## ", 1)[0]


def test_the_protocol_as_documented_says_how_many_texts_a_request_takes() -> None:
    protocol = section(ROOT / "docs" / "worker.md", "## Protocol")
    said = (f"at most {MOST_TEXTS:,} texts" in protocol, f"at most {MOST_BODY >> 20} MiB" in protocol)
    assert said == (True, True)


def kept_files() -> tuple[str, ...]:
    store = (f"NAME{STORE_SUFFIX}{suffix}" for suffix in FILE_SUFFIXES)
    catalog = (f"{CATALOG}{suffix}" for suffix in SQLITE_FILES)
    worker = (f"{WORKER_FOLDER}/{name}" for name in (ADMISSION_FILE, CERTIFICATE, PRIVATE_KEY))
    locks = (f"{name}{WRITER_SUFFIX}" for name in (SERVER, WORKER_FOLDER))
    named = (STATE, f"{SERVERS}/KEY.json", NEW_STORE.format(stem="NAME"), f"{LOG_FOLDER}/COMMAND-DATE.log")
    return (*catalog, *store, *worker, *locks, *named)


def test_every_file_sift_keeps_is_in_the_table_of_its_files_with_its_lifecycle() -> None:
    table = section(ROOT / "docs" / "usage.md", "### Where sift keeps its files")
    rows = [row.split(" | ") for row in table.splitlines() if row.startswith("| `")]
    told = [row[0] for row in rows if len(row) == 3 and row[2].strip(" |")]
    listed = {span.rsplit("/", 1)[-1] for cell in told for span in found_in(SPANNED, cell)}
    assert [name for name in kept_files() if name.rsplit("/", 1)[-1] not in listed] == []
