import re
import subprocess
import zipfile
from pathlib import Path

from tests.tools.repositories import git_path

SAMPLES = Path(__file__).parents[1] / "samples"
FICTIONAL = frozenset({"Ada Example", "Bo Reader", "Di Copy"})
EXAMPLE_DOMAINS = re.compile(r"[\w.+-]+@example\.(com|org|net)")
TAGGED = re.compile(
    r"<(?:dc:creator|dc:contributor|cp:lastModifiedBy|meta:initial-creator|meta:creator)\b[^>]*?(?:/>|>([^<]*)<)"
)
WRITTEN = re.compile(
    r"/(?:Author|Creator)\s*\(([^)]*)\)|\\(?:author|operator) ([^}\\]*)|^(?:From|To|Cc): (.*)$", re.MULTILINE
)


def named_in_zip(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        texts = [archive.read(name).decode("utf-8") for name in archive.namelist() if name.endswith((".xml", ".opf"))]
    return {found.group(1) or "" for text in texts for found in TAGGED.finditer(text)}


def named_in_text(path: Path) -> set[str]:
    text = path.read_bytes().decode("latin-1")
    found = (next(each for each in match.groups() if each is not None) for match in WRITTEN.finditer(text))
    return {person.strip() for listed in found for person in listed.split(",")}


def people(path: Path) -> set[str]:
    named = named_in_zip(path) if zipfile.is_zipfile(path) else named_in_text(path)
    return {EXAMPLE_DOMAINS.sub("", each).strip(" <>") for each in named} - {""}


def test_the_people_a_sample_names_are_found_in_its_metadata(tmp_path: Path) -> None:
    with zipfile.ZipFile(tmp_path / "a.docx", "w") as archive:
        archive.writestr("docProps/core.xml", "<cp:lastModifiedBy>Some One</cp:lastModifiedBy><dc:creator/>")
    (tmp_path / "a.eml").write_text("From: Ann <ann@example.com>\nTo: bo@example.org, Cy Other <cy@example.net>\n")
    assert (people(tmp_path / "a.docx"), people(tmp_path / "a.eml")) == ({"Some One"}, {"Ann", "Cy Other"})


def test_samples_carry_only_fictional_names() -> None:
    named = {path.name: people(path) - FICTIONAL for path in sorted(SAMPLES.iterdir())}
    assert {name: found for name, found in named.items() if found} == {}


def test_every_file_git_holds_that_is_no_text_is_marked_binary() -> None:
    root = SAMPLES.parents[1]
    tracked = subprocess.run(
        [git_path(), "ls-files", "-z"], cwd=root, capture_output=True, check=True, timeout=60
    ).stdout
    paths = [
        name for name in tracked.decode("utf-8").split("\0") if name and b"\0" in (root / name).read_bytes()[:8000]
    ]
    asked = [git_path(), "check-attr", "binary", "--", *paths]
    said = subprocess.run(asked, cwd=root, capture_output=True, text=True, check=True, timeout=60)
    assert (len(paths) > 0, [line for line in said.stdout.splitlines() if not line.endswith(": binary: set")]) == (
        True,
        [],
    )
