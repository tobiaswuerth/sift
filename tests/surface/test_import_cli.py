import dataclasses
import json
from pathlib import Path
from typing import cast

from sift.settings import Settings
from tests.files import write
from tests.former_homes import PINNED_FAKE, former_home, hashed
from tests.surface.commands import ran

NOTE = " note  this home holds the index of sift 1.x, which sift 2 reads once it is imported\n next  sift import\n"


def test_sift_import_brings_a_home_of_sift_1_in_and_shows_its_counts(settings: Settings, tmp_path: Path) -> None:
    settings = dataclasses.replace(settings, embed_model=PINNED_FAKE)
    write(tmp_path / "notes" / "rl.md", "# Reward shaping\n\nShaped rewards.\n")
    former = former_home(tmp_path, {"notes": tmp_path / "notes"}, hashed())
    code, out, err = ran(settings, "import", "--home", str(former))
    heading, row, *_ = out.split("\n")
    assert (code, heading.split(), row.split()) == (
        0,
        ["collection", "documents", "passages", "vectors"],
        ["notes", "1", "->", "1", "1", "->", "1", "1", "->", "1"],
    )
    assert err.endswith(
        " kept  sift 1.x's files are as they were: remove its stores and sift.db once these are as you want them\n"
    )
    code, out, _ = ran(settings, "import", "--home", str(former), "-c", "notes", "--json")
    (one,) = cast("list[dict[str, object]]", cast("dict[str, object]", json.loads(out))["collections"])
    assert (code, one["name"], cast("dict[str, object]", one["outcome"])["vectors"]) == (0, "notes", 1)
    code, out, err = ran(settings, "import", "--home", str(former), "--color", "always")
    assert (code, out.split("\n")[0], "\x1b[" in out, "\x1b[" in err) == (0, heading, False, True)


def test_a_folder_that_holds_no_index_of_sift_1_is_said_to_hold_none(settings: Settings, tmp_path: Path) -> None:
    code, out, err = ran(settings, "import", "--home", str(tmp_path))
    said = f"error  --home: {tmp_path} holds no index of sift 1.x: there is no sift.db in it\n"
    assert (code, out, err) == (1, "", f"{said}  fix  name the folder sift 1.x kept its index in\n")


def test_a_home_holding_the_index_of_sift_1_alone_is_said_to_need_importing(settings: Settings, tmp_path: Path) -> None:
    write(tmp_path / "notes" / "rl.md", "# Reward shaping\n")
    former = former_home(tmp_path, {"notes": tmp_path / "notes"})
    there = dataclasses.replace(settings, home=former)
    code, _, err = ran(there, "status")
    assert (code, err.startswith(NOTE)) == (0, True)
    assert (ran(there, "import")[0], NOTE in ran(there, "status")[2]) == (0, False)


def test_an_import_that_skips_a_collection_says_why_and_fails(settings: Settings, tmp_path: Path) -> None:
    write(tmp_path / "notes" / "rl.md", "# Reward shaping\n")
    former = former_home(tmp_path, {"notes": tmp_path / "notes"})
    (tmp_path / "notes" / "rl.md").unlink()
    (tmp_path / "notes").rmdir()
    code, out, err = ran(settings, "import", "--home", str(former))
    fixed = err.split("\n")[1].startswith("  fix  bring its folder back, then import it again")
    assert (code, out, fixed) == (1, "", True)
