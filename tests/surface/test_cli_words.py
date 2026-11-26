from pathlib import Path

from sift.api.events import Phase, PhaseDone, PhaseStarted
from sift.interfaces.console.progress import Looks, View, shown
from sift.interfaces.console.style import PLAIN
from sift.settings import Settings
from tests.surface.commands import ran


def _notes(settings: Settings, notes: Path) -> None:
    (notes / "a.md").write_bytes(b"# A\n\nalpha [[Bob]]\n")
    (notes / "Bob.md").write_bytes(b"# Bob\n\nbeta\n")
    assert ran(settings, "add", str(notes))[0] == 0


def test_passages_are_called_passages_wherever_they_are_counted(settings: Settings, notes: Path) -> None:
    _notes(settings, notes)
    assert ran(settings, "index")[0] == 0
    header = ran(settings, "status")[1].splitlines()[0].split()
    assert ("passages" in header, "chunks" in header) == (True, False)
    view, _ = shown(View(), PhaseStarted(Phase.EMBED, 2), 0.0, Looks(PLAIN, "linux"))
    assert shown(view, PhaseDone(2), 1.0, Looks(PLAIN, "linux"))[1][0].split()[1:3] == ["2", "passages"]


def test_an_unknown_id_is_said_once_and_alike_by_every_command(settings: Settings, notes: Path) -> None:
    _notes(settings, notes)
    assert ran(settings, "index")[0] == 0
    got = ran(settings, "get", "alice.md")[2].splitlines()[0]
    linked = ran(settings, "links", "alice.md")[2].splitlines()[0]
    assert (got, linked) == ("error  no note or document 'alice.md'", "error  ID: no note or document 'alice.md'")


def test_what_links_into_a_note_is_headed_in_lower_case(settings: Settings, notes: Path) -> None:
    _notes(settings, notes)
    assert ran(settings, "index")[0] == 0
    out = ran(settings, "links", "notes/Bob.md", "--direction", "in")[1]
    assert out.startswith("linking to Bob (notes/Bob.md)")


def test_the_files_index_was_named_are_labelled(settings: Settings, notes: Path) -> None:
    _notes(settings, notes)
    err = ran(settings, "index", "--file", str(notes / "a.md"))[2]
    assert "\n file  notes/a.md: added\n" in err


def test_a_move_says_where_a_store_was_and_where_it_went(settings: Settings, notes: Path, tmp_path: Path) -> None:
    _notes(settings, notes)
    before = settings.home / "collections" / "notes.sqlite"
    err = ran(settings, "move", "notes", str(tmp_path / "stores"))[2]
    assert err.splitlines() == [f"notes  {before}", f"moved  notes  {tmp_path / 'stores' / 'notes.sqlite'}"]
