from collections.abc import Sequence

import pytest

from sift.api.ids import DocumentId
from sift.domain.text.links import LinkKind, name_key
from sift.domain.text.resolving import UNRESOLVED, Candidate, Resolution, Via, link_path, resolved


def candidate(at: int, path: str) -> Candidate:
    return Candidate(DocumentId(at), path)


def via(how: Via, *documents: int) -> Resolution:
    return Resolution(how, tuple(map(DocumentId, documents)))


WIKI, MARKDOWN = LinkKind.WIKILINK, LinkKind.MARKDOWN
PATHS = ("People/Hans.md", "Archive/People/Hans.md", "Projects/Hans.md", "Projects/Plan.md", "A/Dup.md", "B/Dup.md")
HANS = [candidate(at, path) for at, path in enumerate(PATHS[:3])]
DUP = [candidate(4, "A/Dup.md"), candidate(5, "B/Dup.md")]
NONE: tuple[Candidate, ...] = ()


@pytest.mark.parametrize(
    "source, link, candidates, expected",
    [
        ("Projects/Plan.md", (WIKI, "Hans"), (HANS, NONE), via(Via.NAME, 2)),
        ("Projects/Plan.md", (WIKI, "People/Hans"), (HANS, NONE), via(Via.PATH, 0)),
        ("Projects/Plan.md", (WIKI, "Archive/People/Hans"), (HANS, NONE), via(Via.PATH, 1)),
        ("Projects/Sub/Note.md", (WIKI, "Hans"), (HANS, NONE), via(Via.AMBIGUOUS, 0, 1, 2)),
        ("Projects/Sub/Note.md", (WIKI, "archive/people/hans.md"), (HANS, NONE), via(Via.PATH, 1)),
        ("Projects/Sub/Note.md", (WIKI, "Nope/Hans"), (HANS, NONE), UNRESOLVED),
        ("Other.md", (WIKI, "Hansi"), (NONE, [HANS[0]]), via(Via.ALIAS, 0)),
        ("Other.md", (WIKI, "Dup"), (DUP, [candidate(9, "C/X.md")]), via(Via.AMBIGUOUS, 4, 5)),
        (
            "Other.md",
            (WIKI, "X"),
            (NONE, [candidate(7, "a.md"), candidate(8, "b.md")]),
            via(Via.AMBIGUOUS, 7, 8),
        ),
        ("n.md", (WIKI, "nobody"), (NONE, NONE), UNRESOLVED),
        ("Projects/Plan.md", (WIKI, "Plan"), ([candidate(3, "Projects/Plan.md")], NONE), via(Via.NAME, 3)),
        (None, (WIKI, "Hans"), (HANS, NONE), via(Via.AMBIGUOUS, 0, 1, 2)),
        ("Notes/Plan.md", (MARKDOWN, "../People/Hans.md"), (HANS, NONE), via(Via.PATH, 0)),
        ("Notes/Plan.md", (MARKDOWN, "People/Hans.md"), (HANS, NONE), via(Via.PATH, 0)),
        ("Notes/Plan.md", (MARKDOWN, "./Missing.md"), (NONE, NONE), UNRESOLVED),
        ("Notes/Plan.md", (MARKDOWN, "../../outside.md"), (HANS, NONE), UNRESOLVED),
        (
            "n.md",
            (WIKI, "People/Hans"),
            ([candidate(0, "x/People/Hans.md"), candidate(1, "y/People/Hans.md")], NONE),
            via(Via.AMBIGUOUS, 0, 1),
        ),
    ],
)
def test_a_link_resolves_as_obsidian_resolves_it(
    source: str | None,
    link: tuple[LinkKind, str],
    candidates: tuple[Sequence[Candidate], Sequence[Candidate]],
    expected: Resolution,
) -> None:
    assert resolved(source, *link, *candidates) == expected


@pytest.mark.parametrize(
    "target, expected",
    [("Maße", 1), ("MASSE", 0), ("masse", 0), ("Straße", 3), ("strasse", 2), ("ﬁle", 5), ("FILE", 4)],
)
def test_names_casefolding_merges_are_told_apart_by_their_case_insensitive_spelling(target: str, expected: int) -> None:
    named = [
        candidate(0, "T/Masse.md"),
        candidate(1, "T/Maße.md"),
        candidate(2, "U/Strasse.md"),
        candidate(3, "U/Straße.md"),
        candidate(4, "V/file.md"),
        candidate(5, "V/ﬁle.md"),
    ]
    alike = [each for each in named if name_key(each.path.rsplit("/", 1)[-1]) == name_key(target)]
    assert resolved("n.md", WIKI, target, alike, []) == via(Via.NAME, expected)


def test_paths_compare_folded_and_a_note_s_without_its_extension() -> None:
    assert name_key("Hans Mu\N{COMBINING DIAERESIS}ller.MD") == name_key("hans müller") == "hans müller"
    assert name_key("notes.txt") == "notes.txt"
    assert link_path("Notes/Plan.md", MARKDOWN, "../People/Hans.md") == "People/Hans.md"
    assert link_path("Notes/Plan.md", WIKI, "/People/Hans") == "People/Hans"
