import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.domain.text.links import Link, LinkKind, Name, NameKind, aliases, name_key, names
from sift.domain.text.notes import note

type Found = tuple[str | None, int, LinkKind, str, str | None, str | None]


def found(text: str) -> list[Link]:
    return list(note(text.encode(), "n.md").links)


def written(text: str) -> list[Found]:
    return [(link.field, link.line, link.kind, link.target, link.fragment, link.shown) for link in found(text)]


WIKI, EMBED, MARKDOWN = LinkKind.WIKILINK, LinkKind.EMBED, LinkKind.MARKDOWN


@pytest.mark.parametrize(
    "link, expected",
    [
        ("[[Hans Müller]]", (WIKI, "Hans Müller", None, None)),
        ("[[Hans Müller.md]]", (WIKI, "Hans Müller.md", None, None)),
        ("[[01_People/Hans Müller]]", (WIKI, "01_People/Hans Müller", None, None)),
        ("[[Hans Müller|Hans]]", (WIKI, "Hans Müller", None, "Hans")),
        ("| [[Hans Müller\\|Hans]] |", (WIKI, "Hans Müller", None, "Hans")),
        ("[[Hans Müller#Kontakt]]", (WIKI, "Hans Müller", "Kontakt", None)),
        ("[[Hans Müller#^ab12cd|Hans]]", (WIKI, "Hans Müller", "^ab12cd", "Hans")),
        ("[[ Hans Müller ]]", (WIKI, "Hans Müller", None, None)),
        ("[[Release v1.2]]", (WIKI, "Release v1.2", None, None)),
        ("![[contract.pdf]]", (EMBED, "contract.pdf", None, None)),
        ("![[Hans Müller#Kontakt]]", (EMBED, "Hans Müller", "Kontakt", None)),
        ("[text](Other%20Note.md)", (MARKDOWN, "Other Note.md", None, "text")),
        ("[text](<Other Note.md>)", (MARKDOWN, "Other Note.md", None, "text")),
        ("[text](../People/Hans.md#Kontakt)", (MARKDOWN, "../People/Hans.md", "Kontakt", "text")),
        ('[text](Note.md "A title")', (MARKDOWN, "Note.md", None, "text")),
        ("[see [1]](Note%20(old).md)", (MARKDOWN, "Note (old).md", None, "see [1]")),
        ("![scan](scan.pdf)", (MARKDOWN, "scan.pdf", None, "scan")),
        ("[](Note.md)", (MARKDOWN, "Note.md", None, None)),
    ],
)
def test_links_in_a_note(link: str, expected: tuple[LinkKind, str, str | None, str | None]) -> None:
    assert written(f"Before {link} after.\n") == [(None, 1, *expected)]


@pytest.mark.parametrize(
    "text",
    [
        "[[#Kontakt]]",
        "[[]]",
        "[[ |shown]]",
        "[[Folder/]]",
        "![[diagram.png]]",
        "![alt](img/diagram.PNG)",
        "[text](https://example.com/a.md)",
        "[text](mailto:hans@example.com)",
        "[text](obsidian://open?vault=x)",
        "[text](#heading)",
        "[text](//example.com/a.md)",
        "[text](..)",
        "[two words](Some Note.md)",
        "`[[Not a link]]`",
        "``a ` [[Not a link]] b``",
    ],
)
def test_what_is_no_link(text: str) -> None:
    assert written(f"{text}\n") == []


def test_links_in_code_blocks_are_no_links() -> None:
    text = "```\n[[Also not a link]]\n```\n~~~python\n[x](y.md)\n```\n[[still code]]\n~~~\n[[Real]]\n"
    assert written(text) == [(None, 9, WIKI, "Real", None, None)]


@pytest.mark.parametrize(
    "text, line",
    [
        ("> ```bash\n> if [[ -f x ]]; then\n> ```\n[[Real]]\n", 4),
        (">> ~~~\n>> [[code]]\n>> ~~~\n> [[Real]]\n", 4),
        ("- an item\n\n    ```bash\n    if [[ -f x ]]; then\n    ```\n- [[Real]]\n", 6),
        ("1. an item\n\t```\n\tx\t[[code]]\n\t```\n[[Real]]\n", 5),
        ("> ```\n> [[code]]\n\n[[Real]]\n", 4),
        ("- an item\n    ```\n    [[code]]\n\n[[Real]]\n", 5),
    ],
)
def test_links_in_code_blocks_of_a_quote_or_a_list_are_no_links(text: str, line: int) -> None:
    assert written(text) == [(None, line, WIKI, "Real", None, None)]


def test_a_code_block_never_closed_at_the_margin_goes_to_the_end() -> None:
    assert written("```\n[[code]]\n\n    [[still code]]\n> [[and this]]\n") == []
    assert written("   ```\n[[code]]\n\n[[still code]]\n") == []


def test_three_backticks_with_one_after_them_open_no_code_block() -> None:
    assert [link.target for link in found("```inline``` is code\n\nSee [[Target]].\n")] == ["Target"]
    assert written("~~~ `x`\n[[Hidden]]\n~~~\n[[Real]]\n") == [(None, 4, WIKI, "Real", None, None)]


def test_an_unmatched_backtick_hides_nothing() -> None:
    assert written("It's 5` long, see [[Real]] and ``[[code]]``.\n") == [(None, 1, WIKI, "Real", None, None)]


def test_several_links_on_a_line_in_order() -> None:
    kinds = [(link.kind, link.target) for link in found("[b](B.md) and [[A]], ![[C.pdf]] and [[D]] [d](D.md)\n")]
    assert kinds == [(MARKDOWN, "B.md"), (WIKI, "A"), (EMBED, "C.pdf"), (WIKI, "D"), (MARKDOWN, "D.md")]


def test_a_wikilink_in_a_markdown_link_counts_once() -> None:
    assert written("[[A]](B.md)\n") == [(None, 1, WIKI, "A", None, None)]


def test_links_in_front_matter() -> None:
    text = (
        "---\n"
        'org: "[[Acme AG]]"\n'
        "reports_to: [[Peter Frei]]\n"
        'colleague_of: ["[[Anna Keller]]", "[[Peter Frei|Peter]]"]\n'
        "team:\n"
        '  - "[[Hans Müller]]"\n'
        "  - not a link\n"
        "# also: [[Commented out]]\n"
        "contacts:\n"
        '  primary: "[[Hans Müller#Kontakt]]"\n'
        "url: https://example.com/[x](y.md)\n"
        "---\n"
        "Body with [[Acme AG]].\n"
    )
    assert written(text) == [
        ("org", 2, WIKI, "Acme AG", None, None),
        ("reports_to", 3, WIKI, "Peter Frei", None, None),
        ("colleague_of", 4, WIKI, "Anna Keller", None, None),
        ("colleague_of", 4, WIKI, "Peter Frei", None, "Peter"),
        ("team", 6, WIKI, "Hans Müller", None, None),
        ("contacts", 10, WIKI, "Hans Müller", "Kontakt", None),
        (None, 13, WIKI, "Acme AG", None, None),
    ]


def test_link_keys_compare_names_as_obsidian_does() -> None:
    keys = [
        link.key
        for link in found("[[Hans Mu\N{COMBINING DIAERESIS}ller]] [[hans müller.md]] [[01_People/HANS MÜLLER#x]]")
    ]
    assert keys == ["hans müller"] * 3
    assert [link.key for link in found("![[Contract.PDF]] [[St.Gallen]]")] == ["contract.pdf", "st.gallen"]


@pytest.mark.parametrize(
    "line", ["[" * 200_000, "[[" * 100_000, "[a](" * 50_000, "`" * 100_000 + "x", "[x] " * 50_000 + "](y"]
)
def test_no_link_is_found_on_lines_that_make_a_careless_pattern_backtrack(line: str) -> None:
    assert found(line + "\n" + "](\n") == []


def test_a_note_answers_to_its_file_name_and_its_aliases() -> None:
    assert names("People/Hans Müller.md", {"aliases": ["H. Müller", " ", 3], "alias": "Hansi"}) == (
        Name("h. müller", NameKind.ALIAS),
        Name("hans müller", NameKind.NAME),
        Name("hansi", NameKind.ALIAS),
    )
    assert aliases({"aliases": None}) == []
    assert names("report.pdf", {}) == (Name("report.pdf", NameKind.NAME),)


def test_a_name_s_key_leaves_out_a_markdown_extension_alone() -> None:
    assert [name_key(n) for n in ("Note.md", "Note.MARKDOWN", "data.csv", "Mu\N{COMBINING DIAERESIS}ller")] == [
        "note",
        "note",
        "data.csv",
        "müller",
    ]


def test_a_link_in_a_list_item_holding_a_key_belongs_to_the_list_s_field() -> None:
    links = found("---\nrelated:\n  - see: [[Alice]]\n# owner: [[Bob]]\n---\n")
    assert [(link.field, link.line, link.target) for link in links] == [("related", 3, "Alice")]


@pytest.mark.parametrize(
    "head, fields",
    [
        ("org: x\n-x: [[Acme]]", [("-x", "Acme")]),
        ("related:\n- see: [[Alice]]", [("related", "Alice")]),
        ("e\N{COMBINING ACUTE ACCENT}: [[Alice]]", [("\N{LATIN SMALL LETTER E WITH ACUTE}", "Alice")]),
        (
            "e\N{COMBINING ACUTE ACCENT}: [[Alice]]\n\N{LATIN SMALL LETTER E WITH ACUTE}: [[Bob]]",
            [("e\N{COMBINING ACUTE ACCENT}", "Alice"), ("\N{LATIN SMALL LETTER E WITH ACUTE}", "Bob")],
        ),
    ],
    ids=["a key starting with a dash", "an item at the margin", "a decomposed key", "a decomposed key beside its twin"],
)
def test_a_front_matter_link_s_field_is_the_field_its_line_is_read_into(
    head: str, fields: list[tuple[str, str]]
) -> None:
    document = note(f"---\n{head}\n---\n".encode(), "n.md")
    assert [(link.field, link.target) for link in document.links] == fields
    assert {field for field, _ in fields} <= set(document.fields)


KEYS = st.sampled_from(
    ["a", "-a", "- a", "#a", "[[a]]", "\N{LATIN SMALL LETTER E WITH ACUTE}", "e\N{COMBINING ACUTE ACCENT}"]
)
VALUES = st.sampled_from(["", " [[a]]", " x [[a]]", " [[[a]]]", " |", " [", " ]", " {", " # [[a]]"])
LEADS = st.sampled_from(["", "  ", "- ", "  - ", "#"])
head_lines = st.lists(st.tuples(LEADS, KEYS, st.sampled_from([":", ""]), VALUES).map("".join), max_size=8)


@given(head_lines)
def test_the_field_of_every_front_matter_link_is_a_field_of_the_note(lines: list[str]) -> None:
    document = note(("---\n" + "\n".join(lines) + "\n---\nbody\n").encode(), "n.md")
    assert {link.field for link in document.links} - {None} <= set(document.fields)
