from sift.domain.query.expressions import NOTHING, Or, Phrase, Term
from sift.domain.query.grammar import any_phrase, words_in
from sift.domain.text.mentions import names, phrases


def test_a_note_is_named_by_its_file_its_aliases_its_title_and_the_fields_asked() -> None:
    fields = {"title": "Dr. J. Bühler", "aliases": ["JB"], "emails": ["jb@example.com", 3], "phone": "1"}
    assert names("People/Jürg Bühler.md", "", "Jürg Bühler", fields, ("emails",)) == [
        "Jürg Bühler",
        "JB",
        "Dr. J. Bühler",
        "jb@example.com",
    ]
    assert names("x/notes.txt", "", "Notes", {}, ()) == ["notes"]


def test_a_record_is_named_by_its_title_not_by_its_file() -> None:
    assert names("R/papers.jsonl", "p1", "Graph Kernels", {"title": "Graph Kernels"}, ()) == ["Graph Kernels"]


def test_names_are_looked_for_as_written_and_without_umlauts_if_long_enough_and_no_link() -> None:
    found = phrases(["Jürg  Bühler", "Straße", "AI", "C++", "--", "[[Acme AG]]", "Acme", "Acme"])
    assert found == ("Jürg Bühler", "Juerg Buehler", "Straße", "Strasse", "Acme")


def test_phrases_are_asked_as_any_of_them_once_each() -> None:
    assert any_phrase(["Hans Müller", "hans muller", "Acme"]) == Or((Phrase(("hans", "muller")), Term("acme")))
    assert any_phrase(["Acme"]) == Term("acme")
    assert any_phrase(["..."]) == NOTHING


def test_names_of_three_characters_capitals_and_links_inside_are_looked_for() -> None:
    assert phrases(["Bob", "Öl", "Meeting with [[Bob]]"]) == ("Bob", "Öl", "Oel", "Meeting with [[Bob]]")
    assert phrases(["Köln", "ÄÖÜ ß"]) == ("Köln", "Koeln", "ÄÖÜ ß", "AeOeUe ss")


def test_a_name_written_with_combining_marks_is_looked_for_without_umlauts_too() -> None:
    decomposed = "Hans Mu" + chr(0x308) + "ller"
    assert phrases([decomposed]) == (decomposed, "Hans Mueller")


def test_the_words_of_names_are_counted_as_a_query_s() -> None:
    assert words_in(any_phrase([f"name{n} word" for n in range(501)])) == 1002
    assert words_in(any_phrase(["one", "two words", "one"])) == 3


def test_a_name_of_two_wide_characters_is_long_enough_to_look_for() -> None:
    assert phrases(["東京", "王芳", "東"]) == ("東京", "王芳")
    assert phrases(["ab"]) == ()
