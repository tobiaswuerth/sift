from sift.domain.query.expressions import NOTHING, Or, Phrase, Term
from sift.domain.query.grammar import any_phrase
from sift.domain.text.mentions import names, phrases


def test_a_note_is_named_by_its_file_its_aliases_its_title_and_the_fields_asked() -> None:
    fields = {"title": "Dr. J. Bühler", "aliases": ["JB"], "emails": ["jb@example.com", 3], "phone": "1"}
    assert names("People/Jürg Bühler.md", fields, ("emails",)) == [
        "Jürg Bühler",
        "JB",
        "Dr. J. Bühler",
        "jb@example.com",
    ]
    assert names("x/notes.txt", {}, ()) == ["notes"]


def test_names_are_looked_for_as_written_and_without_umlauts_if_long_enough_and_no_link() -> None:
    found = phrases(["Jürg  Bühler", "Straße", "AI", "--", "[[Acme AG]]", "Acme", "Acme"])
    assert found == ("Jürg Bühler", "Juerg Buehler", "Straße", "Strasse", "Acme")


def test_phrases_are_asked_as_any_of_them_once_each() -> None:
    assert any_phrase(["Hans Müller", "hans muller", "Acme"]) == Or((Phrase(("hans", "muller")), Term("acme")))
    assert any_phrase(["Acme"]) == Term("acme")
    assert any_phrase(["..."]) == NOTHING
