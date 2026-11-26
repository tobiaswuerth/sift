from sift.interfaces.web.marking import marked, snippet, sought


def test_words_are_marked_however_they_are_cased_or_accented_as_a_search_finds_them() -> None:
    terms = sought("Müller café")
    assert marked("Hans Müller at the Café, and a MULLER.", terms).text == (
        "Hans <mark>Müller</mark> at the <mark>Café</mark>, and a <mark>MULLER</mark>."
    )
    assert marked("Hans Müller", sought("muller")).text == "Hans <mark>Müller</mark>"


def test_a_word_is_marked_whole_from_where_a_word_sought_begins_it() -> None:
    assert marked("Reinforcement learning, relearned.", sought("learn")).text == (
        "Reinforcement <mark>learning</mark>, relearned."
    )


def test_nothing_is_marked_for_no_words_or_for_operators_and_what_is_marked_is_escaped() -> None:
    assert marked("a <b> and b", sought("")).text == "a &lt;b&gt; and b"
    assert marked("cats and dogs", sought("cats AND dogs")).text == "<mark>cats</mark> and <mark>dogs</mark>"
    assert marked("<dark>", sought("dark")).text == "&lt;<mark>dark</mark>&gt;"


def test_a_snippet_begins_shortly_before_the_first_word_found_whatever_its_accents() -> None:
    text = "filler " * 100 + "the Café at dusk " + "filler " * 100
    shown = snippet(text, sought("cafe")).text
    assert ("<mark>Café</mark>" in shown, shown.startswith("\N{HORIZONTAL ELLIPSIS}")) == (True, True)
    assert snippet("# Title\n\nbody text", sought("zzz"), "Title").text == "body text"
