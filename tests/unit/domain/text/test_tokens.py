import sys

from hypothesis import given, settings
from hypothesis import strategies as st

from sift.domain.text.fts5_table import FOLDS, SEPARATORS
from sift.domain.text.tokens import TRANSLATION, tokens
from tools.fts5_table import fts5_tokens

separators = [point for point, replaced in TRANSLATION.items() if replaced == " "]
folding = [point for point, replaced in TRANSLATION.items() if replaced != " "]
characters = st.one_of(
    st.characters(codec="utf-8"),
    st.sampled_from(separators).map(chr),
    st.sampled_from(folding).map(chr),
    st.characters(min_codepoint=0x300, max_codepoint=0x36F),
    st.characters(max_codepoint=0x7F),
)


@settings(max_examples=500)
@given(st.lists(st.text(characters, max_size=60), min_size=1, max_size=20))
def test_tokens_are_fts5_s(texts: list[str]) -> None:
    assert [tokens(text) for text in texts] == fts5_tokens(texts)


def test_words_are_folded_and_split_as_fts5_splits_them() -> None:
    assert tokens("Schrödinger's first-stage  R\N{LATIN CAPITAL LETTER E WITH ACUTE}SUMÉ\x00x") == [
        "schrodinger",
        "s",
        "first",
        "stage",
        "resume",
        "x",
    ]


def test_every_character_python_splits_at_is_a_separator() -> None:
    assert [point for point in range(sys.maxunicode + 1) if chr(point).isspace() and point not in separators] == []


def test_the_table_is_of_spans_and_folds_alone() -> None:
    assert all(word.count("=") == 1 for word in " ".join(FOLDS).split())
    assert all("=" not in word for word in " ".join(SEPARATORS).split())
