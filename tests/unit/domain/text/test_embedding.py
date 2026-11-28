import pytest

from sift.domain.text.embedding import embedded


@pytest.mark.parametrize(
    "title, text, first, wanted",
    [
        ("Halo", "Dark matter halos.", True, "Dark matter halos."),
        ("Halo", "Their profiles.", False, "Halo\n\nTheir profiles."),
        ("Halo", "## Halo profiles", False, "## Halo profiles"),
        ("Halo", "Halo again.", False, "Halo again."),
        ("", "Untitled.", False, "Untitled."),
    ],
)
def test_chunks_after_the_first_repeat_their_title_unless_they_start_with_it(
    title: str, text: str, *, first: bool, wanted: str
) -> None:
    assert embedded(title, text, first=first) == wanted
