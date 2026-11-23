import pytest

from sift.inference.hub import EMBED_MODEL, PINNED, hub, pinned


def test_a_pinned_model_is_named_and_loaded_at_its_commit() -> None:
    assert pinned(EMBED_MODEL) == f"{EMBED_MODEL}@{PINNED[EMBED_MODEL]}"
    assert hub(EMBED_MODEL) == (EMBED_MODEL, PINNED[EMBED_MODEL])
    assert hub(pinned(EMBED_MODEL)) == (EMBED_MODEL, PINNED[EMBED_MODEL])


@pytest.mark.parametrize(
    "name, found",
    [
        ("owner/model@v2.1", ("owner/model", "v2.1")),
        ("owner/model", ("owner/model", None)),
        ("C:/models/mine@home", ("C:/models/mine@home", None)),
        ("/data/models/mine", ("/data/models/mine", None)),
    ],
)
def test_another_model_is_loaded_at_the_revision_it_names_and_a_folder_whole(
    name: str, found: tuple[str, str | None]
) -> None:
    assert pinned(name) == name
    assert hub(name) == found
