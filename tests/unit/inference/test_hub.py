from pathlib import Path

import pytest

from sift.api.errors import ModelUnavailableError
from sift.inference.hub import EMBED_MODEL, PINNED, hub, pinned


def test_a_pinned_model_is_named_and_loaded_at_its_commit() -> None:
    assert pinned(EMBED_MODEL) == f"{EMBED_MODEL}@{PINNED[EMBED_MODEL]}"
    assert hub(EMBED_MODEL) == (EMBED_MODEL, PINNED[EMBED_MODEL])
    assert hub(pinned(EMBED_MODEL)) == (EMBED_MODEL, PINNED[EMBED_MODEL])


def test_another_model_is_loaded_at_the_revision_it_names_and_a_folder_whole(tmp_path: Path) -> None:
    assert (pinned("owner/model@v2.1"), hub("owner/model@v2.1")) == ("owner/model@v2.1", ("owner/model", "v2.1"))
    folder = tmp_path / "mine@home"
    folder.mkdir()
    assert (pinned(str(folder)), hub(str(folder))) == (str(folder), (str(folder), None))


@pytest.mark.parametrize("name", ["owner/model", "owner/model@", "C:/models/absent@home"])
def test_a_model_of_the_hub_named_without_a_revision_is_refused(name: str) -> None:
    with pytest.raises(ModelUnavailableError, match="names no revision") as refused:
        pinned(name)
    assert refused.value.remedy == f"name a commit of it, e.g. {name}@<commit> from its page under Files and versions"
