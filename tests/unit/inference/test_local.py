from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from sift.inference.local import Local, local, width_of
from sift.inference.models import Models
from tests.models import Hashed
from tests.tiny_model import WIDTH, tiny_model


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tiny_model(tmp_path_factory.mktemp("models"))


def test_a_model_from_a_folder_embeds_texts_as_normalised_halves_in_their_order(tiny: Path) -> None:
    models = local(str(tiny))
    assert models.name == str(tiny)
    texts = ["dark matter halo models", "pruning", "skips ranges", "halo"]
    rows = models.encode(texts, query=False)
    assert (rows.shape, rows.dtype) == ((4, WIDTH), np.float16)
    assert np.allclose(np.linalg.norm(rows.astype(np.float32), axis=1), 1, atol=1e-2)
    alone = models.encode(texts[1:2], query=False)
    assert np.allclose(alone, rows[1:2], atol=1e-2)


def test_the_model_loads_once_when_first_needed(tiny: Path) -> None:
    models = Local(str(tiny))
    assert models.loaded is None
    models.prepare()
    loaded = models.loaded
    models.encode(["halo"], query=True)
    assert models.loaded is loaded


def hashed(_: Path) -> Models:
    return Hashed()


def local_in(folder: Path) -> Models:
    return local(str(folder))


@pytest.mark.parametrize("make", [hashed, local_in])
def test_the_fake_keeps_the_contract_the_local_models_keep(tiny: Path, make: Callable[[Path], Models]) -> None:
    models = make(tiny)
    for query in (False, True):
        rows = models.encode(["dark matter", "halo models", "pruning"], query=query)
        assert rows.dtype == np.float16
        assert rows.shape[0] == 3
        assert np.allclose(np.linalg.norm(rows.astype(np.float32), axis=1), 1, atol=1e-2)
        assert models.encode([], query=query).shape == (0, rows.shape[1])
    models.prepare()
    assert models.name


def test_a_model_that_says_not_how_wide_its_vectors_are_is_refused() -> None:
    assert width_of("m", 768) == 768
    with pytest.raises(ValueError, match="m says not how wide its vectors are"):
        width_of("m", None)
