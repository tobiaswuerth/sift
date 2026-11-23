import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from sift.settings import Settings
from tests.surface.commands import indexed_notes, ran
from tests.tiny_model import tiny_model, tiny_reranker


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("models")
    tiny_reranker(folder)
    return tiny_model(folder)


@pytest.fixture
def embedding(settings: Settings, tiny: Path) -> Settings:
    return dataclasses.replace(settings, embed_model=str(tiny), rerank_model=str(tiny.parent / "reranker"))


def test_embed_then_vsearch_finds_passages_near_the_query(embedding: Settings, notes: Path) -> None:
    indexed_notes(embedding, notes)
    assert ran(embedding, "index")[0] == 0
    code, out, err = ran(embedding, "embed", "--json")
    assert code == 0
    assert "embed          2 chunks" in err
    assert json.loads(out)["collections"][0]["outcome"] == {"stored": 2, "broken": 0, "state": "embedded"}
    code, out, err = ran(embedding, "vsearch", "brush", "-k", "1", "--compact")
    assert code == 0
    assert out.split("\n")[1].strip().startswith("notes/")
    assert " semantic " in err


def test_query_reranks_keyword_and_semantic_candidates(embedding: Settings, notes: Path) -> None:
    indexed_notes(embedding, notes)
    assert ran(embedding, "index")[0] == 0
    assert ran(embedding, "embed")[0] == 0
    code, out, err = ran(embedding, "query", "reward shaping", "-k", "2", "--json")
    assert code == 0
    assert {hit["id"] for hit in cast("dict[str, list[dict[str, str]]]", json.loads(out))["hits"]} == {
        "notes/rl.md",
        "notes/robots/brush.md",
    }
    code, _, err = ran(embedding, "query", "reward shaping", "-k", "1")
    assert (code, " hybrid " in err) == (0, True)


def test_vsearch_before_embedding_says_to_embed(embedding: Settings, notes: Path) -> None:
    indexed_notes(embedding, notes)
    assert ran(embedding, "index")[0] == 0
    code, out, err = ran(embedding, "vsearch", "brush")
    assert (code, out) == (1, "")
    assert err.split("\n")[:2] == ["error  notes: it holds no vectors yet", "  fix  run sift embed"]


def test_the_model_library_is_imported_only_once_a_text_is_embedded() -> None:
    script = (
        "import sys; from sift.inference.loading import deferred; m = deferred('google/embeddinggemma-2', 'r'); "
        "print(m.name, 'torch' in sys.modules)"
    )
    found = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True, timeout=60)
    assert found.stdout.split() == ["google/embeddinggemma-2@f9972ac22e696a105c482202646979e33e4b5a1e", "False"]
