from pathlib import Path

import numpy as np
import pytest

from sift.api.collections import AddCollection
from sift.api.embedding import EmbedCollections
from sift.api.indexing import IndexCollections
from sift.api.search import SearchHybrid, SearchVectors
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.indexing import index
from sift.app.query import query
from sift.app.questions import RememberedQuestions
from sift.app.vsearch import vsearch
from tests.files import write
from tests.models import Hashed


@pytest.fixture
def home(tmp_path: Path) -> Path:
    write(tmp_path / "notes" / "halo.md", "# Halo\n\nDark matter halo models.\n")
    write(tmp_path / "notes" / "brush.md", "# Brush\n\nA robot plans brush strokes.\n")
    home = tmp_path / "home"
    add(home, AddCollection(str(tmp_path / "notes")))
    index(home, IndexCollections(), lambda _: None)
    embed(home, EmbedCollections(), lambda _: None, Hashed())
    return home


def questions_asked(models: Hashed) -> list[tuple[str, ...]]:
    return [texts for texts, query in models.asked if query]


def test_a_question_asked_again_with_other_options_is_not_embedded_again(home: Path) -> None:
    hashed = Hashed()
    models = RememberedQuestions(hashed)
    first = vsearch(home, SearchVectors("robot strokes"), models)
    again = vsearch(home, SearchVectors("robot strokes", k=1), models)
    hybrid = query(home, SearchHybrid("robot strokes"), models)
    assert (again.hits, questions_asked(hashed)) == (first.hits[:1], [("robot strokes",)])
    assert hybrid == query(home, SearchHybrid("robot strokes"), Hashed())
    assert (models.name, models.prepare().notices, models.prepare_ranker().notices) == (hashed.name, (), ())


def test_texts_of_documents_and_several_questions_at_once_are_embedded_each_time() -> None:
    hashed = Hashed()
    models = RememberedQuestions(hashed)
    for _ in range(2):
        models.encode(["a document"], query=False)
        models.encode(["one", "two"], query=True)
    assert len(hashed.asked) == 4


def test_only_the_latest_questions_are_remembered_and_none_can_be_changed() -> None:
    hashed = Hashed()
    models = RememberedQuestions(hashed, kept=2)
    for text in ("one", "two", "one", "three", "one", "two"):
        vector = models.encode([text], query=True)
        assert not vector.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE"):
            vector.flags.writeable = True
    assert questions_asked(hashed) == [("one",), ("two",), ("three",), ("two",)]
    assert np.array_equal(models.encode(["one"], query=True), hashed.encode(["one"], query=True))
