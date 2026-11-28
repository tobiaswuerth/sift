import threading
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import override

import numpy as np
import pytest

from sift.api.errors import ModelUnavailableError
from sift.api.search import SearchHybrid, SearchVectors
from sift.app.query import query
from sift.app.questions import RememberedQuestions
from sift.app.vsearch import vsearch
from sift.domain.codec.codes import Halves
from tests.homes import home_of
from tests.models import Hashed
from tests.waits import WAITED, until, waited


@pytest.fixture
def home(tmp_path: Path) -> Path:
    notes = {
        "notes/halo.md": "# Halo\n\nDark matter halo models.\n",
        "notes/brush.md": "# Brush\n\nA robot plans brush strokes.\n",
    }
    return home_of(tmp_path, notes, embedded=Hashed())


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
    assert models.prepare_many().notices == ("compiled",)


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


@dataclass(frozen=True, slots=True)
class Held(Hashed):
    entered: threading.Event = field(default_factory=threading.Event)
    let_go: threading.Event = field(default_factory=threading.Event)
    failing: bool = False

    @override
    def encode(self, texts: Sequence[str], *, query: bool) -> Halves:
        self.entered.set()
        waited(self.let_go)
        if self.failing and len(self.asked):
            raise ModelUnavailableError("the model is gone", "load it again")
        return Hashed.encode(self, texts, query=query)


def asked_alike(models: RememberedQuestions, texts: Sequence[str]) -> list[Future[Halves]]:
    pool = ThreadPoolExecutor(len(texts))
    try:
        return [pool.submit(models.encode, [text], query=True) for text in texts]
    finally:
        pool.shutdown(wait=False)


def test_questions_asked_while_one_is_embedded_are_embedded_together() -> None:
    held = Held()
    models = RememberedQuestions(held)
    first = asked_alike(models, ["one"])
    waited(held.entered)
    rest = asked_alike(models, ["two", "three", "two"])
    assert until(lambda: len(models.waiting) == 2)
    held.let_go.set()
    found = [each.result(WAITED) for each in (*first, *rest)]
    batches = [set(texts) for texts, _ in held.asked]
    assert (batches, len(held.asked[1][0])) == ([{"one"}, {"two", "three"}], 2)
    for text, vector in zip(["one", "two", "three", "two"], found, strict=True):
        assert np.array_equal(vector, Hashed().encode([text], query=True))
    assert models.encode(["three"], query=True) is found[2]
    assert (len(held.asked), models.waiting) == (2, {})


def test_every_question_embedded_together_is_told_when_the_model_fails() -> None:
    held = Held(failing=True)
    models = RememberedQuestions(held)
    first = asked_alike(models, ["one"])
    waited(held.entered)
    rest = asked_alike(models, ["two", "three"])
    assert until(lambda: len(models.waiting) == 2)
    held.let_go.set()
    assert first[0].result(WAITED).shape == (1, held.width)
    for each in rest:
        with pytest.raises(ModelUnavailableError, match="the model is gone"):
            each.result(WAITED)
    assert (models.waiting, len(models.vectors)) == ({}, 1)


def test_a_question_is_embedded_again_once_the_models_are_others() -> None:
    before, now = Hashed(), Hashed(name="other-words")
    models = RememberedQuestions(before)
    models.encode(["one"], query=True)
    models.models = now
    models.encode(["one"], query=True)
    assert (questions_asked(before), questions_asked(now)) == ([("one",)], [("one",)])
