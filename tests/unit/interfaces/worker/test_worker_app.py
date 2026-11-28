import json
import threading
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field

import numpy as np
import pytest

from sift.domain.arrays import floats
from sift.domain.codec.codes import Floats, Halves
from sift.domain.text.embedding import RECIPE
from sift.inference.prepared import ModelsPrepared
from sift.inference.remote import MOST_TEXTS
from sift.interfaces.worker.app import MOST_BODY
from sift.settings import Settings
from tests.models import Hashed
from tests.serving import VERSION
from tests.workers import LiveWorker, asked, working

TEXTS = ["brush strokes", "pigments"]


@pytest.fixture
def models() -> Hashed:
    return Hashed()


@pytest.fixture
def worker(settings: Settings, models: Hashed) -> Generator[LiveWorker]:
    with working(settings, models) as live:
        yield live


def test_the_worker_says_which_models_it_runs_in_which_sift(worker: LiveWorker, settings: Settings) -> None:
    status, said = asked(worker, "GET", "/")
    expected = {"model": "hashed-words", "width": 48, "reranker": settings.rerank_model, "version": VERSION}
    assert (status, json.loads(said)) == (200, {**expected, "recipe": RECIPE})


def test_the_worker_embeds_texts_and_queries_as_this_machine_would(worker: LiveWorker, models: Hashed) -> None:
    for query in (False, True):
        status, said = asked(worker, "POST", "/embed", {"texts": TEXTS, "query": query})
        assert status == 200
        assert np.frombuffer(said, np.float16).tolist() == Hashed().encode(TEXTS, query=query).ravel().tolist()
    asked(worker, "POST", "/embed", {"texts": TEXTS})
    assert models.asked[-3:] == [(tuple(TEXTS), False), (tuple(TEXTS), True), (tuple(TEXTS), False)]


def test_the_worker_reranks_as_this_machine_would(worker: LiveWorker) -> None:
    texts = ["brush strokes on canvas", "reward shaping"]
    status, said = asked(worker, "POST", "/rerank", {"query": "brush strokes", "texts": texts})
    assert (status, json.loads(said)) == (200, {"scores": floats(Hashed().rerank("brush strokes", texts))})


@pytest.mark.parametrize(
    "path, body",
    [
        ("/embed", b'{"texts": "not a list"}'),
        ("/embed", b'{"texts": [1, 2]}'),
        ("/embed", b'{"texts": ["a"], "query": "yes"}'),
        ("/embed", b"[]"),
        ("/embed", b"not json"),
        ("/embed", b"\xff\xfe"),
        ("/embed", b"[" * 100_000),
        ("/embed", b'{"texts": [[[["a"]]]]}'),
        ("/embed", b'{"texts": ["half \\ud800 of a pair"]}'),
        ("/rerank", b'{"query": "\\udfff", "texts": ["a"]}'),
        ("/rerank", b'{"query": "q", "texts": ["a", "\\ud83d"]}'),
        ("/rerank", b'{"texts": ["a"]}'),
        ("/rerank", b'{"query": 1, "texts": ["a"]}'),
        ("/rerank", b'{"query": "q", "texts": "a"}'),
    ],
)
def test_the_worker_refuses_a_request_of_another_shape_and_says_which_it_takes(
    worker: LiveWorker, path: str, body: bytes
) -> None:
    status, said = asked(worker, "POST", path, body)
    shape = '{"texts": [...], "query": false}' if path == "/embed" else '{"query": "...", "texts": [...]}'
    assert (status, said.decode()) == (400, f"send {shape}")


@pytest.mark.parametrize("path, asking", [("/embed", {}), ("/rerank", {"query": "q"})])
def test_the_worker_refuses_more_texts_than_a_sift_sends_at_once(
    worker: LiveWorker, path: str, asking: dict[str, object]
) -> None:
    status, said = asked(worker, "POST", path, {**asking, "texts": ["a"] * (MOST_TEXTS + 1)})
    assert (status, said.decode()) == (413, f"send at most {MOST_TEXTS} texts at a time, not {MOST_TEXTS + 1}")
    assert asked(worker, "POST", path, {**asking, "texts": ["a"] * MOST_TEXTS})[0] == 200


def test_the_worker_takes_a_whole_batch_of_long_passages(worker: LiveWorker) -> None:
    passages = [f"passage {n} " + "brush strokes on canvas " * 400 for n in range(MOST_TEXTS)]
    status, said = asked(worker, "POST", "/embed", {"texts": passages})
    assert (status, len(said)) == (200, MOST_TEXTS * 48 * 2)


def test_the_worker_refuses_a_request_longer_than_it_reads(worker: LiveWorker) -> None:
    status, said = asked(worker, "POST", "/embed", b" " * (MOST_BODY + 1))
    assert (status, said.decode()) == (413, f"send at most {MOST_BODY:,} bytes at a time")


@pytest.mark.parametrize("authorization", ["", "Bearer wrong", "Basic {token}", "{token}", "Bearer {token}x"])
def test_the_worker_answers_only_requests_with_its_token(worker: LiveWorker, authorization: str) -> None:
    given = authorization.format(token=worker.running.key.token)
    for method, path, body in (("GET", "/", None), ("POST", "/embed", {"texts": ["x"]})):
        assert asked(worker, method, path, body, authorization=given) == (401, b"wrong key")
    assert asked(worker, "POST", "/rerank", {"query": "x", "texts": ["x"]}, authorization=given)[0] == 401


@dataclass(frozen=True, slots=True)
class Threaded:
    hashed: Hashed = field(default_factory=Hashed)
    threads: list[str] = field(default_factory=list[str])

    @property
    def name(self) -> str:
        return self.hashed.name

    def prepare(self) -> ModelsPrepared:
        return self.hashed.prepare()

    def prepare_many(self) -> ModelsPrepared:
        return self.hashed.prepare_many()

    def prepare_ranker(self) -> ModelsPrepared:
        return self.hashed.prepare_ranker()

    def release_idle(self) -> bool:
        return False

    def encode(self, texts: Sequence[str], *, query: bool) -> Halves:
        self.threads.append(threading.current_thread().name)
        return self.hashed.encode(texts, query=query)

    def rerank(self, query: str, texts: Sequence[str]) -> Floats:
        self.threads.append(threading.current_thread().name)
        return self.hashed.rerank(query, texts)


def test_the_models_answer_on_one_thread_of_their_own_one_request_at_a_time(settings: Settings) -> None:
    models = Threaded()
    with working(settings, models) as worker:
        for _ in range(2):
            assert asked(worker, "POST", "/embed", {"texts": TEXTS})[0] == 200
            assert asked(worker, "POST", "/rerank", {"query": "q", "texts": TEXTS})[0] == 200
    assert models.threads[1:] == ["sift-worker_0"] * 4
