import threading
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field

from sift.domain.arrays import read_only
from sift.domain.codec.codes import Floats, Halves
from sift.inference.models import Models
from sift.inference.prepared import ModelsPrepared

MOST_QUESTIONS = 256


@dataclass(slots=True, eq=False)
class RememberedQuestions:
    models: Models
    kept: int = MOST_QUESTIONS
    vectors: OrderedDict[tuple[str, str], Halves] = field(default_factory=OrderedDict[tuple[str, str], Halves])
    waiting: dict[str, Future[Halves]] = field(default_factory=dict[str, Future[Halves]])
    lock: threading.Lock = field(default_factory=threading.Lock)
    embedding: threading.Lock = field(default_factory=threading.Lock)

    @property
    def name(self) -> str:
        return self.models.name

    def prepare(self) -> ModelsPrepared:
        return self.models.prepare()

    def prepare_many(self) -> ModelsPrepared:
        return self.models.prepare_many()

    def prepare_ranker(self) -> ModelsPrepared:
        return self.models.prepare_ranker()

    def rerank(self, query: str, texts: Sequence[str]) -> Floats:
        return self.models.rerank(query, texts)

    def encode(self, texts: Sequence[str], *, query: bool) -> Halves:
        if not query or len(texts) != 1:
            return self.models.encode(texts, query=query)
        with self.lock:
            found = self.vectors.get((self.models.name, texts[0]))
            if found is not None:
                self.vectors.move_to_end((self.models.name, texts[0]))
                return found
            asked = self.waiting.setdefault(texts[0], Future())
        with self.embedding:
            if not asked.done():
                self._embedded()
        return asked.result()

    def _embedded(self) -> None:
        with self.lock:
            taken, self.waiting, models = self.waiting, {}, self.models
        try:
            made = read_only(models.encode(list(taken), query=True))
        except BaseException as error:
            for each in taken.values():
                each.set_exception(error)
            raise
        with self.lock:
            for at, (text, each) in enumerate(taken.items()):
                self.vectors[models.name, text] = vector = made[at : at + 1]
                each.set_result(vector)
            while len(self.vectors) > self.kept:
                self.vectors.popitem(last=False)
