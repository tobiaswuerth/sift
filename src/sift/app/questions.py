import threading
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field

from sift.domain.arrays import read_only
from sift.domain.codec.codes import Floats, Halves
from sift.inference.models import Models
from sift.inference.prepared import ModelsPrepared

QUESTIONS = 256


@dataclass(slots=True, eq=False)
class RememberedQuestions:
    models: Models
    kept: int = QUESTIONS
    vectors: OrderedDict[str, Halves] = field(default_factory=OrderedDict[str, Halves])
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def name(self) -> str:
        return self.models.name

    def prepare(self) -> ModelsPrepared:
        return self.models.prepare()

    def prepare_ranker(self) -> ModelsPrepared:
        return self.models.prepare_ranker()

    def rerank(self, query: str, texts: Sequence[str]) -> Floats:
        return self.models.rerank(query, texts)

    def encode(self, texts: Sequence[str], *, query: bool) -> Halves:
        if not query or len(texts) != 1:
            return self.models.encode(texts, query=query)
        (text,) = texts
        with self.lock:
            found = self.vectors.get(text)
            if found is not None:
                self.vectors.move_to_end(text)
                return found
        made = read_only(self.models.encode(texts, query=True))
        with self.lock:
            self.vectors[text] = made
            while len(self.vectors) > self.kept:
                self.vectors.popitem(last=False)
        return made
