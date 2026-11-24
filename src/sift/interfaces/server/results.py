import threading
from collections import OrderedDict
from dataclasses import dataclass, field

from sift.api.search import SearchResult

RESULTS = 32
KEPT_HITS = 1000


@dataclass(slots=True, eq=False)
class KeptResults:
    kept: int = RESULTS
    most_hits: int = KEPT_HITS
    answers: OrderedDict[str, SearchResult] = field(default_factory=OrderedDict[str, SearchResult])
    writes: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def found(self, key: str) -> SearchResult | None:
        with self.lock:
            answer = self.answers.get(key)
            if answer is not None:
                self.answers.move_to_end(key)
            return answer

    def seen(self) -> int:
        with self.lock:
            return self.writes

    def keep(self, key: str, answer: object, seen: int) -> None:
        if not isinstance(answer, SearchResult) or answer.partial or answer.unsearched:
            return
        if len(answer.hits) > self.most_hits:
            return
        with self.lock:
            if seen != self.writes or seen % 2:
                return
            self.answers[key] = answer
            self.answers.move_to_end(key)
            while len(self.answers) > self.kept:
                self.answers.popitem(last=False)

    def written(self) -> None:
        with self.lock:
            self.writes += 1
            self.answers.clear()
