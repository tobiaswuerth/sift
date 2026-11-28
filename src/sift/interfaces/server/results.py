import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import NamedTuple

from sift.api.search import SearchResult
from sift.storage.watching import Seen

MOST_RESULTS = 32
MOST_KEPT_HITS = 1000


class Moment(NamedTuple):
    writes: int
    seen: Seen


@dataclass(slots=True, eq=False)
class KeptResults:
    kept: int = MOST_RESULTS
    most_hits: int = MOST_KEPT_HITS
    answers: OrderedDict[str, tuple[SearchResult, Moment]] = field(
        default_factory=OrderedDict[str, tuple[SearchResult, Moment]]
    )
    writes: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def found(self, key: str, now: Moment) -> SearchResult | None:
        with self.lock:
            held = self.answers.get(key)
            if held is None:
                return None
            answer, then = held
            if then != now:
                del self.answers[key]
                return None
            self.answers.move_to_end(key)
            return answer

    def seen(self) -> int:
        with self.lock:
            return self.writes

    def keep(self, key: str, answer: object, before: Moment, after: Moment) -> None:
        if not isinstance(answer, SearchResult) or answer.partial or answer.unsearched:
            return
        if len(answer.hits) > self.most_hits or before != after:
            return
        with self.lock:
            if after.writes != self.writes or after.writes % 2:
                return
            self.answers[key] = (answer, after)
            self.answers.move_to_end(key)
            while len(self.answers) > self.kept:
                self.answers.popitem(last=False)

    def written(self) -> None:
        with self.lock:
            self.writes += 1
            self.answers.clear()
