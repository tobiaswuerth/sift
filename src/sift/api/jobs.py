from dataclasses import dataclass
from typing import Annotated

from sift.api.described import Bounds, Help
from sift.api.requests import Request

MOST_JOB = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class ListJobs(Request):
    stop: Annotated[
        int | None, Help("the job to stop at its next step; what it wrote so far is kept"), Bounds(1, MOST_JOB)
    ] = None


@dataclass(frozen=True, slots=True)
class ServedJob:
    id: int
    command: str
    running: bool
    seconds: float
    watched: bool
    collection: str | None = None
    done: int | None = None


@dataclass(frozen=True, slots=True)
class JobsListed:
    jobs: tuple[ServedJob, ...]
    stopped: int | None = None
