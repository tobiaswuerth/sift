from sift.api.jobs import JobsListed, ServedJob
from sift.interfaces.cli.shown.outcome import Shown
from sift.interfaces.console.style import Style, Tone
from sift.interfaces.console.text import labelled, table

HEADER = ("job", "command", "state", "seconds", "collection", "done", "watched")
NONE = "-"
NO_JOBS = "no job runs or waits in the background"


def _row(job: ServedJob) -> tuple[str, ...]:
    done = NONE if job.done is None else f"{job.done:,}"
    state, watched = "runs" if job.running else "waits", "yes" if job.watched else "no"
    return (str(job.id), job.command, state, f"{job.seconds:,.0f}", job.collection or NONE, done, watched)


def listed(result: JobsListed, _platform: str, style: Style) -> Shown:
    said = f"job {result.stopped} stops at its next step; what it wrote so far is kept"
    stopped = () if result.stopped is None else (labelled("stop", said, Tone.DONE, style),)
    if not result.jobs:
        return Shown((), (*stopped, labelled("none", NO_JOBS)))
    return Shown(table([HEADER, *map(_row, result.jobs)], frozenset({0, 3, 5})), stopped)
