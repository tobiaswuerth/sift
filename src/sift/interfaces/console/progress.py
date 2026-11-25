from dataclasses import dataclass, replace
from types import MappingProxyType

from sift.api.events import (
    Advanced,
    CollectionStarted,
    Event,
    FilesKept,
    Finished,
    Listening,
    Noted,
    Phase,
    PhaseDone,
    PhaseStarted,
)
from sift.interfaces.console.shell import command
from sift.interfaces.console.style import Style, Tone, paint
from sift.interfaces.console.text import fitted, labelled, printable
from sift.interfaces.console.units import duration, rate

UNITS = MappingProxyType(
    {
        Phase.REMAKE: ("chunk", "chunks"),
        Phase.SCAN: ("file", "files"),
        Phase.READ: ("file", "files"),
        Phase.MERGE: ("chunk", "chunks"),
        Phase.EMBED: ("chunk", "chunks"),
        Phase.LOAD: ("model", "models"),
        Phase.ORDER: ("chunk", "chunks"),
        Phase.COPY: ("MB", "MB"),
        Phase.TITLE: ("note", "notes"),
    }
)
RATED = frozenset({Phase.REMAKE, Phase.READ, Phase.MERGE, Phase.EMBED, Phase.COPY, Phase.TITLE})
QUIET_WHEN_NONE = frozenset({Phase.LOAD})
SHORTEST_RATED = 0.05
INDENT = "  "
LABEL = 6
COUNT = 10
UNIT = 8
SPEED = 16
TIME = 10
BAR = 16
STOP_HINT = "Ctrl+C stops it and its jobs; sift jobs lists them"
FILLED = ("\N{BOX DRAWINGS HEAVY HORIZONTAL}", "#")
EMPTY = ("\N{BOX DRAWINGS LIGHT HORIZONTAL}", "-")
DOT = ("\N{MIDDLE DOT}", "|")


@dataclass(frozen=True, slots=True)
class Running:
    phase: Phase
    total: int | None
    done: int
    started: float


@dataclass(frozen=True, slots=True)
class View:
    name: str = ""
    collections: int = 0
    started: float = 0.0
    running: Running | None = None
    kept: int | None = None


@dataclass(frozen=True, slots=True)
class Looks:
    style: Style
    platform: str


def _glyph(pair: tuple[str, str], style: Style) -> str:
    return pair[0] if style.unicode else pair[1]


def _label(label: str, tone: Tone | None, style: Style) -> str:
    padding = " " * (LABEL - len(label))
    return INDENT + (paint(label, tone, style) if tone else label) + padding


def _counted(phase: Phase, count: int) -> str:
    one, many = UNITS[phase]
    unit = one if count == 1 else many
    return f"{count:>{COUNT},} {unit:<{UNIT}}"


def _speed(phase: Phase, count: int, seconds: float) -> str:
    rated = phase in RATED and count > 1 and seconds >= SHORTEST_RATED
    return rate(count, seconds, UNITS[phase][1]) if rated else ""


def _row(running: Running, count: int, now: float, style: Style) -> str:
    seconds = now - running.started
    speed = _speed(running.phase, count, seconds)
    took = paint(f"{duration(seconds):>{TIME}}", Tone.SECONDARY, style)
    return f"{_label(running.phase, None, style)}{_counted(running.phase, count)}{speed:>{SPEED}}{took}"


def _done(finished: Finished, view: View, now: float, looks: Looks) -> tuple[str, ...]:
    style = looks.style
    parts = [f"+{finished.added:,}  ~{finished.changed:,}  -{finished.removed:,}"]
    if finished.unreadable:
        parts.append(paint(f"{finished.unreadable:,} with problems", Tone.WARNING, style))
    parts.append(duration(now - view.started))
    line = _label("done", Tone.DONE, style) + f"  {_glyph(DOT, style)}  ".join(parts)
    if not finished.unreadable:
        return (line,)
    listing = command(("sift", "status", view.name, "--unreadable"), looks.platform)
    return line, " " * (len(INDENT) + LABEL) + paint(listing, Tone.SECONDARY, style)


def _said(said: Noted | Listening, style: Style) -> tuple[str, ...]:
    if isinstance(said, Listening):
        return (
            labelled("serve", f"{printable(said.url)}  {printable(said.home)}", Tone.STRONG, style),
            labelled("stop", STOP_HINT),
        )
    fix = (labelled("fix", said.remedy, Tone.SECONDARY, style),) if said.remedy else ()
    return (labelled("note", said.text, Tone.WARNING, style), *fix)


def _header(started: CollectionStarted, style: Style) -> str:
    name, folder = printable(started.name), printable(started.folder)
    return f"{paint(name, Tone.STRONG, style)}  {paint(folder, Tone.SECONDARY, style)}"


def _progressed(view: View, event: Advanced | FilesKept) -> View:
    if isinstance(event, FilesKept):
        return replace(view, kept=event.files)
    running = view.running
    return view if running is None else replace(view, running=replace(running, done=event.done))


def shown(view: View, event: Event, now: float, looks: Looks) -> tuple[View, tuple[str, ...]]:
    match event:
        case CollectionStarted():
            gap = ("",) if view.collections else ()
            return View(event.name, view.collections + 1, now), (*gap, _header(event, looks.style))
        case PhaseStarted():
            return replace(view, running=Running(event.phase, event.total, 0, now), kept=None), ()
        case Advanced() | FilesKept():
            return _progressed(view, event), ()
        case PhaseDone():
            running = view.running
            quiet = running is None or (running.phase in QUIET_WHEN_NONE and not event.count)
            lines = () if quiet or running is None else (_row(running, event.count, now, looks.style),)
            return replace(view, running=None, kept=None), lines
        case Noted() | Listening():
            return view, _said(event, looks.style)
        case _:
            return replace(view, running=None), _done(event, view, now, looks)


def _bar(done: int, total: int, style: Style) -> str:
    filled = round(BAR * done / total) if total else BAR
    return _glyph(FILLED, style) * filled + _glyph(EMPTY, style) * (BAR - filled)


def running_line(view: View, now: float, columns: int, style: Style) -> str | None:
    running = view.running
    if running is None:
        return None
    seconds, head = now - running.started, f"{INDENT}{running.phase:<{LABEL}}"
    if running.total is None:
        found = _counted(running.phase, running.done) if running.done else " " * (COUNT + 1 + UNIT)
        return fitted(f"{head}{found}{'':>{SPEED}}{duration(seconds):>{TIME}}", columns - 1)
    total, done = running.total, running.done
    within = min(done, total)
    share = within / total if total else 1.0
    per_second = done / seconds if seconds > 0 else 0.0
    left = f"{duration((total - within) / per_second)} left" if per_second and within < total else ""
    progress = f"{_bar(within, total, style)} {share:>4.0%}{_speed(running.phase, done, seconds):>{SPEED}}  {left}"
    return fitted(f"{head}{_counted(running.phase, done)}{progress}".rstrip(), columns - 1)
