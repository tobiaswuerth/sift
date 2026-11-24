from dataclasses import dataclass, replace
from types import MappingProxyType

from sift.api.events import Advanced, CollectionStarted, Event, Finished, Noted, Phase, PhaseDone, PhaseStarted
from sift.interfaces.console.shell import command
from sift.interfaces.console.style import Style, Tone, paint
from sift.interfaces.console.text import fitted, printable
from sift.interfaces.console.units import duration, rate

UNITS = MappingProxyType(
    {
        Phase.REMAKE: "chunk",
        Phase.SCAN: "file",
        Phase.READ: "file",
        Phase.MERGE: "chunk",
        Phase.EMBED: "chunk",
        Phase.LOAD: "model",
        Phase.ORDER: "chunk",
    }
)
RATED = frozenset({Phase.REMAKE, Phase.READ, Phase.MERGE, Phase.EMBED})
INDENT = "  "
LABEL = 6
COUNT = 10
UNIT = 8
SPEED = 9
TIME = 10
BAR = 16
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
    unit = UNITS[phase] + ("" if count == 1 else "s")
    return f"{count:>{COUNT},} {unit:<{UNIT}}"


def _row(running: Running, count: int, now: float, style: Style) -> str:
    seconds = now - running.started
    speed = rate(count, seconds) if running.phase in RATED and count and seconds > 0 else ""
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


def _noted(noted: Noted, style: Style) -> tuple[str, ...]:
    fix = (_label("fix", Tone.SECONDARY, style) + printable(noted.remedy),) if noted.remedy else ()
    return (_label("note", Tone.WARNING, style) + printable(noted.text), *fix)


def _header(started: CollectionStarted, style: Style) -> str:
    name, folder = printable(started.name), printable(started.folder)
    return f"{paint(name, Tone.STRONG, style)}  {paint(folder, Tone.SECONDARY, style)}"


def shown(view: View, event: Event, now: float, looks: Looks) -> tuple[View, tuple[str, ...]]:
    match event:
        case CollectionStarted():
            gap = ("",) if view.collections else ()
            return View(event.name, view.collections + 1, now), (*gap, _header(event, looks.style))
        case PhaseStarted():
            return replace(view, running=Running(event.phase, event.total, 0, now)), ()
        case Advanced():
            running = view.running
            return (view if running is None else replace(view, running=replace(running, done=event.done))), ()
        case PhaseDone():
            running = view.running
            lines = () if running is None else (_row(running, event.count, now, looks.style),)
            return replace(view, running=None), lines
        case Noted():
            return view, _noted(event, looks.style)
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
    progress = f"{_bar(within, total, style)} {share:>4.0%}{rate(done, seconds):>{SPEED}}  {left}"
    return fitted(f"{head}{_counted(running.phase, done)}{progress}".rstrip(), columns - 1)
