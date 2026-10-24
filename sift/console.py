"""What sift tells the user while it works. Everything goes to stderr, so stdout stays for results: messages (`say`),
and one kind of progress line (`Progress`) for every long step, which redraws itself so the time keeps moving even
while a single step takes minutes."""

import gc
import shlex
import subprocess
import sys
import threading
from time import perf_counter

from . import logs

REDRAW_SECONDS = 1  # a progress line redraws itself this often, so a slow step still shows sift at work
LOG_SECONDS = 60  # and goes into the command's log this often, which has no line to redraw
_drawn = None  # the Progress whose line is on screen now, if any
_held = False  # whether progress lines hold the collector off, so that it runs where they are updated
_collections = 0  # made since they did


def say(*parts):
    """A message; while a progress line is drawn, on a line of its own above it, as a Progress's note. Into the
    command's log too, if it keeps one."""
    text = " ".join(map(str, parts))
    logs.said.info(text)
    if _drawn is not None:
        _drawn.note(text)
    else:
        print(text, file=sys.stderr, flush=True)


def count(n, noun):
    """'1 chunk', '2 chunks', '12,345 chunks'."""
    return f"{n:,} {noun}{'' if n == 1 else 's'}"


def duration(seconds):
    """Seconds as H:MM:SS."""
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02}:{seconds % 60:02}"


def command(*words, platform=sys.platform):
    """A command line to suggest, quoted so it can be pasted into the shells there: double quotes for cmd and
    PowerShell on Windows, single quotes elsewhere."""
    return subprocess.list2cmdline(words) if platform == "win32" else shlex.join(words)


def _collect():
    """While progress lines hold the collector off, collect here, on the thread that updates them and does the work,
    as often as the collector would: the young objects when enough were made, every tenth time the older ones too,
    every hundredth all."""
    global _collections
    if _held and gc.get_count()[0] >= gc.get_threshold()[0]:
        _collections += 1
        gc.collect(2 if _collections % 100 == 0 else 1 if _collections % 10 == 0 else 0)


class Line:
    """One line on stderr, rewritten in place; a shorter text blanks out what a longer one left behind."""

    def __init__(self):
        self._width = 0

    def show(self, text):
        self._width = max(self._width, len(text))
        print(f"\r{text:<{self._width}}", end="", file=sys.stderr, flush=True)

    def end(self):
        """Move on to the next line; the next text starts afresh."""
        print(file=sys.stderr, flush=True)
        self._width = 0


class Progress:
    """A progress line, used as a context manager: a bar, `done` of `total` units, the speed, and the time taken and
    left. It draws at once, on every update, and every REDRAW_SECONDS in between; into the command's log every
    LOG_SECONDS, and where it ended."""

    def __init__(self, total, unit, prefix=""):
        self.total, self.unit, self.prefix, self.done, self.start = total, unit, prefix, 0, perf_counter()
        self._logged = 0.0  # seconds in, when it last went into the log
        self._line, self._lock, self._stop = Line(), threading.Lock(), threading.Event()
        self._thread = threading.Thread(target=self._redraw, name="sift-progress", daemon=True)

    def __enter__(self):
        global _drawn, _held, _collections
        self._holds = gc.isenabled()  # the outermost line, unless the collector is off anyway
        if self._holds:
            gc.disable()  # its thread mustn't free PyTorch's tensors, which crashes PyTorch (Python 3.15)
            _held, _collections = True, 0
        self.draw()
        self._thread.start()
        self._outer, _drawn = _drawn, self
        return self

    def __exit__(self, *exc):
        global _drawn, _held
        _drawn = self._outer
        self._stop.set()
        self._thread.join()
        with self._lock:
            self._line.end()
        if self._holds:
            _held = False
            gc.enable()
        logs.said.info(self.text())  # where it ended, once: a log has no line to redraw

    def _redraw(self):
        while not self._stop.wait(REDRAW_SECONDS):
            self.draw()
            if self.elapsed - self._logged >= LOG_SECONDS:
                self._logged = self.elapsed
                logs.said.info(self.text())

    @property
    def elapsed(self):
        """Seconds since the progress began."""
        return perf_counter() - self.start

    def update(self, done):
        self.done = done
        _collect()
        self.draw()

    def note(self, text):
        """A message, on a line of its own where the progress line was; the progress continues below it."""
        with self._lock:
            self._line.show(text)
            self._line.end()
        self.draw()

    def text(self):
        """The progress line as it is now."""
        seconds, share = self.elapsed, self.done / self.total if self.total else 1
        text = f"{self.prefix}[{'#' * round(20 * share):<20}] {share:4.0%} {self.done:,}/{self.total:,} {self.unit}, "
        rate = self.done / seconds if seconds > 0 else 0  # coarse clocks can return the same reading twice
        if rate:
            text += f"{rate:.1f}/s, " if rate < 10 else f"{rate:,.0f}/s, "
        text += f"{duration(seconds)} elapsed"
        if rate:
            text += f", {duration((self.total - self.done) / rate)} left"
        return text

    def draw(self):
        text = self.text()
        with self._lock:
            self._line.show(text)
