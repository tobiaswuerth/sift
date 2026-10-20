"""What sift tells the user while it works. Everything goes to stderr, so stdout stays for results: messages (`say`),
and one kind of progress line (`Progress`) for every long step, which redraws itself so the time keeps moving even
while a single step takes minutes."""

import sys
import threading
from time import perf_counter

REDRAW_SECONDS = 1  # a progress line redraws itself this often, so a slow step still shows sift at work


def say(*parts):
    print(*parts, file=sys.stderr, flush=True)


def count(n, noun):
    """'1 chunk', '2 chunks', '12,345 chunks'."""
    return f"{n:,} {noun}{'' if n == 1 else 's'}"


def duration(seconds):
    """Seconds as H:MM:SS."""
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds // 60 % 60:02}:{seconds % 60:02}"


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
    left. It draws at once, on every update, and every REDRAW_SECONDS in between."""

    def __init__(self, total, unit, prefix=""):
        self.total, self.unit, self.prefix, self.done, self.start = total, unit, prefix, 0, perf_counter()
        self._line, self._lock, self._stop = Line(), threading.Lock(), threading.Event()
        self._thread = threading.Thread(target=self._redraw, name="sift-progress", daemon=True)

    def __enter__(self):
        self.draw()
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        with self._lock:
            self._line.end()

    def _redraw(self):
        while not self._stop.wait(REDRAW_SECONDS):
            self.draw()

    @property
    def elapsed(self):
        """Seconds since the progress began."""
        return perf_counter() - self.start

    def update(self, done):
        self.done = done
        self.draw()

    def note(self, text):
        """A message, on a line of its own where the progress line was; the progress continues below it."""
        with self._lock:
            self._line.show(text)
            self._line.end()
        self.draw()

    def draw(self):
        seconds, share = self.elapsed, self.done / self.total if self.total else 1
        text = f"{self.prefix}[{'#' * round(20 * share):<20}] {share:4.0%} {self.done:,}/{self.total:,} {self.unit}, "
        rate = self.done / seconds if seconds > 0 else 0  # coarse clocks can return the same reading twice
        if rate:
            text += f"{rate:.1f}/s, " if rate < 10 else f"{rate:,.0f}/s, "
        text += f"{duration(seconds)} elapsed"
        if rate:
            text += f", {duration((self.total - self.done) / rate)} left"
        with self._lock:
            self._line.show(text)
