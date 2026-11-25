STEP = 1000
LARGER_UNITS = ("kB", "MB", "GB", "TB", "PB")
SHORT = 10
MINUTE = 60
HOUR = 3600
DAY = 24 * HOUR
MILLISECONDS = 1000


def size(stored: int) -> str:
    if stored < STEP:
        return f"{stored} B"
    value, units = stored / STEP, iter(LARGER_UNITS)
    unit = next(units)
    while round(value) >= STEP and (larger := next(units, None)):
        value, unit = value / STEP, larger
    return f"{value:.1f} {unit}" if round(value, 1) < SHORT else f"{value:.0f} {unit}"


def duration(seconds: float) -> str:
    if round(seconds * MILLISECONDS) < MILLISECONDS:
        return f"{round(seconds * MILLISECONDS)} ms"
    if round(seconds, 1) < SHORT:
        return f"{seconds:.1f} s"
    whole = round(seconds)
    if whole < MINUTE:
        return f"{whole} s"
    if whole < HOUR:
        return f"{whole // MINUTE} m {whole % MINUTE} s"
    if whole < DAY:
        return f"{whole // HOUR} h {whole % HOUR // MINUTE} m"
    return f"{whole // DAY} d {whole % DAY // HOUR} h"


def rate(done: int, seconds: float, unit: str) -> str:
    per_second = done / seconds if seconds > 0 else 0.0
    number = f"{per_second:.1f}" if round(per_second, 1) < SHORT else f"{per_second:,.0f}"
    return f"{number} {unit}/s"
