STEP = 1000
LARGER_UNITS = ("kB", "MB", "GB", "TB", "PB")
SHORT = 10
MINUTE = 60
HOUR = 3600
DAY = 24 * HOUR
MILLISECONDS = 1000
RATED_BY = ((1, "s"), (MINUTE, "min"), (HOUR, "h"))


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
    slowest = RATED_BY[-1] if per_second else RATED_BY[0]
    span, name = next((each for each in RATED_BY if round(per_second * each[0], 1) >= 1), slowest)
    held = per_second * span
    number = f"{held:.1f}" if round(held, 1) < SHORT else f"{held:,.0f}"
    return f"{number} {unit}/{name}"
