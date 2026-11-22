STEP = 1000
LARGER_UNITS = ("kB", "MB", "GB", "TB", "PB")
SHORT = 10


def size(stored: int) -> str:
    if stored < STEP:
        return f"{stored} B"
    value, units = stored / STEP, iter(LARGER_UNITS)
    unit = next(units)
    while round(value) >= STEP and (larger := next(units, None)):
        value, unit = value / STEP, larger
    return f"{value:.1f} {unit}" if round(value, 1) < SHORT else f"{value:.0f} {unit}"
