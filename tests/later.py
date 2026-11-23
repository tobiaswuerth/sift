from sift.app.filing import Readers

LATER_NS = 2**62


def later() -> int:
    return LATER_NS


LATER = Readers(clock=later)
