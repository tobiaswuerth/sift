from enum import Enum

REMAKING = "remaking "


class KeywordState(Enum):
    CURRENT = "current"
    REMAKING = "remaking"
    TAKEN_OVER = "taken over"
    STALE = "stale"


def remaking(made: str) -> str:
    return REMAKING + made


def keyword_state(found: str | None, made: str) -> KeywordState:
    if found == made:
        return KeywordState.CURRENT
    if found == remaking(made):
        return KeywordState.REMAKING
    if found is not None and found.startswith(REMAKING):
        return KeywordState.TAKEN_OVER
    return KeywordState.STALE
