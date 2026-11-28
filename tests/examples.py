from hypothesis import settings

PLAIN = "sift"
EXPLORE = "explore"
EXPLORED = 2000
FEWEST_EXPLORED = 200


def _exploring() -> bool:
    current = settings.default
    return current is not None and current.max_examples >= EXPLORED


def few(n: int) -> settings:
    return settings(max_examples=max(n, FEWEST_EXPLORED) if _exploring() else n)


def more(n: int) -> settings:
    return settings(max_examples=max(n, EXPLORED) if _exploring() else n)
