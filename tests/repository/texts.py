import re


def matches(pattern: str, text: str) -> list[str]:
    found: list[str] = re.findall(pattern, text)
    return found
