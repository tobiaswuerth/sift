from collections.abc import Mapping, Sequence

type FieldValue = str | int | float | bool | Sequence[FieldValue] | Mapping[str, FieldValue] | None
type Fields = Mapping[str, FieldValue]


FIELDS = 'front matter to show, comma-separated, e.g. "url,created"; "*" for all'


def listed(text: str) -> tuple[str, ...]:
    return tuple(name.strip() for name in text.split(","))
