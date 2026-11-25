import html
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

VOID = frozenset({"br", "hr", "img", "input", "link", "meta"})


@dataclass(frozen=True, slots=True)
class Html:
    text: str


type Content = Html | str | int | Iterable[Content]
type Attributes = Mapping[str, str | int | bool | None]


def joined(parts: Content) -> Html:
    match parts:
        case Html():
            return parts
        case str():
            return Html(html.escape(parts))
        case int():
            return Html(str(parts))
        case _:
            return Html("".join(joined(part).text for part in parts))


def _attribute(name: str, *, value: str | int | bool | None) -> str:
    if value is None or value is False:
        return ""
    if value is True:
        return f" {name}"
    return f' {name}="{html.escape(str(value))}"'


def tag(name: str, attributes: Attributes | None = None, *children: Content) -> Html:
    given = "".join(_attribute(key, value=value) for key, value in (attributes or {}).items())
    opened = f"<{name}{given}>"
    if name in VOID:
        return Html(opened)
    return Html(f"{opened}{joined(children).text}</{name}>")
