from sift.api.links import LinkIn, LinkOut, Links, Unresolved
from sift.interfaces.cli.shown.outcome import Shown, unsearched_lines
from sift.interfaces.console.style import Style
from sift.interfaces.console.text import labelled, printable, table


def _written(link: LinkOut) -> str:
    field = f"{link.field}: " if link.field else ""
    fragment = f"#{link.fragment}" if link.fragment else ""
    return f"{field}{link.target}{fragment}"


def _leads(link: LinkOut) -> str:
    if link.id is not None:
        return link.id + (" (alias)" if link.via == "alias" else "")
    return f"ambiguous: {', '.join(link.ambiguous)}" if link.ambiguous else "unresolved"


def _through(through: str | None) -> str:
    return f"  (through {through})" if through else ""


def _out(heading: str, found: tuple[LinkOut, ...]) -> tuple[str, ...]:
    lines = [f"{each.line:>6}  {_written(each)}  ->  {_leads(each)}{_through(each.through)}" for each in found]
    return (f"{heading}:" if found else f"{heading}: nothing", *lines)


def _in(heading: str, found: tuple[LinkIn, ...]) -> tuple[str, ...]:
    lines = [
        line
        for each in found
        for line in (
            f"  {each.id}  {each.title}{_through(each.through)}",
            *(f"{linking.line:>6}  {linking.text}" for linking in each.lines),
        )
    ]
    return (f"{heading}:" if found else f"{heading}: nothing", *lines)


def links(result: Links, _platform: str, style: Style) -> Shown:
    note = f"{result.title} ({result.id})"
    out = () if result.out is None else _out(f"{note} links to", result.out)
    into = () if result.into is None else _in(f"Linking to {note}", result.into)
    return Shown(tuple(printable(line) for line in (*out, *into)), unsearched_lines(result.unsearched, style))


def unresolved(result: Unresolved, _platform: str, style: Style) -> Shown:
    problems = unsearched_lines(result.unsearched, style)
    if not result.targets:
        nothing = () if result.unsearched else (labelled("nothing", "every link leads to a note"),)
        return Shown((), (*problems, *nothing))
    rows = [
        (f"{each.documents:,}", each.target, ", ".join(each.ids) + (", ..." if each.documents > len(each.ids) else ""))
        for each in result.targets
    ]
    return Shown(table(rows, frozenset({0})), problems)
