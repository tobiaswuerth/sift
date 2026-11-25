from sift.api.documents import Documents, DocumentText
from sift.interfaces.cli.shown.outcome import Shown, notice_lines, value_text
from sift.interfaces.console.style import Style
from sift.interfaces.console.text import printable


def _text(text: str) -> tuple[str, ...]:
    return (text.removesuffix("\n"),) if text else ()


def _document(found: DocumentText) -> tuple[str, ...]:
    held = tuple(printable(f"{key}: {value_text(value)}") for key, value in (found.fields or {}).items())
    return (*held, "", *_text(found.text)) if held else _text(found.text)


def documents(result: Documents, _platform: str, style: Style) -> Shown:
    if len(result.found) == 1 and not result.not_read:
        return Shown(_document(result.found[0]), ())
    shown = [line for found in result.found for line in (f"==> {printable(found.id)} <==", *_document(found), "")]
    return Shown(tuple(shown[:-1]), notice_lines(result.not_read, style))
