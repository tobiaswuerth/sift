import posixpath
import unicodedata
from collections.abc import Sequence
from types import MappingProxyType

from sift.api.fields import Fields
from sift.domain.text.links import WIKILINK, aliases, each_of
from sift.domain.text.tokens import tokens

MENTION_CHARS = 3
WIDE = frozenset({"W", "F"})
UMLAUTS = MappingProxyType(str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}))


def names(path: str, key: str, title: str, fields: Fields, also: Sequence[str]) -> list[str]:
    own = title if key else posixpath.splitext(path.rsplit("/", 1)[-1])[0]
    held = [value for field in ("title", *also) for value in each_of(fields.get(field)) if isinstance(value, str)]
    return list(dict.fromkeys([own, *aliases(fields), *held]))


def _width(token: str) -> int:
    return sum(2 if unicodedata.east_asian_width(each) in WIDE else 1 for each in token)


def _named(variant: str) -> bool:
    return sum(map(_width, tokens(variant))) >= MENTION_CHARS and not WIKILINK.fullmatch(variant)


def _spaced(name: str) -> str:
    return " ".join(name.split())


def phrases(found: Sequence[str]) -> tuple[str, ...]:
    pairs = ((_spaced(name), _spaced(unicodedata.normalize("NFC", name).translate(UMLAUTS))) for name in found)
    return tuple(dict.fromkeys(variant for written, folded in pairs if _named(folded) for variant in (written, folded)))
