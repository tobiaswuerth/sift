import posixpath
import re
import unicodedata
from collections.abc import Sequence

from sift.api.fields import Fields
from sift.domain.text.links import WIKILINK, aliases, each_of

MENTION_CHARS = 3
UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"})
WORDLIKE = re.compile(r"\w")


def names(path: str, fields: Fields, also: Sequence[str]) -> list[str]:
    own = posixpath.splitext(path.rsplit("/", 1)[-1])[0]
    held = [value for field in ("title", *also) for value in each_of(fields.get(field)) if isinstance(value, str)]
    return [own, *aliases(fields), *held]


def _named(variant: str) -> bool:
    return len(variant) >= MENTION_CHARS and bool(WORDLIKE.search(variant)) and not WIKILINK.fullmatch(variant)


def phrases(found: Sequence[str]) -> tuple[str, ...]:
    written = ((name, unicodedata.normalize("NFC", name).translate(UMLAUTS)) for name in found)
    variants = (" ".join(variant.split()) for pair in written for variant in pair)
    return tuple(dict.fromkeys(variant for variant in variants if _named(variant)))
