import asyncio
from collections.abc import Callable
from typing import TYPE_CHECKING

import xberg
from xberg.options import ConversionOptions, SecurityLimits

from sift.domain.text.documents import Conversion, Converted, Unread

if TYPE_CHECKING:
    from sift.api.fields import FieldValue

MOST_COMPRESSION = 10_000
BUILT_IN = "builtins"


def _config() -> xberg.ExtractionConfig:
    return xberg.ExtractionConfig(
        output_format="markdown",
        use_cache=False,
        disable_ocr=True,
        html_options=ConversionOptions(extract_metadata=False),
        security_limits=SecurityLimits(max_compression_ratio=MOST_COMPRESSION),
    )


def extracted(raw: bytes, name: str) -> xberg.ExtractionResult:
    return asyncio.run(xberg.extract(xberg.ExtractInput(kind="bytes", bytes=raw, filename=name), _config()))


type Extract = Callable[[bytes, str], xberg.ExtractionResult]


def converted(raw: bytes, name: str, extract: Extract = extracted) -> Conversion:
    try:
        result = extract(raw, name)
    except RecursionError:
        raise
    except (RuntimeError, xberg.XbergError) as error:
        return Unread(str(error))
    except BaseException as panic:
        if isinstance(panic, Exception) or type(panic).__module__ == BUILT_IN:
            raise
        return Unread(f"the converter failed: {panic}", next_release=True)
    (found,) = result.results
    about = found.metadata
    named: dict[str, FieldValue] = {
        "title": about.title,
        "authors": about.authors,
        "subject": about.subject,
        "keywords": about.keywords,
        "language": about.language,
        "created": about.created_at,
        "modified": about.modified_at,
    }
    return Converted(found.content, {name: value for name, value in named.items() if value})
