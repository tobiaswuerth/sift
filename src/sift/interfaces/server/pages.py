import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, PlainTextResponse, RedirectResponse, Response

from sift.api.collections import Audience
from sift.api.errors import SiftError
from sift.api.wire import Json
from sift.app.documents import source_of
from sift.interfaces.server.app import Sent, Serving, read_for
from sift.interfaces.server.handled import Handled
from sift.interfaces.web.page import page
from sift.interfaces.web.style import STYLE
from sift.interfaces.web.view import PREFIX, view_of

PAGE_POLICY = (
    "default-src 'none'; style-src 'self'; img-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"
)
UNKEPT = MappingProxyType(
    {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "Referrer-Policy": "no-referrer",
        "Cross-Origin-Resource-Policy": "same-origin",
    }
)
PAGE_HEADERS = MappingProxyType({**UNKEPT, "Content-Security-Policy": PAGE_POLICY})
UNFRAMED = "frame-ancestors 'none'"
NO_FILE = "no such file"
PDF = "application/pdf"


@dataclass(frozen=True, slots=True)
class PeopleReads:
    serving: Serving
    handled: Mapping[str, Handled]

    def read(self, name: str, data: Json) -> object:
        return read_for(self.serving, Sent(name, self.handled[name], data, Audience.PEOPLE))


async def page_answered(serving: Serving, handled: Mapping[str, Handled], request: Request) -> Response:
    view = view_of(list(request.query_params.multi_items()))
    built = await asyncio.wrap_future(serving.pool.submit(page, view, PeopleReads(serving, handled)))
    return HTMLResponse(built.text, headers=dict(PAGE_HEADERS))


async def style_answered(_request: Request) -> Response:
    return Response(STYLE, media_type="text/css", headers=dict(UNKEPT))


async def home_answered(_request: Request) -> Response:
    return RedirectResponse(PREFIX)


def _file(serving: Serving, id_: str) -> Path | None:
    try:
        path = source_of(serving.settings.home, id_)
    except SiftError:
        return None
    return path if path.is_file() else None


async def file_answered(serving: Serving, request: Request) -> Response:
    id_ = request.query_params.get("id", "")
    path = await asyncio.wrap_future(serving.pool.submit(_file, serving, id_))
    if path is None:
        return PlainTextResponse(NO_FILE, status_code=404, headers=dict(UNKEPT))
    response = FileResponse(path, filename=path.name, content_disposition_type="inline", headers=dict(UNKEPT))
    response.headers["Content-Security-Policy"] = UNFRAMED if response.media_type == PDF else f"sandbox; {UNFRAMED}"
    return response
