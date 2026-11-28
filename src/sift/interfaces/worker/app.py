import asyncio
import hmac
import json
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import cast
from urllib.parse import quote

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route

from sift.domain.arrays import floats
from sift.domain.text.decoding import encodable
from sift.domain.text.json_values import deeper_than
from sift.inference.loading import Resting
from sift.inference.remote import EMBED, INFO, MODEL_HEADER, MOST_TEXTS, RERANK
from sift.interfaces.server.app import TOO_LONG_BODY, UnreadBody, body_of

MOST_BODY = 32 << 20
BODY_SECONDS = 300.0
MOST_NESTING = 4
OCTETS = "application/octet-stream"
WRONG_KEY = "wrong key"
EMBED_SHAPE = 'send {"texts": [...], "query": false}'
RERANK_SHAPE = 'send {"query": "...", "texts": [...]}'
TOO_LONG = f"send at most {MOST_BODY:,} bytes at a time"


@dataclass(frozen=True, slots=True, eq=False)
class Lent:
    token: str
    models: Resting
    said: Mapping[str, object]
    work: ThreadPoolExecutor

    @property
    def model(self) -> str:
        return str(self.said["model"])


def _authorized(lent: Lent, request: Request) -> bool:
    given = (request.headers.get("authorization") or "").encode()
    return hmac.compare_digest(given, f"Bearer {lent.token}".encode())


def _texts(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    listed = cast("list[object]", value)
    texts = [each for each in listed if isinstance(each, str) and encodable(each)]
    return texts if len(texts) == len(listed) else None


async def _asked(lent: Lent, request: Request) -> dict[str, object] | Response:
    if not _authorized(lent, request):
        return PlainTextResponse(WRONG_KEY, status_code=401)
    body = await body_of(request, seconds=BODY_SECONDS, most=MOST_BODY)
    if isinstance(body, UnreadBody):
        return PlainTextResponse(TOO_LONG if body is TOO_LONG_BODY else body.error.message, status_code=body.status)
    try:
        text = body.decode()
        read = None if deeper_than(text, MOST_NESTING) else cast("object", json.loads(text))
    except ValueError:
        read = None
    return cast("dict[str, object]", read) if isinstance(read, dict) else {}


def _too_many(texts: list[str]) -> Response | None:
    if len(texts) <= MOST_TEXTS:
        return None
    return PlainTextResponse(f"send at most {MOST_TEXTS:,} texts at a time, not {len(texts):,}", status_code=413)


async def _worked[T](lent: Lent, run: Callable[[], T]) -> T:
    return await asyncio.wrap_future(lent.work.submit(run))


async def info_answered(lent: Lent, request: Request) -> Response:
    if not _authorized(lent, request):
        return PlainTextResponse(WRONG_KEY, status_code=401)
    return JSONResponse(dict(lent.said))


async def embed_answered(lent: Lent, request: Request) -> Response:
    asked = await _asked(lent, request)
    if isinstance(asked, Response):
        return asked
    texts, query = _texts(asked.get("texts")), asked.get("query", False)
    if texts is None or not isinstance(query, bool):
        return PlainTextResponse(EMBED_SHAPE, status_code=400)
    refused = _too_many(texts)
    if refused is not None:
        return refused
    vectors = await _worked(lent, partial(lent.models.encode, texts, query=query))
    return Response(vectors.tobytes(), media_type=OCTETS, headers={MODEL_HEADER: quote(lent.model, safe="")})


async def rerank_answered(lent: Lent, request: Request) -> Response:
    asked = await _asked(lent, request)
    if isinstance(asked, Response):
        return asked
    query, texts = asked.get("query"), _texts(asked.get("texts"))
    if texts is None or not isinstance(query, str) or not encodable(query):
        return PlainTextResponse(RERANK_SHAPE, status_code=400)
    refused = _too_many(texts)
    if refused is not None:
        return refused
    scores = await _worked(lent, partial(lent.models.rerank, query, texts))
    return JSONResponse({"scores": floats(scores)})


def worker_app(lent: Lent) -> Starlette:
    info: Callable[[Request], Awaitable[Response]] = partial(info_answered, lent)
    embed: Callable[[Request], Awaitable[Response]] = partial(embed_answered, lent)
    rerank: Callable[[Request], Awaitable[Response]] = partial(rerank_answered, lent)
    return Starlette(
        routes=[Route(INFO, info), Route(EMBED, embed, methods=["POST"]), Route(RERANK, rerank, methods=["POST"])]
    )
