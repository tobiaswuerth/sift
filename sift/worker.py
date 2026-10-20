"""`sift worker`: lends this machine's GPU to sift on other machines. They send texts and get their embeddings back;
the worker never sees an index or a file."""

import socket
import sys

import anyio

from . import __version__, embed
from .server import _release_gpu_when_idle, serve_app


def app():
    """The worker's HTTP API. GET / names the model and the length of its vectors; POST /embed takes
    {"texts": [...], "query": false} as JSON and returns the vectors as raw fp16 bytes, one row per text."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, PlainTextResponse, Response
    from starlette.routing import Route

    dim = embed.encode_here(["the length of a vector"]).shape[1]  # loads the model
    count = 0

    async def info(request):
        return JSONResponse({"model": embed.EMBED_MODEL, "dim": dim, "version": __version__})

    async def embed_texts(request):
        nonlocal count
        # JSON only: a web page may send text/plain anywhere without asking, so any page could use the GPU
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            return PlainTextResponse("send JSON", status_code=415)
        try:
            body = await request.json()
            texts, query = body["texts"], bool(body.get("query", False))
        except (ValueError, KeyError, TypeError, AttributeError):  # not JSON, or not an object with texts
            texts = None
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            return PlainTextResponse('send {"texts": [...], "query": false}', status_code=400)
        # Async, so it runs on the event loop's thread: one request at a time on the GPU, and PyTorch stays on the
        # main thread, as it must on some platforms.
        vectors = embed.encode_here(texts, query)
        count += len(texts)
        print(f"\rembedded {count:,} texts for {request.client.host}", end="", file=sys.stderr, flush=True)
        return Response(vectors.tobytes(), media_type="application/octet-stream")

    return Starlette(routes=[Route("/", info), Route("/embed", embed_texts, methods=["POST"])])


async def _serve(application, host, port):
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(_release_gpu_when_idle)  # the GPU stays free for others while nobody sends work
        await serve_app(application, host, port, access_log=False)  # the line above shows progress instead
        tasks.cancel_scope.cancel()


def serve(host="0.0.0.0", port=8766):
    print(f"loading {embed.EMBED_MODEL} ...", file=sys.stderr)
    application = app()
    embed.calibrate_batch_size()
    print(f"sift worker on http://{host}:{port}, embedding with {embed.EMBED_MODEL} on {embed.DEVICE}", file=sys.stderr)
    print(f"on other machines, set SIFT_EMBED_REMOTE=http://{socket.gethostname()}:{port}", file=sys.stderr)
    try:
        anyio.run(_serve, application, host, port)
    except BaseExceptionGroup as group:  # from the task group; the error inside is uvicorn's (or Ctrl+C)
        raise group.exceptions[0] from None
