"""`sift worker`: lends this machine's GPU to sift on other machines. They send texts and get their embeddings and
reranking scores back; the worker never sees an index or a file.

The traffic is HTTPS with the worker's own certificate, and requests need the worker's token. Both are made on the
first start and kept in SIFT_HOME/worker; the key the worker prints for other machines carries the token and the
certificate's fingerprint, so they talk to this worker only and it answers them only."""

import datetime
import functools
import hmac
import os
import secrets
import socket
import ssl
import sys
from collections import namedtuple

import anyio

from . import __version__, embed, remote, store
from .server import _release_gpu_when_idle, serve_app

Credentials = namedtuple("Credentials", "token pin cert private_key")
Credentials.key = property(lambda self: f"{self.token}.{self.pin}")  # what other machines put before @


def credentials():
    """The worker's token, certificate and private key, made on its first start and kept, so the key other machines
    were given stays valid. Deleting SIFT_HOME/worker makes a new one."""
    folder = store.HOME / "worker"
    token, cert, private_key = folder / "token", folder / "cert.pem", folder / "key.pem"
    if not (token.exists() and cert.exists() and private_key.exists()):
        folder.mkdir(parents=True, exist_ok=True)
        _make_certificate(cert, private_key)
        _write(token, secrets.token_urlsafe(32).encode())
    pin = remote.fingerprint(ssl.PEM_cert_to_DER_cert(cert.read_text(encoding="ascii")))
    return Credentials(token.read_text(encoding="ascii").strip(), pin, cert, private_key)


def _make_certificate(cert, private_key):
    """A certificate of the worker's own, EC P-256: the fingerprint in the key vouches for it, not an authority."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import NameOID

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "sift worker")])
    now = datetime.datetime.now(datetime.UTC)
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=100 * 365))  # as long as the key handed out
        .sign(key, hashes.SHA256())
    )
    pem, pkcs8 = serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8
    _write(private_key, key.private_bytes(pem, pkcs8, serialization.NoEncryption()))
    _write(cert, certificate.public_bytes(pem))


def _write(path, data):
    """Write a file only this user may read (on POSIX; on Windows the home folder's permissions do that)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_BINARY", 0), 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


async def _json(request):
    """The request's JSON object; {} for anything else."""
    try:
        body = await request.json()
    except ValueError:  # not JSON, or not UTF-8
        return {}
    return body if isinstance(body, dict) else {}


def _texts(value):
    """Whether a request field is a list of texts."""
    return isinstance(value, list) and all(isinstance(v, str) for v in value)


def app(token):
    """The worker's HTTP API, for requests that carry `Authorization: Bearer <token>`.

    GET / names the models and the length of the vectors. POST /embed takes {"texts": [...], "query": false} and
    returns the vectors as raw fp16 bytes, one row per text. POST /rerank takes {"query": "...", "texts": [...]} and
    returns {"scores": [...]}. Handlers are async, so they run on the event loop's thread: one request at a time on
    the GPU, and PyTorch stays on the main thread, as it must on some platforms.
    """
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, PlainTextResponse, Response
    from starlette.routing import Route

    dim = embed.encode_here(["the length of a vector"]).shape[1]  # loads the model
    expected, done = f"Bearer {token}".encode(), {"texts": 0, "queries": 0}
    status = "\r{texts:,} texts embedded, {queries:,} queries reranked (last for {host})"

    def authorized(handler):
        """The handler, for requests with the token only; afterwards the status line shows what was done."""

        @functools.wraps(handler)
        async def endpoint(request):
            if not hmac.compare_digest(request.headers.get("authorization", "").encode(), expected):
                return PlainTextResponse("wrong key", status_code=401)
            response = await handler(request)
            print(status.format(**done, host=request.client.host), end="", file=sys.stderr, flush=True)
            return response

        return endpoint

    @authorized
    async def info(request):
        return JSONResponse(
            {"model": embed.EMBED_MODEL, "dim": dim, "reranker": embed.RERANK_MODEL, "version": __version__}
        )

    @authorized
    async def embed_texts(request):
        body = await _json(request)
        if not _texts(texts := body.get("texts")):
            return PlainTextResponse('send {"texts": [...], "query": false}', status_code=400)
        vectors = embed.encode_here(texts, bool(body.get("query")))
        done["texts"] += len(texts)
        return Response(vectors.tobytes(), media_type="application/octet-stream")

    @authorized
    async def rerank(request):
        body = await _json(request)
        if not (isinstance(query := body.get("query"), str) and _texts(texts := body.get("texts"))):
            return PlainTextResponse('send {"query": "...", "texts": [...]}', status_code=400)
        scores = embed.rerank_here(query, texts)
        done["queries"] += 1
        return JSONResponse({"scores": [float(s) for s in scores]})

    routes = [
        Route("/", info),
        Route("/embed", embed_texts, methods=["POST"]),
        Route("/rerank", rerank, methods=["POST"]),
    ]
    return Starlette(routes=routes)


async def _serve(application, host, port, found):
    async with anyio.create_task_group() as tasks:
        tasks.start_soon(_release_gpu_when_idle)  # the GPU stays free for others while nobody sends work
        tls = {"ssl_certfile": str(found.cert), "ssl_keyfile": str(found.private_key)}
        await serve_app(application, host, port, access_log=False, **tls)  # the count line shows progress instead
        tasks.cancel_scope.cancel()


def serve(host="0.0.0.0", port=8766):
    found = credentials()
    print(f"loading {embed.EMBED_MODEL} and {embed.RERANK_MODEL} ...", file=sys.stderr)
    application = app(found.token)
    embed.reranker()  # now, so the first query is quick
    embed.calibrate_batch_size()
    say = functools.partial(print, file=sys.stderr)
    say(f"sift worker on https://{host}:{port}, embedding and reranking on {embed.DEVICE}")
    say(f"on other machines, set SIFT_EMBED_REMOTE=https://{found.key}@{socket.gethostname()}:{port}")
    say("keep that key private: it lets a machine use this worker")
    try:
        anyio.run(_serve, application, host, port, found)
    except BaseExceptionGroup as group:  # from the task group; the error inside is uvicorn's (or Ctrl+C)
        raise group.exceptions[0] from None
