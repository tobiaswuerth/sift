"""The client side of `sift worker`: embedding and reranking on another machine's GPU.

It needs no PyTorch, so a machine that searches through a worker starts in a fraction of a second, and it imports
NumPy only when it embeds or reranks, so the CLI starts with the standard library alone. The traffic is
HTTPS, and SIFT_EMBED_REMOTE is the address the worker prints, https://<token>.<fingerprint>@<machine>:<port>: the
fingerprint of the worker's own certificate is checked before anything is sent, and the token admits this machine.
"""

import base64
import hashlib
import http.client
import json
import os
import ssl
import urllib.parse
from functools import cache

REMOTE = os.environ.get("SIFT_EMBED_REMOTE")  # a `sift worker` to embed and rerank on, as it prints its address
PORT = 8766  # the worker's default
BATCH = 256  # texts per request to embed
TIMEOUT = 600  # seconds to wait for an answer; a busy worker can take long, a dead one fails at once


class RemoteError(ConnectionError):
    """The worker failed or can't be reached; the message says how, and never shows the key."""


def fingerprint(der):
    """A certificate's SHA-256, as the key carries it."""
    return base64.urlsafe_b64encode(hashlib.sha256(der).digest()).rstrip(b"=").decode()


def _parts():
    """The worker's host, port, token and certificate fingerprint."""
    url = urllib.parse.urlsplit(REMOTE)
    token, _, pin = (url.username or "").partition(".")
    if url.scheme != "https" or not (url.hostname and token and pin):
        raise RemoteError("SIFT_EMBED_REMOTE must be the https://<key>@<machine>:<port> address `sift worker` prints")
    return url.hostname, PORT if url.port is None else url.port, token, pin


def address():
    """The worker's address without its key, to show."""
    host, port, _, _ = _parts()
    return f"https://{host}:{port}"


def _request(path, payload=None):
    """The body of the worker's answer to a GET of `path`, or to a POST of `payload` as JSON."""
    host, port, token, pin = _parts()
    # The worker's certificate is its own, signed by no authority: the fingerprint in the key vouches for it instead,
    # checked before anything is sent, so a machine posing as the worker gets neither the token nor any text.
    context = ssl.create_default_context()
    context.check_hostname, context.verify_mode = False, ssl.CERT_NONE
    connection = http.client.HTTPSConnection(host, port, timeout=TIMEOUT, context=context)
    try:
        connection.connect()
        if fingerprint(connection.sock.getpeercert(binary_form=True)) != pin:
            raise RemoteError(f"embedding remote {address()} has another certificate than its key: not that worker")
        body = None if payload is None else json.dumps(payload).encode()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        connection.request("GET" if body is None else "POST", path, body, headers)
        response = connection.getresponse()
        data = response.read()
    except RemoteError:
        raise
    except (OSError, http.client.HTTPException) as e:  # refused, timed out, TLS failed, connection dropped
        raise RemoteError(f"embedding remote {address()} is unreachable or failed: {e}") from None
    finally:
        connection.close()
    if response.status == 401:
        raise RemoteError(f"embedding remote {address()} refused the key; copy it again from `sift worker`")
    if response.status != 200:
        raise RemoteError(f"embedding remote {address()} answered {response.status}: {data[:200].decode()}")
    return data


@cache
def _info(url):
    return json.loads(_request("/"))


def model():
    """The worker's embedding model and the length of its vectors, asked once per worker."""
    info = _info(REMOTE)
    return info["model"], info["dim"]


def encode(texts, query=False):
    """Normalised fp16 embeddings from the worker, one row per text."""
    import numpy as np  # here, not at the top: the CLI imports this module and starts with the standard library alone

    dim = model()[1]
    raw = _request("/embed", {"texts": texts, "query": query})
    return np.frombuffer(bytearray(raw), np.float16).reshape(len(texts), dim)  # writable: PyTorch wants that


def rerank(query, texts):
    """The worker's reranking scores of (query, text) pairs."""
    import numpy as np

    return np.array(json.loads(_request("/rerank", {"query": query, "texts": texts}))["scores"], np.float32)
