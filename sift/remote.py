"""The client side of `sift worker`: embedding and reranking on another machine's GPU.

It needs no PyTorch, so a machine that searches through a worker starts in a fraction of a second, and it imports
NumPy only when it embeds or reranks, so the CLI starts with the standard library alone. The traffic is HTTPS, and
SIFT_EMBED_REMOTE is the address the worker prints, https://<token>.<fingerprint>@<machine>:<port>: the fingerprint
of the worker's own certificate is checked before anything is sent, and the token admits this machine.
"""

import base64
import hashlib
import http.client
import json
import socket
import ssl
import threading
import urllib.parse
import weakref
from collections.abc import Iterable, Iterator
from functools import cached_property
from typing import TYPE_CHECKING, Any

from . import __version__, settings

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

PORT = 8766  # the worker's default
BATCH = 256  # texts per request to embed
MOST_TEXTS = 20_000  # texts a worker takes in one request: `query` reranks twice search.MOST_HITS candidates at most
TIMEOUT = 600  # seconds to wait for an answer; a busy worker can take long, a dead one fails at once


class RemoteError(ConnectionError):
    """The worker failed or can't be reached; the message says how, and never shows the key."""


def fingerprint(der: bytes) -> str:
    """A certificate's SHA-256, as the key carries it."""
    return base64.urlsafe_b64encode(hashlib.sha256(der).digest()).rstrip(b"=").decode()


def _series(version: object) -> list[str]:
    """What two sifts must share to work together, by semantic versioning: from 1.0 on the major version, before it
    the minor one too."""
    parts = str(version).split(".")
    return parts[:1] if parts[0] != "0" else parts[:2]


class _Connection(http.client.HTTPSConnection):
    """HTTPS to the worker, kept open between requests. Every connect checks that the worker shows the certificate
    its key names, also when http.client reconnects by itself."""

    def __init__(self, host: str, port: int, pin: str) -> None:
        # The worker's certificate is its own, signed by no authority: the fingerprint in the key vouches for it
        # instead, checked before anything is sent, so a machine posing as the worker gets neither the token nor any
        # text.
        self._tls = ssl.create_default_context()
        self._tls.check_hostname, self._tls.verify_mode = False, ssl.CERT_NONE
        super().__init__(host, port, timeout=TIMEOUT, context=self._tls)
        self._pin = pin

    def connect(self) -> None:
        # The worker listens on IPv4, but a name often resolves to an IPv6 link-local address first. Where a firewall
        # dropped those attempts, every request waited 21 s for one to fail before the system tried IPv4.
        found = socket.getaddrinfo(self.host, self.port, type=socket.SOCK_STREAM)
        # A Python built without IPv6 hands an IPv6 address over as (family, bytes), which it can't connect to
        ips = [(family, ip) for family, _, _, _, (ip, *_) in found if isinstance(ip, str)]
        if not ips:
            raise RemoteError(f"embedding remote {self.host} resolves to no address this Python can connect to")
        ip = min(ips, key=lambda a: a[0] != socket.AF_INET)[1]  # IPv4 if the name has one
        self.sock = self._tls.wrap_socket(
            socket.create_connection((ip, self.port), self.timeout), server_hostname=self.host
        )
        der = self.sock.getpeercert(binary_form=True)  # None where the worker showed no certificate
        if der is None or fingerprint(der) != self._pin:
            self.close()
            address = f"https://{self.host}:{self.port}"
            raise RemoteError(f"embedding remote {address} has another certificate than its key: not that worker")


class Remote:
    """A `sift worker`, used like the models of this machine (see `models.Models`)."""

    batch = BATCH
    main_thread = False  # no PyTorch: a request that hangs holds the call waiting for it, not the server's event loop

    def __init__(self, url: str) -> None:
        parts = urllib.parse.urlsplit(url)
        self._token, _, self._pin = (parts.username or "").partition(".")
        if parts.scheme != "https" or not (parts.hostname and self._token and self._pin):
            raise RemoteError(
                "SIFT_EMBED_REMOTE must be the https://<key>@<machine>:<port> address `sift worker` prints"
            )
        self.host, self.port = parts.hostname, PORT if parts.port is None else parts.port
        self.address = f"https://{self.host}:{self.port}"  # without the key, to show
        self._connection, self._lock = _Connection(self.host, self.port, self._pin), threading.Lock()
        weakref.finalize(self, self._connection.close)  # the kept connection closes with this object

    @cached_property
    def info(self) -> dict[str, Any]:
        """What the worker says about itself: its models, the length of its vectors and its version. Asked once."""
        info = json.loads(self._request("/"))
        if _series(info.get("version")) != _series(__version__):
            raise RemoteError(
                f"embedding remote {self.address} runs sift {info.get('version')}, this machine sift {__version__}: "
                "install the same version on both"
            )
        return info

    @property
    def name(self) -> str:
        """The worker's embedding model at its revision, which it sends beside the name. A worker of sift 1.14 sends
        none, and loads the default model at its latest commit, which holds the weights of the commit sift pins."""
        revision = self.info.get("revision")
        return settings.pinned(self.info["model"] + (f"@{revision}" if revision else ""))

    def load(self) -> None:
        """Reach the worker now, to find out early whether it answers."""
        self.info  # noqa: B018 - asked for its side effect

    def lengths(self, texts: list[str]) -> list[int]:
        """Characters: the tokens are the worker's to count, and asking it would cost a request."""
        return [len(text) for text in texts]

    def prepare(self) -> None:
        """The worker sizes its batches itself."""

    def encode(self, texts: list[str], query: bool = False) -> "NDArray[np.float16]":
        """Normalised fp16 embeddings from the worker, one row per text."""
        import numpy as np  # here, not at the top: the CLI imports this module and starts with the standard library

        raw = self._request("/embed", {"texts": texts, "query": query})
        return np.frombuffer(raw, np.float16).reshape(len(texts), self.info["dim"])

    def encode_batches(self, batches: Iterable[list[str]], query: bool = False) -> Iterator["NDArray[np.float16]"]:
        """The worker's embeddings of each batch, in order, a request each."""
        return (self.encode(texts, query) for texts in batches)

    def rerank(self, query: str, texts: list[str]) -> "NDArray[np.float32]":
        """The worker's scores of (query, text) pairs."""
        import numpy as np

        return np.array(json.loads(self._request("/rerank", {"query": query, "texts": texts}))["scores"], np.float32)

    def release_idle(self) -> bool:
        """The worker hands its GPU back itself."""
        return False

    def _request(self, path: str, payload: dict[str, Any] | None = None) -> bytes:
        """The body of the worker's answer to a GET of `path`, or to a POST of `payload` as JSON, one request at a
        time over the connection kept from the last one."""
        body = None if payload is None else json.dumps(payload).encode()
        headers = {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}
        with self._lock:
            try:
                response, data = self._exchange("GET" if body is None else "POST", path, body, headers)
            except RemoteError:
                raise
            except (OSError, http.client.HTTPException) as e:  # refused, timed out, TLS failed, connection dropped
                self._connection.close()
                raise RemoteError(f"embedding remote {self.address} is unreachable or failed: {e}") from None
        if response.status == 401:
            raise RemoteError(f"embedding remote {self.address} refused the key; copy it again from `sift worker`")
        if response.status != 200:
            shown = data[:200].decode(errors="replace")  # cut inside a character, maybe
            raise RemoteError(f"embedding remote {self.address} answered {response.status}: {shown}")
        return data

    def _exchange(self, *request: Any) -> tuple[http.client.HTTPResponse, bytes]:
        """The worker's response and its body. The worker closes connections that idle for a few seconds, so if the
        one kept from an earlier request turns out closed, the request goes once more, on a new connection."""
        kept = self._connection.sock is not None
        try:
            return self._send(*request)
        except ConnectionError as e:  # the worker had closed it: reset, or no answer at all
            self._connection.close()
            if isinstance(e, RemoteError) or not kept:
                raise
        return self._send(*request)

    def _send(
        self, method: str, path: str, body: bytes | None, headers: dict[str, str]
    ) -> tuple[http.client.HTTPResponse, bytes]:
        self._connection.request(method, path, body, headers)
        response = self._connection.getresponse()
        return response, response.read()
