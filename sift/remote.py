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
import ssl
import urllib.parse
from functools import cached_property

PORT = 8766  # the worker's default
BATCH = 256  # texts per request to embed
TIMEOUT = 600  # seconds to wait for an answer; a busy worker can take long, a dead one fails at once


class RemoteError(ConnectionError):
    """The worker failed or can't be reached; the message says how, and never shows the key."""


def fingerprint(der):
    """A certificate's SHA-256, as the key carries it."""
    return base64.urlsafe_b64encode(hashlib.sha256(der).digest()).rstrip(b"=").decode()


class Remote:
    """A `sift worker`, used like the models of this machine (see `models.Models`)."""

    batch = BATCH

    def __init__(self, url):
        parts = urllib.parse.urlsplit(url)
        self._token, _, self._pin = (parts.username or "").partition(".")
        if parts.scheme != "https" or not (parts.hostname and self._token and self._pin):
            raise RemoteError(
                "SIFT_EMBED_REMOTE must be the https://<key>@<machine>:<port> address `sift worker` prints"
            )
        self.host, self.port = parts.hostname, PORT if parts.port is None else parts.port
        self.address = f"https://{self.host}:{self.port}"  # without the key, to show

    @cached_property
    def info(self):
        """What the worker says about itself: its models and the length of its vectors. Asked once."""
        return json.loads(self._request("/"))

    @property
    def name(self):
        return self.info["model"]

    def load(self):
        """Reach the worker now, to find out early whether it answers."""
        self.info  # noqa: B018 - asked for its side effect

    def prepare(self, longest):
        """The worker sizes its batches itself."""

    def encode(self, texts, query=False):
        """Normalised fp16 embeddings from the worker, one row per text."""
        import numpy as np  # here, not at the top: the CLI imports this module and starts with the standard library

        raw = self._request("/embed", {"texts": texts, "query": query})
        return np.frombuffer(bytearray(raw), np.float16).reshape(len(texts), self.info["dim"])  # writable for PyTorch

    def rerank(self, query, texts):
        """The worker's scores of (query, text) pairs."""
        import numpy as np

        return np.array(json.loads(self._request("/rerank", {"query": query, "texts": texts}))["scores"], np.float32)

    def vectors(self, ids, matrix):
        """The vectors stay in RAM, scored with NumPy: the worker only embeds and reranks."""
        from .vectors import InRam

        return InRam(ids, matrix)

    def release_idle(self):
        """The worker hands its GPU back itself."""
        return False

    def _request(self, path, payload=None):
        """The body of the worker's answer to a GET of `path`, or to a POST of `payload` as JSON."""
        # The worker's certificate is its own, signed by no authority: the fingerprint in the key vouches for it
        # instead, checked before anything is sent, so a machine posing as the worker gets neither the token nor any
        # text.
        context = ssl.create_default_context()
        context.check_hostname, context.verify_mode = False, ssl.CERT_NONE
        connection = http.client.HTTPSConnection(self.host, self.port, timeout=TIMEOUT, context=context)
        try:
            connection.connect()
            if fingerprint(connection.sock.getpeercert(binary_form=True)) != self._pin:
                raise RemoteError(
                    f"embedding remote {self.address} has another certificate than its key: not that worker"
                )
            body = None if payload is None else json.dumps(payload).encode()
            headers = {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}
            connection.request("GET" if body is None else "POST", path, body, headers)
            response = connection.getresponse()
            data = response.read()
        except RemoteError:
            raise
        except (OSError, http.client.HTTPException) as e:  # refused, timed out, TLS failed, connection dropped
            raise RemoteError(f"embedding remote {self.address} is unreachable or failed: {e}") from None
        finally:
            connection.close()
        if response.status == 401:
            raise RemoteError(f"embedding remote {self.address} refused the key; copy it again from `sift worker`")
        if response.status != 200:
            raise RemoteError(f"embedding remote {self.address} answered {response.status}: {data[:200].decode()}")
        return data
