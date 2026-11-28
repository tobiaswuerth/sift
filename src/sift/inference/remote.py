import base64
import hashlib
import http.client
import json
import socket
import ssl
import threading
import weakref
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import cast, final, override
from urllib.parse import quote, unquote

import numpy as np

from sift.api.errors import ModelUnavailableError
from sift.api.package import PACKAGE
from sift.api.serving import EMBED_REMOTE_VARIABLE, WorkerAddress
from sift.domain.codec.codes import MOST_BITS, Floats, Halves
from sift.domain.text.embedding import RECIPE
from sift.domain.text.wording import counted
from sift.inference.prepared import ModelsPrepared

MOST_TEXTS = 256
CONNECT_SECONDS = 10.0
ANSWER_SECONDS = 600.0
INFO = "/"
EMBED = "/embed"
RERANK = "/rerank"
JSON = "application/json"
SHOWN_BYTES = 200
HALF_BYTES = 2
UNAUTHORIZED = 401
ANSWERED = 200
SAME_SIFT = "install the same sift on both, and reach the worker directly"
REACH_REMEDY = f"start the worker on that machine and let it through its firewall, or unset {EMBED_REMOTE_VARIABLE}"
KEY_REMEDY = f"copy the address the worker prints into {EMBED_REMOTE_VARIABLE} again"
FAILED_REMEDY = "see the worker's log for what failed"
AGAIN_REMEDY = "ask again: this sift then takes the worker's model as it is now"
MODEL_HEADER = "Sift-Model"
NOTHING_SAID = "nothing a worker says of itself"
CLOSED_BY_THE_WORKER = (ConnectionError, ssl.SSLEOFError)
UNIT_SLACK = 0.01
WIDE_REMEDY = "lend a model whose vectors hold fewer numbers"

type Resolve = Callable[[str, int], Sequence[tuple[int, object]]]


@dataclass(frozen=True, slots=True)
class Waits:
    connect: float = CONNECT_SECONDS
    answer: float = ANSWER_SECONDS


WAITS = Waits()


def addresses(host: str, port: int) -> list[tuple[int, object]]:
    return [(family, found[0]) for family, _, _, _, found in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)]


def fingerprint(der: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(der).digest()).rstrip(b"=").decode()


def shows(der: bytes | None, pin: str) -> bool:
    return der is not None and fingerprint(der) == pin


def series(release: str) -> tuple[str, ...]:
    parts = tuple(release.split("."))
    return parts[:1] if parts[0] != "0" else parts[:2]


def unverified_tls() -> ssl.SSLContext:
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    tls.minimum_version = ssl.TLSVersion.TLSv1_2
    tls.check_hostname = False
    tls.verify_mode = ssl.CERT_NONE
    return tls


def _connected(ips: Sequence[str], port: int, seconds: float) -> socket.socket:
    *others, last = ips
    for ip in others:
        try:
            return socket.create_connection((ip, port), seconds)
        except OSError:
            continue
    return socket.create_connection((last, port), seconds)


@final
class PinnedConnection(http.client.HTTPSConnection):
    def __init__(self, worker: WorkerAddress, resolve: Resolve, waits: Waits = WAITS) -> None:
        self.tls: ssl.SSLContext = unverified_tls()
        super().__init__(worker.host, worker.port, timeout=waits.answer, context=self.tls)
        self.worker: WorkerAddress = worker
        self.resolve: Resolve = resolve
        self.waits: Waits = waits

    @override
    def connect(self) -> None:
        reachable = [(family, ip) for family, ip in self.resolve(self.host, self.port) if isinstance(ip, str)]
        if not reachable:
            said = f"embedding remote {self.worker.host} resolves to no address this Python can connect to"
            raise ModelUnavailableError(said, REACH_REMEDY)
        ordered = sorted(reachable, key=lambda each: each[0] != socket.AF_INET)
        raw = _connected([ip for _, ip in ordered], self.port, self.waits.connect)
        raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        held = self.tls.wrap_socket(raw, server_hostname=self.host)
        held.settimeout(self.waits.answer)
        if not shows(held.getpeercert(binary_form=True), self.worker.pin):
            held.close()
            said = f"embedding remote {self.worker.address} shows another certificate than its key: not that worker"
            raise ModelUnavailableError(said, KEY_REMEDY)
        self.sock = held


@dataclass(frozen=True, slots=True)
class WorkerSays:
    model: str
    width: int


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _whole(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _read(raw: bytes) -> object:
    try:
        return cast("object", json.loads(raw))
    except ValueError:
        return None


def _scores(raw: bytes) -> list[float] | None:
    read = _read(raw)
    found = cast("dict[str, object]", read).get("scores") if isinstance(read, dict) else None
    if not isinstance(found, list):
        return None
    scores = cast("list[object]", found)
    numbers = [each for each in scores if isinstance(each, int | float) and not isinstance(each, bool)]
    return [float(each) for each in numbers] if len(numbers) == len(scores) else None


def _reason(error: BaseException) -> str:
    return (error.strerror if isinstance(error, OSError) else None) or str(error) or type(error).__name__


class OtherModelError(ModelUnavailableError):
    pass


def _unfit(
    worker: WorkerAddress, answered: str, kind: type[ModelUnavailableError] = ModelUnavailableError
) -> ModelUnavailableError:
    told = f"embedding remote {worker.address} answered {answered}: not what this sift asked it for"
    return kind(told, SAME_SIFT)


def _same_sift(worker: WorkerAddress, release: str, said: dict[str, object]) -> None:
    theirs = said.get("version")
    if not isinstance(theirs, str) or series(theirs) != series(release):
        told = f"embedding remote {worker.address} runs sift {theirs}, this machine sift {release}"
        raise ModelUnavailableError(told, SAME_SIFT)
    if said.get("recipe") != RECIPE:
        told = f"embedding remote {worker.address} makes passages into vectors another way"
        raise ModelUnavailableError(f"{told} (recipe {said.get('recipe')}, here {RECIPE})", SAME_SIFT)


def _checked(worker: WorkerAddress, release: str, raw: bytes) -> WorkerSays:
    read = _read(raw)
    if not isinstance(read, dict):
        raise _unfit(worker, NOTHING_SAID)
    said = cast("dict[str, object]", read)
    _same_sift(worker, release, said)
    model, width = _text(said.get("model")), _whole(said.get("width"))
    if model is None:
        raise _unfit(worker, "no model")
    if width is None:
        raise _unfit(worker, "no length of its vectors")
    if width > MOST_BITS:
        told = f"embedding remote {worker.address} embeds with {model}, whose vectors hold {width:,} numbers"
        raise ModelUnavailableError(f"{told}: sift codes at most {MOST_BITS:,}", WIDE_REMEDY)
    return WorkerSays(model, width)


def _vectors_in(worker: WorkerAddress, raw: bytes, texts: int, width: int) -> Halves:
    if len(raw) != texts * width * HALF_BYTES:
        raise _unfit(worker, f"{len(raw):,} bytes for {counted(texts, 'text')} of {width:,} numbers each")
    vectors = np.frombuffer(raw, np.float16).reshape(texts, width)
    lengths = np.linalg.norm(vectors.astype(np.float32), axis=1)
    off = lengths[np.isfinite(lengths) & (lengths > 0) & (np.abs(lengths - 1) > UNIT_SLACK)]
    if len(off):
        raise _unfit(worker, f"a vector of length {off[0]:.3g}, not 1")
    return vectors


def _scores_in(worker: WorkerAddress, raw: bytes, texts: int) -> Floats:
    scores = _scores(raw)
    if scores is None:
        raise _unfit(worker, "no list of scores")
    if len(scores) != texts:
        raise _unfit(worker, f"{counted(len(scores), 'score')} for {counted(texts, 'text')}")
    return np.array(scores, np.float32)


@dataclass(frozen=True, slots=True)
class WorkerAnswer:
    status: int
    data: bytes
    model: str | None


def _sent(
    connection: PinnedConnection, method: str, path: str, body: bytes | None, headers: dict[str, str]
) -> WorkerAnswer:
    connection.request(method, path, body, headers)
    response = connection.getresponse()
    return WorkerAnswer(response.status, response.read(), response.getheader(MODEL_HEADER))


def _other_model(worker: WorkerAddress, before: str, now: str | None) -> ModelUnavailableError:
    if now is None:
        return _unfit(worker, "vectors without the name of their model", OtherModelError)
    told = f"embedding remote {worker.address} embeds with {unquote(now)} now, not with {before} as before"
    return OtherModelError(told, AGAIN_REMEDY)


def _unreachable(worker: WorkerAddress, error: BaseException) -> ModelUnavailableError:
    said = f"embedding remote {worker.address} is unreachable or failed: {_reason(error)}"
    return ModelUnavailableError(said, REACH_REMEDY)


@dataclass(slots=True, eq=False, weakref_slot=True)
class WorkerLine:
    worker: WorkerAddress
    resolve: Resolve = addresses
    waits: Waits = WAITS
    lock: threading.Lock = field(default_factory=threading.Lock)
    connection: PinnedConnection | None = None

    def asked(self, path: str, payload: object) -> WorkerAnswer:
        body = None if payload is None else json.dumps(payload).encode()
        headers = {"Authorization": f"Bearer {self.worker.token}", "Content-Type": JSON}
        with self.lock:
            answer = self._exchanged("GET" if body is None else "POST", path, body, headers)
        if answer.status == UNAUTHORIZED:
            raise ModelUnavailableError(f"embedding remote {self.worker.address} refused the key", KEY_REMEDY)
        if answer.status != ANSWERED:
            shown = answer.data[:SHOWN_BYTES].decode(errors="replace")
            told = f"embedding remote {self.worker.address} answered {answer.status}: {shown}"
            raise ModelUnavailableError(told, FAILED_REMEDY)
        return answer

    def close(self) -> None:
        with self.lock:
            if self.connection is not None:
                self.connection.close()

    def _exchanged(self, method: str, path: str, body: bytes | None, headers: dict[str, str]) -> WorkerAnswer:
        connection = self.connection or self._connected()
        again = connection.sock is not None
        while True:
            try:
                return _sent(connection, method, path, body, headers)
            except ModelUnavailableError:
                connection.close()
                raise
            except (OSError, http.client.HTTPException) as error:
                connection.close()
                if not (again and isinstance(error, CLOSED_BY_THE_WORKER)):
                    raise _unreachable(self.worker, error) from None
                again = False

    def _connected(self) -> PinnedConnection:
        connection = PinnedConnection(self.worker, self.resolve, self.waits)
        weakref.finalize(self, connection.close)
        self.connection = connection
        return connection


def _embedded(line: WorkerLine, said: WorkerSays, part: Sequence[str], *, query: bool) -> Halves:
    answer = line.asked(EMBED, {"texts": list(part), "query": query})
    if answer.model != quote(said.model, safe=""):
        raise _other_model(line.worker, said.model, answer.model)
    return _vectors_in(line.worker, answer.data, len(part), said.width)


@dataclass(slots=True, eq=False)
class Remote:
    worker: WorkerAddress
    release: str
    resolve: Resolve = addresses
    said: WorkerSays | None = None
    waits: Waits = WAITS
    line: WorkerLine = field(init=False)

    def __post_init__(self) -> None:
        self.line = WorkerLine(self.worker, self.resolve, self.waits)

    @property
    def name(self) -> str:
        return self._said().model

    def prepare(self) -> ModelsPrepared:
        self._said()
        return ModelsPrepared((), built=False)

    def prepare_many(self) -> ModelsPrepared:
        return self.prepare()

    def prepare_ranker(self) -> ModelsPrepared:
        return self.prepare()

    def release_idle(self) -> bool:
        return False

    def close(self) -> None:
        self.line.close()

    def encode(self, texts: Sequence[str], *, query: bool) -> Halves:
        said = self._said()
        try:
            parts = [
                _embedded(self.line, said, part, query=query)
                for part in (texts[at : at + MOST_TEXTS] for at in range(0, len(texts), MOST_TEXTS))
            ]
        except OtherModelError:
            self.said = None
            raise
        return np.concatenate(parts) if parts else np.empty((0, said.width), np.float16)

    def rerank(self, query: str, texts: Sequence[str]) -> Floats:
        parts = [
            _scores_in(self.worker, self.line.asked(RERANK, {"query": query, "texts": list(part)}).data, len(part))
            for part in (texts[at : at + MOST_TEXTS] for at in range(0, len(texts), MOST_TEXTS))
        ]
        return np.concatenate(parts) if parts else np.empty(0, np.float32)

    def _said(self) -> WorkerSays:
        if self.said is None:
            self.said = _checked(self.worker, self.release, self.line.asked(INFO, None).data)
        return self.said


def remote(worker: WorkerAddress) -> Remote:
    return Remote(worker, version(PACKAGE))
