import json
import re
import socket
import threading
from collections.abc import Generator, Sequence
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import cast, override

import numpy as np
import pytest

from sift.api.embedding import CollectionEmbedded, EmbedCollections
from sift.api.errors import ModelUnavailableError, Problem
from sift.api.indexing import Skipped
from sift.api.search import SearchHybrid, SearchVectors
from sift.api.serving import WorkerAddress
from sift.app.embedding import embed
from sift.app.query import query
from sift.app.vsearch import vsearch
from sift.domain.codec.codes import Floats, Halves
from sift.domain.text.embedding import RECIPE
from sift.inference.prepared import ModelsPrepared
from sift.inference.remote import MODEL_HEADER, MOST_TEXTS, Remote, Waits, fingerprint, shows
from sift.interfaces.worker.keys import worker_key
from sift.settings import Settings
from sift.storage.vectors import EMBED_REMEDY
from tests.homes import home_of, quiet
from tests.models import Hashed
from tests.serving import VERSION
from tests.waits import joined, paused, until
from tests.workers import LOOPBACK, QUICK, LiveWorker, canned, nobody_listens, working

TEXTS = ["brush strokes", "pigments"]
ANOTHER = "another"
NOTES = {
    "notes/brush.md": "# Brush strokes\n\nA robot plans brush strokes on canvas.\n",
    "notes/halo.md": "# Halo\n\nDark matter halo models.\n",
}


@pytest.fixture
def lent_settings(tmp_path: Path) -> Settings:
    return Settings(home=tmp_path / "worker", user=None, cwd=tmp_path, state=tmp_path / "worker-state")


@pytest.fixture
def worker(lent_settings: Settings) -> Generator[LiveWorker]:
    with working(lent_settings, Hashed()) as live:
        yield live


def test_a_worker_s_models_are_used_as_this_machine_s_would_be(worker: LiveWorker) -> None:
    remote, here = worker.remote(), Hashed()
    assert (remote.name, remote.prepare(), remote.prepare_ranker(), remote.prepare_many()) == (
        "hashed-words",
        ModelsPrepared((), built=False),
        ModelsPrepared((), built=False),
        ModelsPrepared((), built=False),
    )
    for query_ in (False, True):
        assert remote.encode(TEXTS, query=query_).tolist() == here.encode(TEXTS, query=query_).tolist()
    assert remote.rerank("brush", TEXTS).tolist() == here.rerank("brush", TEXTS).tolist()
    assert (remote.encode([], query=False).shape, remote.rerank("brush", []).shape) == ((0, 48), (0,))
    assert remote.release_idle() is False


def test_more_texts_than_one_request_takes_are_sent_in_several_over_one_connection(worker: LiveWorker) -> None:
    remote, texts, seen = worker.remote(), [f"text {n}" for n in range(MOST_TEXTS + 1)], set[int]()
    remote.prepare()
    seen |= worker.connections()
    vectors = remote.encode(texts, query=False)
    seen |= worker.connections()
    scores = remote.rerank("text", texts)
    seen |= worker.connections()
    assert vectors.tolist() == Hashed().encode(texts, query=False).tolist()
    assert scores.tolist() == Hashed().rerank("text", texts).tolist()
    assert (worker.requests(), len(seen)) == (5, 1)


def test_a_connection_the_worker_closed_is_opened_again(lent_settings: Settings) -> None:
    with working(lent_settings, Hashed(), how=replace(QUICK, keep_alive_seconds=1)) as worker:
        remote = worker.remote()
        remote.encode(["brush strokes"], query=False)
        assert until(worker.closing_all), "the worker kept the idle connection"
        assert remote.encode(["pigments"], query=False).tolist() == Hashed().encode(["pigments"], query=False).tolist()


def test_a_worker_is_reached_over_ipv4_where_its_name_has_an_ipv6_address_first(worker: LiveWorker) -> None:
    def names(host: str, port: int) -> Sequence[tuple[int, object]]:
        assert (host, port) == ("workstation.test", worker.running.port)
        return [(socket.AF_INET6, "::1"), (socket.AF_INET, LOOPBACK)]

    address = replace(worker.address, host="workstation.test")
    assert worker.remote(address=address, resolve=names).encode(TEXTS, query=False).shape == (2, 48)


def test_a_worker_whose_name_has_no_address_this_python_reaches_is_said_so(worker: LiveWorker) -> None:
    def names(_host: str, _port: int) -> Sequence[tuple[int, object]]:
        return [(socket.AF_INET6, 0)]

    remote = worker.remote(address=replace(worker.address, host="workstation.test"), resolve=names)
    said = "embedding remote workstation.test resolves to no address this Python can connect to"
    with pytest.raises(ModelUnavailableError, match=re.escape(said)):
        remote.prepare()


def test_a_worker_is_known_by_its_certificate_before_anything_is_sent(worker: LiveWorker) -> None:
    posing = worker.remote(address=replace(worker.address, pin="A" * 43))
    said = f"embedding remote {worker.address.address} shows another certificate than its key: not that worker"
    with pytest.raises(ModelUnavailableError, match=re.escape(said)):
        posing.encode(TEXTS, query=False)
    assert worker.requests() == 0


def test_no_certificate_and_another_certificate_are_not_the_one_a_key_names() -> None:
    der = b"a certificate"
    assert (shows(der, fingerprint(der)), shows(der, fingerprint(b"another")), shows(None, fingerprint(der))) == (
        True,
        False,
        False,
    )


def test_a_wrong_token_is_refused_and_the_key_never_said(worker: LiveWorker) -> None:
    wrong = worker.remote(address=replace(worker.address, token=ANOTHER))
    with pytest.raises(ModelUnavailableError) as refused:
        wrong.prepare()
    assert (refused.value.message, refused.value.remedy) == (
        f"embedding remote {worker.address.address} refused the key",
        "copy the address the worker prints into SIFT_EMBED_REMOTE again",
    )


def test_a_worker_whose_models_fail_says_so(lent_settings: Settings) -> None:
    class Failing(Hashed):
        @override
        def encode(self, texts: Sequence[str], *, query: bool) -> Halves:
            if texts == ["fails"]:
                raise RuntimeError("CUDA error: an illegal memory access was encountered")
            return super().encode(texts, query=query)

    with working(lent_settings, Failing()) as worker:
        said = f"embedding remote {worker.address.address} answered 500: Internal Server Error"
        with pytest.raises(ModelUnavailableError, match=re.escape(said)):
            worker.remote().encode(["fails"], query=False)


def test_an_unreachable_worker_is_said_so_without_its_key() -> None:
    port = nobody_listens()
    remote = Remote(WorkerAddress(LOOPBACK, port, "s3cret", "p1n"), VERSION)
    with pytest.raises(ModelUnavailableError, match=re.escape(f"embedding remote https://{LOOPBACK}:{port} is")) as e:
        remote.prepare()
    assert ("s3cret" in str(e.value), e.value.remedy) == (
        False,
        "start the worker on that machine and let it through its firewall, or unset SIFT_EMBED_REMOTE",
    )


@pytest.mark.parametrize("here, there", [("2.0.0", "2.4.2"), ("2.4.2", "2.0.0"), ("0.9.1", "0.9.4")])
def test_a_worker_of_the_same_series_of_sift_is_used(lent_settings: Settings, here: str, there: str) -> None:
    with working(lent_settings, Hashed(), version=there) as worker:
        assert worker.remote(here).name == "hashed-words"


@pytest.mark.parametrize("here, there", [("2.0.0", "3.0.0"), ("0.9.1", "0.10.0")])
def test_a_worker_of_another_series_of_sift_is_refused(lent_settings: Settings, here: str, there: str) -> None:
    with working(lent_settings, Hashed(), version=there) as worker:
        remote = worker.remote(here)
        with pytest.raises(ModelUnavailableError, match=re.escape(f"runs sift {there}, this machine sift {here}")):
            remote.prepare()


ABOUT = {"version": VERSION, "model": "an/embedder", "width": 4, "recipe": RECIPE}
EMBEDDER = {MODEL_HEADER: "an%2Fembedder"}


@pytest.mark.parametrize(
    "path, answer, said",
    [
        ("/embed", b"\x00" * 12, "12 bytes for 2 texts of 4 numbers each"),
        ("/rerank", b'{"scores": [0.5]}', "1 score for 2 texts"),
        ("/rerank", b'{"scores": [0.5, 0.1, 0.9]}', "3 scores for 2 texts"),
        ("/rerank", b'{"scores": 0.5}', "no list of scores"),
        ("/rerank", b'{"scores": [0.5, "high"]}', "no list of scores"),
        ("/rerank", b'{"scores": [0.5, true]}', "no list of scores"),
        ("/rerank", b'{"ranked": [1, 0]}', "no list of scores"),
        ("/rerank", b"<html>a proxy's page</html>", "no list of scores"),
        ("/", b"<html>a proxy's page</html>", "nothing a worker says of itself"),
        ("/", b'["sift"]', "nothing a worker says of itself"),
        ("/", json.dumps({**ABOUT, "width": None}).encode(), "no length of its vectors"),
        ("/", json.dumps({**ABOUT, "width": "4"}).encode(), "no length of its vectors"),
        ("/", json.dumps({**ABOUT, "width": True}).encode(), "no length of its vectors"),
        ("/", json.dumps({**ABOUT, "width": 0}).encode(), "no length of its vectors"),
        ("/", json.dumps({**ABOUT, "model": None}).encode(), "no model"),
        ("/", json.dumps({**ABOUT, "model": ""}).encode(), "no model"),
    ],
)
def test_an_answer_that_fits_no_request_is_said_so(tmp_path: Path, path: str, answer: bytes, said: str) -> None:
    answers = {"/": (200, json.dumps(ABOUT).encode()), path: (200, answer)}
    with canned(worker_key(tmp_path), answers, EMBEDDER) as address:
        remote = Remote(address, VERSION)
        ranking = partial(remote.rerank, "q", ["a", "b"])
        asking = ranking if path == "/rerank" else partial(remote.encode, ["a", "b"], query=False)
        told = f"embedding remote {address.address} answered {said}: not what this sift asked it for"
        with pytest.raises(ModelUnavailableError, match=re.escape(told)):
            asking()


def test_a_worker_started_again_with_another_model_is_noticed_at_its_next_vectors(tmp_path: Path) -> None:
    answers = {"/": (200, json.dumps(ABOUT).encode()), "/embed": (200, np.zeros((2, 4), np.float16).tobytes())}
    said = dict(EMBEDDER)
    with canned(worker_key(tmp_path), answers, said) as address:
        remote = Remote(address, VERSION)
        assert remote.encode(["a", "b"], query=False).shape == (2, 4)
        answers["/"] = (200, json.dumps({**ABOUT, "model": "another/embedder"}).encode())
        said[MODEL_HEADER] = "another%2Fembedder"
        with pytest.raises(ModelUnavailableError) as changed:
            remote.encode(["a", "b"], query=False)
        told = f"embedding remote {address.address} embeds with another/embedder now, not with an/embedder as before"
        assert (changed.value.message, changed.value.remedy) == (
            told,
            "ask again: this sift then takes the worker's model as it is now",
        )
        assert remote.name == "another/embedder"
        assert remote.encode(["a", "b"], query=False).shape == (2, 4)
        remote.close()


def test_vectors_a_worker_sends_without_naming_their_model_are_refused(tmp_path: Path) -> None:
    answers = {"/": (200, json.dumps(ABOUT).encode()), "/embed": (200, np.zeros((2, 4), np.float16).tobytes())}
    with canned(worker_key(tmp_path), answers) as address:
        remote = Remote(address, VERSION)
        told = f"embedding remote {address.address} answered vectors without the name of their model"
        with pytest.raises(ModelUnavailableError, match=re.escape(told)):
            remote.encode(["a", "b"], query=False)
        remote.close()


def test_a_model_named_in_letters_no_header_holds_is_known_as_the_worker_s_all_the_same(
    lent_settings: Settings,
) -> None:
    name = "D:/models/\N{LATIN SMALL LETTER E WITH ACUTE}mbedder \N{CJK UNIFIED IDEOGRAPH-6A21}"
    with working(lent_settings, Hashed(name=name)) as worker:
        remote = worker.remote()
        assert (remote.name, remote.encode(TEXTS, query=False).shape) == (name, (2, 48))


def test_a_worker_that_makes_vectors_another_way_is_refused(tmp_path: Path) -> None:
    other = {"/": (200, json.dumps({**ABOUT, "recipe": 99}).encode())}
    with canned(worker_key(tmp_path), other) as address, pytest.raises(ModelUnavailableError) as refused:
        Remote(address, VERSION).prepare()
    told = f"embedding remote {address.address} makes passages into vectors another way (recipe 99, here {RECIPE})"
    assert refused.value.message == told


def test_a_worker_s_error_cut_inside_a_character_is_said_as_far_as_it_reads(tmp_path: Path) -> None:
    cut = {"/": (502, b"x" * 199 + "\N{LATIN SMALL LETTER E WITH ACUTE}".encode())}
    with canned(worker_key(tmp_path), cut) as address, pytest.raises(ModelUnavailableError) as failed:
        Remote(address, VERSION).prepare()
    shown = f"{'x' * 199}\N{REPLACEMENT CHARACTER}"
    assert failed.value.message == f"embedding remote {address.address} answered 502: {shown}"


def test_searches_embed_and_rerank_on_the_worker_as_they_would_here(tmp_path: Path, worker: LiveWorker) -> None:
    home = home_of(tmp_path, NOTES, embedded=worker.remote())
    here, there, asked = Hashed(), worker.remote(), SearchHybrid("brush strokes")
    assert (
        vsearch(home, SearchVectors("brush strokes"), there).hits
        == vsearch(home, SearchVectors("brush strokes"), here).hits
    )
    assert query(home, asked, there).hits == query(home, asked, here).hits
    assert worker.requests() > 2


def test_a_collection_embedded_by_another_model_is_skipped_through_a_worker(
    tmp_path: Path, lent_settings: Settings
) -> None:
    home = home_of(tmp_path, NOTES, embedded=Hashed())
    with working(lent_settings, Hashed(name="other-model")) as worker:
        (skipped,) = embed(home, EmbedCollections(), quiet, worker.remote()).collections
    assert skipped == CollectionEmbedded(
        "notes",
        Skipped(
            Problem.OTHER_MODEL,
            "the store holds vectors of hashed-words, not of other-model",
            EMBED_REMEDY.format(model="other-model"),
        ),
    )


def test_a_slow_worker_s_answer_is_awaited(lent_settings: Settings) -> None:
    class Slow(Hashed):
        @override
        def rerank(self, query: str, texts: Sequence[str]) -> Floats:
            paused(0.5)
            return super().rerank(query, texts)

    with working(lent_settings, Slow()) as worker:
        assert worker.remote().rerank("brush", TEXTS).tolist() == Hashed().rerank("brush", TEXTS).tolist()


def test_vectors_and_scores_from_a_worker_are_this_machine_s_numbers(worker: LiveWorker) -> None:
    vectors = worker.remote().encode(TEXTS, query=False)
    assert (vectors.dtype, vectors.flags.writeable) == (np.dtype(np.float16), True)
    assert worker.remote().rerank("brush", TEXTS).dtype == np.dtype(np.float32)


def refusing_first(_host: str, _port: int) -> Sequence[tuple[int, object]]:
    return [(socket.AF_INET, "127.0.0.2"), (socket.AF_INET, LOOPBACK)]


def test_a_worker_is_reached_at_the_next_address_of_its_name_where_the_first_refuses(worker: LiveWorker) -> None:
    remote = worker.remote(address=replace(worker.address, host="workstation.test"), resolve=refusing_first)
    assert remote.encode(TEXTS, query=False).shape == (2, 48)
    held = remote.line.connection
    assert held is not None
    assert held.sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY) != 0


def test_a_worker_no_address_of_whose_name_answers_is_said_unreachable() -> None:
    nobody = WorkerAddress("workstation.test", nobody_listens(), "token", "pin")
    with pytest.raises(ModelUnavailableError, match="is unreachable or failed"):
        Remote(nobody, VERSION, refusing_first).prepare()


def test_a_worker_that_takes_the_connection_and_says_nothing_is_given_up_on_within_the_connect_wait() -> None:
    failed = list[ModelUnavailableError]()

    def asked(remote: Remote) -> None:
        with pytest.raises(ModelUnavailableError) as refused:
            remote.prepare()
        failed.append(refused.value)

    with socket.create_server((LOOPBACK, 0)) as silent:
        mute = WorkerAddress(LOOPBACK, cast("tuple[str, int]", silent.getsockname())[1], "token", "pin")
        thread = threading.Thread(target=asked, args=(Remote(mute, VERSION, waits=Waits(connect=0.2)),), daemon=True)
        thread.start()
        joined(thread)
    assert [each.message.split(": ")[0] for each in failed] == [
        f"embedding remote {mute.address} is unreachable or failed"
    ]


def test_vectors_of_a_worker_that_are_no_unit_vectors_are_refused(tmp_path: Path) -> None:
    long = np.array([[50, 0, 0, 0], [0, 1, 0, 0]], np.float16).tobytes()
    answers = {"/": (200, json.dumps(ABOUT).encode()), "/embed": (200, long)}
    with canned(worker_key(tmp_path), answers, EMBEDDER) as address:
        remote = Remote(address, VERSION)
        told = f"embedding remote {address.address} answered a vector of length 50, not 1"
        with pytest.raises(ModelUnavailableError, match=re.escape(told)):
            remote.encode(["a", "b"], query=False)
        remote.close()


def test_a_worker_whose_vectors_are_wider_than_sift_codes_is_refused(tmp_path: Path) -> None:
    wide = {"/": (200, json.dumps({**ABOUT, "width": 70_000}).encode())}
    with canned(worker_key(tmp_path), wide) as address, pytest.raises(ModelUnavailableError) as refused:
        Remote(address, VERSION).prepare()
    told = f"embedding remote {address.address} embeds with an/embedder, whose vectors hold 70,000 numbers"
    assert (refused.value.message, refused.value.remedy) == (
        f"{told}: sift codes at most 65,535",
        "lend a model whose vectors hold fewer numbers",
    )
