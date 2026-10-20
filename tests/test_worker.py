"""`sift worker` lends a machine's GPU to other machines' sift: they send texts over HTTPS, with the worker's key, and
get vectors back."""

import json
import os
import re
import threading
import time
import types

import numpy as np
import pytest

from sift import __version__, cli, store, worker
from sift.search import Index

from .conftest import FakeSentenceTransformer, fake_vector, nobody_listens

TOKEN = "s3cret-token"


@pytest.fixture
def client(fake_models):
    from starlette.testclient import TestClient

    with TestClient(worker.app(TOKEN), headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        yield c


def post(client, payload):
    return client.post("/embed", content=json.dumps(payload), headers={"Content-Type": "application/json"})


def test_the_worker_says_which_model_it_embeds_with(client, fake_models):
    assert client.get("/").json() == {"model": fake_models.EMBED_MODEL, "dim": 64, "version": __version__}


def test_the_worker_embeds_texts_as_this_machine_would(client, fake_models, capsys):
    response = post(client, {"texts": ["brush strokes", "pigments"]})
    assert response.status_code == 200
    vectors = np.frombuffer(response.content, np.float16).reshape(2, 64)
    assert vectors.tolist() == fake_models.encode_here(["brush strokes", "pigments"]).tolist()
    assert "embedded 2 texts for testclient" in capsys.readouterr().err  # it shows that it works


def test_the_worker_embeds_queries_as_queries(client, monkeypatch):
    seen = []
    real = FakeSentenceTransformer.encode_query
    monkeypatch.setattr(
        FakeSentenceTransformer, "encode_query", lambda self, t, **kw: seen.extend(t) or real(self, t, **kw)
    )
    post(client, {"texts": ["how do robots paint"], "query": True})
    assert seen == ["how do robots paint"]


@pytest.mark.parametrize("content", ['{"texts": "not a list"}', '{"texts": [1, 2]}', "[]", "not json"])
def test_the_worker_refuses_anything_but_lists_of_texts(client, content):
    response = client.post("/embed", content=content, headers={"Content-Type": "application/json"})
    assert response.status_code == 400


@pytest.mark.parametrize("authorization", [None, "Bearer wrong", f"Basic {TOKEN}", TOKEN])
def test_the_worker_answers_only_with_its_token(fake_models, authorization):
    from starlette.testclient import TestClient

    headers = {"Authorization": authorization} if authorization else {}
    with TestClient(worker.app(TOKEN)) as c:
        assert c.get("/", headers=headers).status_code == 401
        assert c.post("/embed", json={"texts": ["x"]}, headers=headers).status_code == 401


# --- the worker's key --------------------------------------------------------


def test_the_worker_makes_its_key_once_and_keeps_it(home):
    # Other machines keep the key they were given, so the worker must not make a new one on every start.
    first = worker.credentials()
    assert first == worker.credentials()
    assert first.key == f"{first.token}.{first.pin}"
    assert len(first.token) >= 40  # 32 random bytes
    assert first.cert.parent == first.private_key.parent == home / "worker"


def test_another_home_makes_another_key(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "HOME", tmp_path / "a")
    a = worker.credentials()
    monkeypatch.setattr(store, "HOME", tmp_path / "b")
    b = worker.credentials()
    assert a.token != b.token
    assert a.pin != b.pin


@pytest.mark.skipif(os.name == "nt", reason="Windows guards files with ACLs; a user's home is private by default")
def test_the_key_files_are_private(home):
    found = worker.credentials()
    for path in (home / "worker" / "token", found.private_key):
        assert path.stat().st_mode & 0o077 == 0


# --- a machine that embeds through a worker ---------------------------------


@pytest.fixture
def remote(fake_models, home):
    """A real worker serving HTTPS on a free local port, from a thread (fine for fake models; PyTorch needs the main
    one). Records the paths of the requests that reach it."""
    import uvicorn

    found, seen, app = worker.credentials(), [], worker.app(worker.credentials().token)

    async def recording(scope, receive, send):
        if scope["type"] == "http":
            seen.append(scope["path"])
        await app(scope, receive, send)

    config = uvicorn.Config(
        recording,
        host="127.0.0.1",
        port=0,
        log_level="warning",
        ssl_certfile=str(found.cert),
        ssl_keyfile=str(found.private_key),
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10  # a hang detector, not a timing assumption
    while not server.started:
        assert time.monotonic() < deadline, "the worker did not start"
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield types.SimpleNamespace(
        url=f"https://{found.key}@127.0.0.1:{port}",
        address=f"https://127.0.0.1:{port}",
        port=port,
        seen=seen,
        **found._asdict(),
    )
    server.should_exit = True
    thread.join(timeout=10)


def test_embed_through_a_remote_worker(corpus, fake_models, remote, monkeypatch, capsys):
    db, _ = corpus
    monkeypatch.setattr(fake_models, "REMOTE", remote.url)
    monkeypatch.setattr(fake_models, "calibrate_batch_size", lambda *a: pytest.fail("the model loaded here"))
    fake_models.embed_pending(db)
    rows = db.execute("""SELECT c.seq, t.title, t.text, v.vec FROM chunks c JOIN chunk_text t ON t.id = c.id
                         JOIN vectors v ON v.chunk = c.id""").fetchall()
    assert len(rows) == 3
    for seq, title, text, vec in rows:
        expected = fake_vector(fake_models._text(seq, title, text)).astype(np.float16)
        assert np.frombuffer(vec, np.float16).tolist() == expected.tolist()
    assert store.get_meta(db, "model") == fake_models.EMBED_MODEL
    err = capsys.readouterr().err
    assert f"embedding 3 chunks with {fake_models.EMBED_MODEL} on {remote.address}" in err
    assert remote.token not in err  # the key stays off the screen


def test_searches_embed_their_query_on_the_remote(embedded, fake_models, remote, monkeypatch):
    db, _ = embedded
    here = [h["id"] for h in Index(db).vsearch("brush strokes")]
    monkeypatch.setattr(fake_models, "REMOTE", remote.url)
    assert [h["id"] for h in Index(db).vsearch("brush strokes")] == here
    assert "/embed" in remote.seen  # it really asked the remote


def test_the_workers_certificate_is_checked_before_anything_is_sent(corpus, fake_models, remote, monkeypatch):
    # A machine posing as the worker must get neither the token nor any text.
    monkeypatch.setattr(fake_models, "REMOTE", f"https://{remote.token}.{'A' * 43}@127.0.0.1:{remote.port}")
    with pytest.raises(ConnectionError, match="certificate"):
        fake_models.embed_pending(corpus[0])
    assert remote.seen == []


def test_a_wrong_token_is_refused(corpus, fake_models, remote, monkeypatch):
    monkeypatch.setattr(fake_models, "REMOTE", f"https://wrong.{remote.pin}@127.0.0.1:{remote.port}")
    with pytest.raises(ConnectionError, match="refused the key"):
        fake_models.embed_pending(corpus[0])


def test_a_failing_worker_is_a_clear_error(corpus, fake_models, remote, monkeypatch):
    monkeypatch.setattr(fake_models, "REMOTE", remote.url)
    fake_models.remote_model()  # asked before the model fails

    def broken(self, texts, *args, **kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(FakeSentenceTransformer, "encode_document", broken)
    with pytest.raises(ConnectionError, match=f"embedding remote {remote.address} answered 500"):
        fake_models.embed_pending(corpus[0])


@pytest.mark.parametrize(
    "url", ["http://t.p@workstation:8766", "https://workstation:8766", "https://token@workstation:8766", "pc:8766"]
)
def test_the_remote_must_be_the_https_address_with_the_key(corpus, fake_models, monkeypatch, url):
    monkeypatch.setattr(fake_models, "REMOTE", url)
    with pytest.raises(ConnectionError, match=r"SIFT_EMBED_REMOTE must be the https://<key>@"):
        fake_models.embed_pending(corpus[0])


def test_an_unreachable_remote_is_a_clear_error(corpus, fake_models, monkeypatch):
    monkeypatch.setattr(fake_models, "REMOTE", nobody_listens())
    with pytest.raises(
        ConnectionError, match=re.escape("embedding remote https://127.0.0.1:0 is unreachable")
    ) as error:
        fake_models.embed_pending(corpus[0])
    assert "s3cret" not in str(error.value)


def test_a_remote_with_another_model_is_refused(embedded, fake_models, monkeypatch):
    monkeypatch.setattr(fake_models, "REMOTE", "https://t.p@workstation:8766")
    monkeypatch.setattr(fake_models, "remote_model", lambda: ("other/model", 64))
    with pytest.raises(SystemExit, match=f"index was embedded with {fake_models.EMBED_MODEL}"):
        fake_models.embed_pending(embedded[0])


def test_the_cli_reports_an_unreachable_remote(corpus, fake_models, monkeypatch):
    monkeypatch.setattr(fake_models, "REMOTE", nobody_listens())
    with pytest.raises(SystemExit, match="^" + re.escape("error: embedding remote https://127.0.0.1:0 is unreachable")):
        cli.main(["embed"])


# --- running the worker -----------------------------------------------------


@pytest.fixture
def started(monkeypatch):
    """Captures how the worker starts uvicorn, instead of serving."""
    import uvicorn

    configs = []

    async def capture(self, *args, **kwargs):
        configs.append(self.config)

    monkeypatch.setattr(uvicorn.Server, "serve", capture)
    return configs


def test_worker_serves_https_with_its_key_on_all_interfaces(home, fake_models, started, monkeypatch, capsys):
    calibrated = []
    monkeypatch.setattr(fake_models, "calibrate_batch_size", lambda *a: calibrated.append(True))
    worker.serve()
    (config,) = started
    found = worker.credentials()
    assert (config.host, config.port) == ("0.0.0.0", 8766)
    assert (config.ssl_certfile, config.ssl_keyfile) == (str(found.cert), str(found.private_key))
    assert calibrated == [True]  # batches sized to this GPU, before the first request
    assert f"SIFT_EMBED_REMOTE=https://{found.key}@" in capsys.readouterr().err  # what to set on the other machine


@pytest.mark.parametrize("error", [KeyboardInterrupt, OSError])
def test_worker_raises_server_errors_as_they_are(home, fake_models, monkeypatch, error):
    # The idle check shares a task group with uvicorn, which wraps errors in an exception group: Ctrl+C would print
    # a traceback, and "address already in use" would hide inside one.
    import uvicorn

    async def failing(self, *args, **kwargs):
        raise error("stop")

    monkeypatch.setattr(uvicorn.Server, "serve", failing)
    monkeypatch.setattr(fake_models, "calibrate_batch_size", lambda *a: None)
    with pytest.raises(error, match="stop"):
        worker.serve()


def test_cli_worker(home, monkeypatch):
    calls = []
    monkeypatch.setattr("sift.worker.serve", lambda *args: calls.append(args))
    cli.main(["worker", "--port", "9000"])
    assert calls == [("0.0.0.0", 9000)]
