"""`sift worker` lends a machine's GPU to other machines' sift: they send texts over HTTP and get vectors back."""

import json
import threading
import time

import numpy as np
import pytest

from sift import __version__, cli, store, worker
from sift.search import Index

from .conftest import FakeSentenceTransformer, fake_vector, nobody_listens


@pytest.fixture
def client(fake_models):
    from starlette.testclient import TestClient

    with TestClient(worker.app()) as c:
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


@pytest.mark.parametrize(
    "content, content_type",
    [
        ('{"texts": "not a list"}', "application/json"),
        ('{"texts": [1, 2]}', "application/json"),
        ("[]", "application/json"),
        ("not json", "application/json"),
        # A web page can send text/plain to any address without asking, so the worker takes JSON only: otherwise any
        # page the user opens could put the worker's GPU to work.
        ('{"texts": ["x"]}', "text/plain"),
    ],
)
def test_the_worker_refuses_anything_but_json_lists_of_texts(client, content, content_type):
    response = client.post("/embed", content=content, headers={"Content-Type": content_type})
    assert response.status_code in (400, 415)


# --- a machine that embeds through a worker ---------------------------------


@pytest.fixture
def worker_url(fake_models):
    """A real worker on a free local port, serving from a thread (fine for fake models; PyTorch needs the main one)."""
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(worker.app(), host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10  # a hang detector, not a timing assumption
    while not server.started:
        assert time.monotonic() < deadline, "the worker did not start"
        time.sleep(0.01)
    yield f"http://127.0.0.1:{server.servers[0].sockets[0].getsockname()[1]}"
    server.should_exit = True
    thread.join(timeout=10)


def test_embed_through_a_remote_worker(corpus, fake_models, worker_url, monkeypatch, capsys):
    db, _ = corpus
    monkeypatch.setattr(fake_models, "REMOTE", worker_url)
    monkeypatch.setattr(fake_models, "calibrate_batch_size", lambda *a: pytest.fail("the model loaded here"))
    fake_models.embed_pending(db)
    rows = db.execute("""SELECT c.seq, t.title, t.text, v.vec FROM chunks c JOIN chunk_text t ON t.id = c.id
                         JOIN vectors v ON v.chunk = c.id""").fetchall()
    assert len(rows) == 3
    for seq, title, text, vec in rows:
        expected = fake_vector(fake_models._text(seq, title, text)).astype(np.float16)
        assert np.frombuffer(vec, np.float16).tolist() == expected.tolist()
    assert store.get_meta(db, "model") == fake_models.EMBED_MODEL
    assert f"embedding 3 chunks with {fake_models.EMBED_MODEL} on {worker_url}" in capsys.readouterr().err


def test_searches_embed_their_query_on_the_remote(embedded, fake_models, worker_url, monkeypatch):
    db, _ = embedded
    here = [h["id"] for h in Index(db).vsearch("brush strokes")]
    monkeypatch.setattr(fake_models, "REMOTE", worker_url)
    assert [h["id"] for h in Index(db).vsearch("brush strokes")] == here
    monkeypatch.setattr(fake_models, "REMOTE", nobody_listens())
    with pytest.raises(ConnectionError, match="embedding remote"):
        Index(db).vsearch("brush strokes")  # it really asked the remote


def test_an_unreachable_remote_is_a_clear_error(corpus, fake_models, monkeypatch):
    url = nobody_listens()
    monkeypatch.setattr(fake_models, "REMOTE", url)
    with pytest.raises(ConnectionError, match=f"embedding remote {url} is unreachable"):
        fake_models.embed_pending(corpus[0])


def test_a_remote_with_another_model_is_refused(embedded, fake_models, monkeypatch):
    monkeypatch.setattr(fake_models, "REMOTE", "http://workstation:8766")
    monkeypatch.setattr(fake_models, "remote_model", lambda: ("other/model", 64))
    with pytest.raises(SystemExit, match=f"index was embedded with {fake_models.EMBED_MODEL}"):
        fake_models.embed_pending(embedded[0])


def test_the_cli_reports_an_unreachable_remote(corpus, fake_models, monkeypatch):
    url = nobody_listens()
    monkeypatch.setattr(fake_models, "REMOTE", url)
    with pytest.raises(SystemExit, match=f"^error: embedding remote {url} is unreachable"):
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


def test_worker_loads_the_model_and_listens_on_all_interfaces(fake_models, started, monkeypatch, capsys):
    calibrated = []
    monkeypatch.setattr(fake_models, "calibrate_batch_size", lambda *a: calibrated.append(True))
    worker.serve()
    (config,) = started
    assert (config.host, config.port) == ("0.0.0.0", 8766)
    assert calibrated == [True]  # batches sized to this GPU, before the first request
    err = capsys.readouterr().err
    assert "SIFT_EMBED_REMOTE=http://" in err  # tells the user what to set on the other machine


@pytest.mark.parametrize("error", [KeyboardInterrupt, OSError])
def test_worker_raises_server_errors_as_they_are(fake_models, monkeypatch, error):
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
