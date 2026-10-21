"""`sift worker` lends a machine's GPU to other machines' sift: they send texts over HTTPS, with the worker's key, and
get vectors and reranking scores back."""

import asyncio
import json
import os
import re
import select
import socket
import subprocess
import sys
import textwrap
import threading
import time
import types

import anyio
import numpy as np
import pytest

from sift import __version__, cli, embed, models, remote, store, worker
from sift.search import Index

from .conftest import FakeSentenceTransformer, NumpyModels, fake_vector, nobody_listens

TOKEN = "s3cret-token"


class Client:
    """Requests to an app that runs on this, the main, thread, as under uvicorn. Starlette's TestClient runs it in a
    thread of its own, where PyTorch crashes on Python 3.15 (macOS)."""

    def __init__(self, app, headers=None):
        self.app, self.headers = app, headers or {}

    def request(self, method, path, **kwargs):
        import httpx2  # what MCP and Starlette use

        async def send():
            transport = httpx2.ASGITransport(self.app, client=("testclient", 50000))
            async with httpx2.AsyncClient(transport=transport, base_url="http://testserver", headers=self.headers) as c:
                return await c.request(method, path, **kwargs)

        return anyio.run(send)

    def get(self, path, **kwargs):
        return self.request("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self.request("POST", path, **kwargs)


@pytest.fixture
def client(fake_models):
    return Client(worker.app(TOKEN, fake_models), headers={"Authorization": f"Bearer {TOKEN}"})


def post(client, path, payload):
    return client.post(path, content=json.dumps(payload), headers={"Content-Type": "application/json"})


def test_the_worker_says_which_models_it_runs(client, fake_models):
    assert client.get("/").json() == {
        "model": fake_models.name,
        "dim": 64,
        "reranker": fake_models.reranker_name,
        "version": __version__,
    }


def test_the_worker_embeds_texts_as_this_machine_would(client, fake_models, capsys):
    response = post(client, "/embed", {"texts": ["brush strokes", "pigments"]})
    assert response.status_code == 200
    vectors = np.frombuffer(response.content, np.float16).reshape(2, 64)
    assert vectors.tolist() == fake_models.encode(["brush strokes", "pigments"]).tolist()
    assert "2 texts embedded, 0 queries reranked (last for testclient)" in capsys.readouterr().err  # it shows it works


def test_the_worker_embeds_queries_as_queries(client, monkeypatch):
    seen = []
    real = FakeSentenceTransformer.encode_query
    monkeypatch.setattr(
        FakeSentenceTransformer, "encode_query", lambda self, t, **kw: seen.extend(t) or real(self, t, **kw)
    )
    post(client, "/embed", {"texts": ["how do robots paint"], "query": True})
    assert seen == ["how do robots paint"]


def test_the_worker_reranks_as_this_machine_would(client, fake_models):
    texts = ["brush strokes on canvas", "reward shaping"]
    response = post(client, "/rerank", {"query": "brush strokes", "texts": texts})
    assert response.json() == {"scores": fake_models.rerank("brush strokes", texts).tolist()}


@pytest.mark.parametrize("path, payload", [("/embed", {"texts": ["a"]}), ("/rerank", {"query": "q", "texts": ["a"]})])
def test_the_models_run_on_the_main_thread(client, fake_models, monkeypatch, path, payload):
    # As under uvicorn: PyTorch crashes in other threads on Python 3.15 (macOS). Starlette's TestClient runs the app
    # in a thread of its own, and so crashed the tests.
    threads = []
    for name in ("encode", "rerank"):
        real = getattr(fake_models, name)
        monkeypatch.setattr(
            fake_models, name, lambda *a, real=real, **kw: threads.append(threading.current_thread()) or real(*a, **kw)
        )
    assert post(client, path, payload).status_code == 200
    assert threads == [threading.main_thread()]


@pytest.mark.parametrize(
    "path, content",
    [
        ("/embed", '{"texts": "not a list"}'),
        ("/embed", '{"texts": [1, 2]}'),
        ("/embed", "[]"),
        ("/embed", "not json"),
        ("/rerank", '{"texts": ["a"]}'),
        ("/rerank", '{"query": 1, "texts": ["a"]}'),
        ("/rerank", '{"query": "q", "texts": "a"}'),
    ],
)
def test_the_worker_refuses_malformed_requests(client, path, content):
    response = client.post(path, content=content, headers={"Content-Type": "application/json"})
    assert response.status_code == 400


@pytest.mark.parametrize("authorization", [None, "Bearer wrong", f"Basic {TOKEN}", TOKEN])
def test_the_worker_answers_only_with_its_token(fake_models, authorization):
    c = Client(worker.app(TOKEN, fake_models), headers={"Authorization": authorization} if authorization else {})
    assert c.get("/").status_code == 401
    assert c.post("/embed", json={"texts": ["x"]}).status_code == 401
    assert c.post("/rerank", json={"query": "x", "texts": ["x"]}).status_code == 401


# --- the worker's key --------------------------------------------------------


def test_the_worker_makes_its_key_once_and_keeps_it(home):
    # Other machines keep the key they were given, so the worker must not make a new one on every start.
    first = worker.credentials()
    assert first == worker.credentials()
    assert first.key == f"{first.token}.{first.pin}"
    assert len(first.token) >= 40  # 32 random bytes
    assert first.cert.parent == first.private_key.parent == home / "worker"


def test_another_home_makes_another_key(tmp_path, monkeypatch, configure):
    configure(home=tmp_path / "a")
    a = worker.credentials()
    configure(home=tmp_path / "b")
    b = worker.credentials()
    assert a.token != b.token
    assert a.pin != b.pin


@pytest.mark.skipif(os.name == "nt", reason="Windows guards files with ACLs; a user's home is private by default")
def test_the_key_files_are_private(home, fake_models):
    found = worker.credentials()
    for path in (home / "worker" / "token", found.private_key):
        assert path.stat().st_mode & 0o077 == 0


# --- a machine that embeds and reranks through a worker ----------------------


@pytest.fixture
def live(fake_models, home):
    """A real worker serving HTTPS on a free local port, from a thread, with models of its own that compute as the
    fakes do, without PyTorch, which needs the main thread. Records the paths of the requests that reach it, and the
    client ports they came from, one per connection. It closes connections that idle for a second."""
    import uvicorn

    found, seen, clients, worker_models, loops = worker.credentials(), [], [], NumpyModels(), []
    app = worker.app(found.token, worker_models)

    async def recording(scope, receive, send):
        if scope["type"] == "http":
            seen.append(scope["path"])
            clients.append(scope["client"][1])
            loops[:] = [asyncio.get_running_loop()]
        await app(scope, receive, send)

    tls = {"ssl_certfile": str(found.cert), "ssl_keyfile": str(found.private_key)}
    # As the worker does, it stops within a second even with connections open, such as one a client kept.
    config = uvicorn.Config(
        recording, port=0, log_level="warning", timeout_keep_alive=1, timeout_graceful_shutdown=1, **tls
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
        clients=clients,
        server=server,
        models=worker_models,
        **found._asdict(),
    )
    for loop in loops:  # drop the connections clients kept, as a worker that stops does, before stopping
        loop.call_soon_threadsafe(lambda: [c.transport.abort() for c in list(server.server_state.connections)])
    deadline = time.monotonic() + 10  # a hang detector
    while server.server_state.connections and time.monotonic() < deadline:
        time.sleep(0.01)
    server.should_exit = True
    thread.join(timeout=10)


def test_embed_through_a_worker(corpus, fake_models, live, monkeypatch, capsys, configure):
    db, _ = corpus
    configure(remote=live.url)
    monkeypatch.setattr(fake_models, "embedder", lambda: pytest.fail("the model loaded here"))
    embed.embed_pending(db)
    rows = db.execute("""SELECT c.seq, t.title, t.text, v.vec FROM chunks c JOIN chunk_text t ON t.id = c.id
                         JOIN vectors v ON v.chunk = c.id""").fetchall()
    assert len(rows) == 3
    for seq, title, text, vec in rows:
        expected = fake_vector(embed._text(seq, title, text)).astype(np.float16)
        assert np.frombuffer(vec, np.float16).tolist() == expected.tolist()
    assert store.get_meta(db, "model") == fake_models.name
    err = capsys.readouterr().err
    assert f"embedding 3 chunks with {fake_models.name} on {live.address}" in err
    assert live.token not in err  # the key stays off the screen


@pytest.mark.parametrize("mode, paths", [("vsearch", ["/embed"]), ("query", ["/embed", "/rerank"])])
def test_searches_embed_and_rerank_on_the_worker(embedded, live, monkeypatch, mode, paths, configure):
    db, _ = embedded
    here = getattr(Index(db), mode)("brush strokes")
    configure(remote=live.url)
    there = getattr(Index(db), mode)("brush strokes")
    assert [h["id"] for h in there] == [h["id"] for h in here]
    assert [h["score"] for h in there] == pytest.approx([h["score"] for h in here], abs=1e-3)
    assert set(paths) <= set(live.seen)  # it really asked the worker


def test_requests_to_a_worker_share_one_connection(live):
    # A connection per request repeated the name lookup, the connection attempts and the TLS handshake for every 256
    # texts.
    worker_remote = remote.Remote(live.url)
    for text in ("brush strokes", "pigments", "canvas"):
        worker_remote.encode([text])
    assert sorted(live.seen) == ["/", "/embed", "/embed", "/embed"]  # three batches, and the vectors' length once
    assert len(set(live.clients)) == 1


def test_a_connection_the_worker_closed_is_opened_again(live):
    # The worker closes connections that idle for a few seconds, as between two searches; the next request must
    # still get its answer.
    worker_remote = remote.Remote(live.url)
    worker_remote.encode(["brush strokes"])
    deadline = time.monotonic() + 10  # a hang detector, not a timing assumption
    while not all(c.transport.is_closing() for c in live.server.server_state.connections):
        assert time.monotonic() < deadline, "the worker kept the idle connection"
        time.sleep(0.02)
    assert worker_remote.encode(["pigments"]).tolist() == live.models.encode(["pigments"]).tolist()
    assert len(set(live.clients)) == 2


@pytest.fixture
def ipv6_listener():
    """A socket listening on the IPv6 loopback that never answers, as an address the worker doesn't serve."""
    listener = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    try:
        listener.bind(("::1", 0))
    except OSError:
        listener.close()
        pytest.skip("this machine has no IPv6 loopback")
    listener.listen()
    yield listener
    listener.close()


def test_a_worker_is_reached_over_ipv4_where_it_listens(live, ipv6_listener, monkeypatch):
    # The worker listens on IPv4, but a name often resolves to an IPv6 link-local address first. Where a firewall
    # dropped those connection attempts, each request waited 21 s for the attempt to fail before trying IPv4.
    resolve = socket.getaddrinfo

    def names(host, port, family=0, type=0, proto=0, flags=0):
        if host != "workstation.test":
            return resolve(host, port, family, type, proto, flags)
        ipv6 = (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("::1", ipv6_listener.getsockname()[1], 0, 0))
        ipv4 = (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", live.port))
        return [a for a in (ipv6, ipv4) if family in (0, a[0])]

    monkeypatch.setattr(socket, "getaddrinfo", names)
    monkeypatch.setattr(remote, "TIMEOUT", 2)  # the IPv6 address takes the connection but never answers
    worker_remote = remote.Remote(f"https://{live.token}.{live.pin}@workstation.test:{live.port}")
    assert worker_remote.encode(["brush strokes"]).tolist() == live.models.encode(["brush strokes"]).tolist()
    ready, _, _ = select.select([ipv6_listener], [], [], 0)
    assert not ready  # no connection waits there


def test_searching_through_a_worker_never_loads_pytorch(tmp_path):
    # PyTorch takes seconds to load, more on a notebook; a machine whose worker embeds and reranks doesn't need it.
    script = tmp_path / "search.py"
    script.write_text(
        textwrap.dedent(f"""
            import os
            import sys
            os.environ["SIFT_EMBED_REMOTE"] = "https://t.p@worker:8766"
            import numpy as np
            from sift import indexing, remote, store
            from sift.search import Index

            remote.Remote.encode = lambda self, texts, query=False: np.full((len(texts), 4), 0.5, np.float16)
            remote.Remote.rerank = lambda self, query, texts: np.ones(len(texts), np.float32)
            notes = {str(tmp_path / "notes")!r}
            __import__("os").makedirs(notes)
            open(notes + "/brush.md", "w").write("# Brush strokes\\n\\nA robot paints.\\n")
            db = store.connect({str(tmp_path / "home")!r})
            store.add_collection(db, notes, "notes")
            indexing.index(db)
            for (chunk,) in db.execute("SELECT id FROM chunks").fetchall():
                db.execute("INSERT INTO vectors VALUES(?, ?)", (chunk, np.full(4, 0.5, np.float16).tobytes()))
            store.set_meta(db, "dim", 4)
            db.commit()
            index = Index(db)
            assert index.vsearch("robot") and index.query("robot")
            print("torch" in sys.modules)
        """),
        encoding="utf-8",
    )
    run = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    assert run.stdout.strip() == "False"


def test_the_workers_certificate_is_checked_before_anything_is_sent(corpus, fake_models, live, monkeypatch, configure):
    # A machine posing as the worker must get neither the token nor any text.
    configure(remote=f"https://{live.token}.{'A' * 43}@127.0.0.1:{live.port}")
    with pytest.raises(ConnectionError, match="certificate"):
        embed.embed_pending(corpus[0])
    assert live.seen == []


def test_a_wrong_token_is_refused(corpus, fake_models, live, monkeypatch, configure):
    configure(remote=f"https://wrong.{live.pin}@127.0.0.1:{live.port}")
    with pytest.raises(ConnectionError, match="refused the key"):
        embed.embed_pending(corpus[0])


def test_a_failing_worker_is_a_clear_error(corpus, fake_models, live, monkeypatch, configure):
    configure(remote=live.url)
    models.default().load()  # the worker is asked about its models before they fail

    def broken(texts, *args, **kwargs):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    monkeypatch.setattr(live.models, "encode", broken)
    with pytest.raises(ConnectionError, match=f"embedding remote {live.address} answered 500"):
        embed.embed_pending(corpus[0])


@pytest.mark.parametrize(
    "url", ["http://t.p@workstation:8766", "https://workstation:8766", "https://token@workstation:8766", "pc:8766"]
)
def test_the_remote_must_be_the_https_address_with_the_key(corpus, fake_models, monkeypatch, url, configure):
    configure(remote=url)
    with pytest.raises(ConnectionError, match=r"SIFT_EMBED_REMOTE must be the https://<key>@"):
        embed.embed_pending(corpus[0])


def test_an_unreachable_remote_is_a_clear_error(corpus, fake_models, monkeypatch, configure):
    configure(remote=nobody_listens())
    unreachable = re.escape("embedding remote https://127.0.0.1:0 is unreachable")
    with pytest.raises(ConnectionError, match=unreachable) as error:
        embed.embed_pending(corpus[0])
    assert "s3cret" not in str(error.value)


def test_a_remote_with_another_model_is_refused(embedded, fake_models, monkeypatch, configure):
    configure(remote="https://t.p@workstation:8766")
    monkeypatch.setattr(remote.Remote, "name", "other/model")
    with pytest.raises(SystemExit, match=f"index was embedded with {fake_models.name}"):
        embed.embed_pending(embedded[0])


def test_the_cli_reports_an_unreachable_remote(corpus, fake_models, monkeypatch, configure):
    configure(remote=nobody_listens())
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
    monkeypatch.setattr(fake_models, "prepare", lambda longest: calibrated.append(longest()))
    worker.serve(models=fake_models)
    (config,) = started
    found = worker.credentials()
    assert (config.host, config.port) == ("0.0.0.0", 8766)
    assert (config.ssl_certfile, config.ssl_keyfile) == (str(found.cert), str(found.private_key))
    assert calibrated == [None]  # batches sized to this GPU for the longest input, before the first request
    assert fake_models._loaded  # both models loaded at start, so the first request is quick
    assert f"SIFT_EMBED_REMOTE=https://{found.key}@" in capsys.readouterr().err  # what to set on the other machine


def test_a_worker_serves_its_own_machines_models(home, fake_models, started, monkeypatch, capsys, configure):
    # Even with SIFT_EMBED_REMOTE set: a worker passing work on to another would be a chain nobody asked for.
    configure(remote="https://t.p@elsewhere:8766")
    monkeypatch.setattr("sift.local.Local", lambda: fake_models)
    worker.serve()
    assert f"embedding and reranking on {fake_models.address}" in capsys.readouterr().err


@pytest.mark.parametrize("error", [KeyboardInterrupt, OSError])
def test_worker_raises_server_errors_as_they_are(home, fake_models, monkeypatch, error):
    # The idle check shares a task group with uvicorn, which wraps errors in an exception group: Ctrl+C would print
    # a traceback, and "address already in use" would hide inside one.
    import uvicorn

    async def failing(self, *args, **kwargs):
        raise error("stop")

    monkeypatch.setattr(uvicorn.Server, "serve", failing)
    monkeypatch.setattr(fake_models, "prepare", lambda longest: None)
    with pytest.raises(error, match="stop"):
        worker.serve(models=fake_models)


def test_cli_worker(home, monkeypatch):
    calls = []
    monkeypatch.setattr("sift.worker.serve", lambda *args: calls.append(args))
    cli.main(["worker", "--port", "9000"])
    assert calls == [("0.0.0.0", 9000)]
