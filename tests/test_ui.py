"""`sift ui`: the index in a browser. `sift serve` serves it under /ui: a page of filters and folders, results, and
the document they lead to, whose parts htmx swaps in place."""

import html
import re
import socket
import threading
import time
import urllib.parse

import pytest

from sift import cli, indexing, server, store, ui
from sift.search import Index

from .conftest import Client, write

LOCAL = "http://127.0.0.1:8765"


def page(index):
    """A browser on the app `sift serve` runs over HTTP, for this index."""
    return Client(server.http_app(server.build(index)[0], index, "127.0.0.1"), base_url=LOCAL)


@pytest.fixture
def browser(embedded):
    return page(Index(embedded[0]))


def ids(response):
    """The results a page lists, in order."""
    return re.findall(r'class="hit[^"]*" data-id="([^"]+)"', response.text)


def test_the_page_shows_what_is_indexed(browser):
    response = browser.get("/ui/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "notes" in response.text
    assert "3 documents" in response.text
    assert '<script src="/ui/static/htmx.min.js"' in response.text


def test_the_address_of_the_server_leads_to_the_ui(browser):
    response = browser.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/ui/"


def test_a_search_lists_its_hits_with_the_words_marked(browser):
    response = browser.get("/ui/", params={"q": "brush"})
    assert ids(response) == ["notes/robots/brush.md"]
    assert "Brush stroke planning" in response.text
    assert re.search(r"<mark>brush\w*</mark>", response.text, re.I)
    assert "1 result" in response.text
    snippet = re.search(r'<span class="snippet">(.*?)</span>', response.text)[1]
    assert snippet.startswith("A painting robot plans <mark>brush")  # not the title again, nor its "#"


@pytest.mark.parametrize("mode", ["search", "vsearch", "query"])
def test_each_mode_finds_what_its_search_finds(embedded, browser, mode):
    expected = [h["id"] for h in getattr(Index(embedded[0]), mode)("robots painting pigments", 30)]
    assert ids(browser.get("/ui/", params={"q": "robots painting pigments", "mode": mode})) == expected


def test_an_unknown_mode_is_a_keyword_search(browser):
    assert ids(browser.get("/ui/", params={"q": "reward", "mode": "nonsense"})) == ["notes/rl.md"]


def test_filters_narrow_the_results_and_show_as_chips(browser):
    response = browser.get("/ui/", params={"q": "reward OR brush", "f": "year=2024"})
    assert ids(response) == ["notes/robots/brush.md"]  # the number matches the number in the front matter
    chip = re.search(r'<a class="chip" href="([^"]+)"[^>]*>year: 2024', response.text)
    assert chip
    assert "f=" not in html.unescape(chip[1])  # removing the chip removes the filter


def test_facets_count_the_values_of_the_results_and_filter_by_them(browser):
    response = browser.get("/ui/", params={"c": "notes"})
    link = re.search(r'<a href="([^"]+)"[^>]*>cs\.RO <span class="n">1</span>', response.text)
    assert link
    filtered = browser.get(html.unescape(link[1]))
    assert ids(filtered) == ["notes/robots/brush.md"]


def test_a_field_whose_values_all_differ_is_no_facet(embedded):
    # Filtering by an id or a link finds one document each time: such a field only crowds the facets.
    db, root = embedded
    for n in range(ui.FACET_VALUES + 1):
        write(root / "refs" / f"{n}.md", f"---\nref: R-{n}\nkind: memo\n---\n# Memo {n}\n")
    indexing.index(db)
    side = re.search(r'<nav class="side".*?</nav>', page(Index(db)).get("/ui/", params={"c": "notes"}).text, re.S)[0]
    assert "<h2>kind</h2>" in side
    assert "<h2>ref</h2>" not in side


def test_browsing_a_collection_shows_its_folders_and_documents(browser):
    response = browser.get("/ui/", params={"c": "notes"})
    assert ids(response) == ["notes/plain.txt", "notes/rl.md", "notes/robots/brush.md"]
    folder = re.search(r'<a href="([^"]+)"[^>]*>robots/</a>', response.text)
    inside = browser.get(html.unescape(folder[1]))
    assert ids(inside) == ["notes/robots/brush.md"]
    assert "robots" in re.search(r'<nav class="crumbs">(.*?)</nav>', inside.text, re.S)[1]


def test_a_search_inside_a_folder_stays_inside_it(browser):
    response = browser.get("/ui/", params={"q": "reward OR brush", "c": "notes", "path": "robots/"})
    assert ids(response) == ["notes/robots/brush.md"]


def test_more_results_are_one_click_away(embedded):
    db, root = embedded
    for n in range(5):
        write(root / f"extra{n}.md", f"# Extra {n}\n\nbrush\n")
    indexing.index(db)
    response = page(Index(db)).get("/ui/", params={"q": "brush", "k": "3"})
    assert len(ids(response)) == 3
    more = re.search(r'<a class="more" href="([^"]+)"', response.text)
    assert f"k={3 + ui.PAGE}" in html.unescape(more[1])  # a page more


def test_a_document_opens_in_the_reader_with_its_hit_marked(browser):
    response = browser.get("/ui/", params={"q": "reinforcement", "doc": "notes/rl.md", "lines": "3-3"})
    reader = re.search(r'<article id="reader".*?</article>', response.text, re.S)[0]
    assert "<h1>Reward shaping</h1>" in reader
    assert '<div class="l h1" id="L1">Reward shaping</div>' in reader  # a heading, without its "#"
    assert re.search(r'<div class="l hit" id="L3">.*<mark>reinforcement</mark>', reader)
    assert 'id="L2"' in reader  # the whole document, not only the hit


def test_the_reader_shows_the_front_matter_as_filters(browser):
    response = browser.get("/ui/", params={"doc": "notes/robots/brush.md"})
    reader = re.search(r'<article id="reader".*?</article>', response.text, re.S)[0]
    link = re.search(r'<a href="([^"]+)"[^>]*>cs\.GR</a>', reader)
    assert ids(browser.get(html.unescape(link[1]))) == ["notes/robots/brush.md"]


def test_links_in_the_front_matter_open_their_page(embedded):
    db, root = embedded
    write(root / "linked.md", "---\nurl: https://example.org/paper\n---\n# Linked\n")
    indexing.index(db)
    browser = page(Index(db))
    reader = browser.get("/ui/", params={"doc": "notes/linked.md"}).text
    assert '<a href="https://example.org/paper" target="_blank" rel="noopener" hx-boost="false">' in reader
    side = re.search(r'<nav class="side".*?</nav>', browser.get("/ui/", params={"c": "notes"}).text, re.S)[0]
    assert "example.org" not in side  # no facet: a link is no kind of document


def test_similar_documents_are_one_click_away(embedded, browser):
    response = browser.get("/ui/", params={"similar": "notes/rl.md"})
    assert ids(response) == [h["id"] for h in Index(embedded[0]).similar("notes/rl.md", 30)]
    assert "Similar to" in response.text


def test_htmx_gets_only_the_part_it_swaps(browser):
    asked = {"HX-Request": "true", "HX-Target": "explore"}
    explore = browser.get("/ui/", params={"q": "brush"}, headers=asked)
    assert "<html" not in explore.text
    assert '<div id="explore"' in explore.text
    assert 'id="controls" hx-swap-oob="true"' in explore.text  # the form follows the view
    assert '<article id="reader"' not in explore.text
    reader = browser.get("/ui/", params={"doc": "notes/rl.md"}, headers=asked | {"HX-Target": "reader"})
    assert reader.text.lstrip().startswith('<article id="reader"')
    assert '<div id="explore"' not in reader.text


def test_going_back_gets_the_whole_page(browser):
    headers = {"HX-Request": "true", "HX-Target": "explore", "HX-History-Restore-Request": "true"}
    assert "<html" in browser.get("/ui/", params={"q": "brush"}, headers=headers).text


@pytest.mark.parametrize(
    "params, message",
    [
        ({"q": "brush", "f": "bad-field=1"}, "invalid filter field"),
        ({"q": "brush", "f": "no equals sign"}, "must be field=value"),
        ({"q": "brush", "c": "nope"}, "no collection named"),
        ({"doc": "notes/missing.md"}, "unknown document"),
        ({"doc": "notes/rl.md", "lines": "9-1"}, "lines must be"),
    ],
)
def test_errors_show_in_place(browser, params, message):
    response = browser.get("/ui/", params=params)
    assert response.status_code == 200  # htmx swaps only successful answers
    assert re.search(r'<p class="error">[^<]*' + re.escape(html.escape(message, quote=True)), response.text)


def test_searching_by_meaning_needs_embeddings(corpus, fake_models):
    response = page(Index(corpus[0])).get("/ui/", params={"q": "brush", "mode": "vsearch"})
    assert "no embeddings yet" in response.text


def test_text_from_documents_is_escaped(embedded):
    db, root = embedded
    write(root / "evil.md", "# <script>alert(1)</script>\n\nbrush <img src=x onerror=alert(2)>\n")
    indexing.index(db)
    browser = page(Index(db))
    for response in (browser.get("/ui/", params={"q": "brush"}), browser.get("/ui/", params={"doc": "notes/evil.md"})):
        assert "<script>alert" not in response.text
        assert "<img src=x" not in response.text
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text


def test_the_original_file_opens_in_the_browser(embedded, browser, tmp_path):
    db, root = embedded
    response = browser.get("/ui/file", params={"id": "notes/rl.md"})
    assert response.status_code == 200
    assert response.content == (root / "rl.md").read_bytes()
    assert response.headers["content-disposition"].startswith("inline")
    assert 'href="/ui/file?id=notes%2Frl.md"' in browser.get("/ui/", params={"doc": "notes/rl.md"}).text
    write(tmp_path / "qa" / "faq.jsonl", '{"id": "q1", "text": "a record"}\n')
    store.add_collection(db, tmp_path / "qa", "qa")
    indexing.index(db)
    for missing in ("notes/missing.md", "qa/faq/q1"):  # unknown, and a record, which is no file of its own
        assert browser.get("/ui/file", params={"id": missing}).status_code == 404
    assert "/ui/file?" not in browser.get("/ui/", params={"doc": "qa/faq/q1"}).text


def test_the_ui_answers_local_hosts_only(browser):
    # A web page can reach a local server under its own domain (DNS rebinding) and read the documents through it.
    assert browser.get("/ui/", headers={"Host": "attacker.example:8765"}).status_code == 400
    assert browser.get("/ui/", headers={"Host": "localhost:8765"}).status_code == 200


def test_a_server_for_other_machines_answers_any_host(embedded):
    index = Index(embedded[0])
    app = server.http_app(server.build(index)[0], index, "0.0.0.0")
    assert Client(app, base_url="http://workstation:8765").get("/ui/").status_code == 200


def test_the_ui_brings_its_scripts_and_styles(browser):
    for name, kind in (("htmx.min.js", "javascript"), ("ui.js", "javascript"), ("ui.css", "text/css")):
        response = browser.get(f"/ui/static/{name}")
        assert response.status_code == 200
        assert kind in response.headers["content-type"]


def test_searches_run_on_the_event_loop_thread(embedded, monkeypatch):
    # PyTorch crashes in other threads on Python 3.15 (Windows, macOS), so the UI must not search in a thread pool.
    index, threads = Index(embedded[0]), []
    real = index.query
    monkeypatch.setattr(index, "query", lambda *a, **kw: threads.append(threading.current_thread()) or real(*a, **kw))
    page(index).get("/ui/", params={"q": "brush", "mode": "query"})
    assert threads == [threading.main_thread()]


def test_the_collections_are_counted_once_per_change_of_the_index(embedded, monkeypatch):
    # Counting millions of documents for every keystroke would make typing slow.
    db, root = embedded
    browser, counted = page(Index(db)), []
    real = ui._collections
    monkeypatch.setattr(ui, "_collections", lambda db: counted.append(1) or real(db))
    browser.get("/ui/", params={"q": "brush"})
    browser.get("/ui/", params={"q": "brus"})
    assert len(counted) == 1
    write(root / "new.md", "# New\n")
    indexing.index(db)
    assert "4 documents" in browser.get("/ui/").text


# --- sift ui ------------------------------------------------------------------


@pytest.fixture
def opened(monkeypatch):
    """The pages `sift ui` opens in the browser, instead of opening them."""
    pages = []
    monkeypatch.setattr(ui.webbrowser, "open", pages.append)
    return pages


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def answering(status):
    """A server on a free local port that answers every request with `status`; returns its port and a stop()."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(status)
            self.end_headers()

        def log_message(self, *args):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd.server_address[1], lambda: (httpd.shutdown(), httpd.server_close())


def test_sift_ui_opens_the_server_that_runs(home, opened, monkeypatch):
    port, stop = answering(200)
    monkeypatch.setattr(server, "serve", lambda *a, **kw: pytest.fail("started another server"))
    try:
        cli.main(["ui", "--port", str(port)])
    finally:
        stop()
    assert opened == [f"http://127.0.0.1:{port}/ui/"]


def test_sift_ui_starts_the_server_when_none_runs(home, opened, monkeypatch):
    port, started = free_port(), []

    def serve(host, port):  # listens, as `sift serve` does once its models are loaded, until the page opened
        started.append((host, port))
        with socket.create_server(("127.0.0.1", port)):
            deadline = time.monotonic() + 10  # a hang detector
            while not opened:
                assert time.monotonic() < deadline, "the page never opened"
                time.sleep(0.01)

    monkeypatch.setattr(server, "serve", serve)
    cli.main(["ui", "--port", str(port)])
    assert started == [("127.0.0.1", port)]
    assert opened == [f"http://127.0.0.1:{port}/ui/"]


def test_sift_ui_on_all_interfaces_opens_the_local_address(home, opened, monkeypatch):
    port, stop = answering(200)
    try:
        cli.main(["ui", "--host", "0.0.0.0", "--port", str(port)])
    finally:
        stop()
    assert opened == [f"http://127.0.0.1:{port}/ui/"]


def test_sift_ui_needs_a_server_with_the_ui(home, opened):
    port, stop = answering(404)  # an older `sift serve`
    try:
        with pytest.raises(SystemExit, match=f"the server on port {port} has no UI"):
            cli.main(["ui", "--port", str(port)])
    finally:
        stop()
    assert opened == []


def test_the_page_opens_only_once_the_server_answers(opened):
    ui._open_when_up("127.0.0.1", free_port(), "http://example/ui/", wait=0.3)  # nobody ever listens
    assert opened == []


def test_serve_says_where_the_ui_is(embedded, monkeypatch, capsys):
    import uvicorn

    async def nothing(self, *args, **kwargs):
        pass

    monkeypatch.setattr(uvicorn.Server, "serve", nothing)
    server.serve(port=9000)
    assert "browse the index at http://127.0.0.1:9000/ui/" in capsys.readouterr().err


def test_links_carry_the_whole_view(browser):
    # Every view has its address: reloading or sharing a link shows the same page.
    response = browser.get("/ui/", params={"q": "brush", "mode": "query", "c": "notes", "f": "year=2024"})
    link = re.search(r'class="hit[^"]*" data-id="notes/robots/brush.md" href="([^"]+)"', response.text)
    params = urllib.parse.parse_qs(urllib.parse.urlsplit(html.unescape(link[1])).query)
    assert params == {
        "q": ["brush"],
        "mode": ["query"],
        "c": ["notes"],
        "f": ["year=2024"],
        "doc": ["notes/robots/brush.md"],
        "lines": params["lines"],
    }
