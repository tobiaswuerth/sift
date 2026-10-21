"""`sift ui`: the index in a browser. `sift serve` serves it under /ui: a page of filters and folders, results, and
the document they lead to, whose parts htmx swaps in place."""

import html
import json
import re
import socket
import threading
import time
import urllib.parse
from pathlib import Path

import pytest

from sift import cli, indexing, server, store, ui
from sift.search import Index

from .conftest import Client, write

LOCAL = "http://127.0.0.1:8765"
SAMPLES = Path(__file__).parent / "samples"
BRUSH = "notes/robots/brush.md"


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


def test_facets_filter_by_numbers_written_as_text(embedded):
    # "007" and "1.10" are text in the front matter; read from the address as the numbers 7 and 1.1, they matched
    # nothing.
    db, root = embedded
    write(root / "agent.md", '---\ncode: "007"\nversion: "1.10"\nyear: 2024\n---\n# Agent\n')
    indexing.index(db)
    browser = page(Index(db))
    for pair in ("code=007", "version=1.10"):
        assert ids(browser.get("/ui/", params={"c": "notes", "f": pair})) == ["notes/agent.md"]
    assert ids(browser.get("/ui/", params={"c": "notes", "f": "year=2024"})) == ["notes/agent.md", BRUSH]


def test_yes_no_fields_are_facets(embedded):
    db, root = embedded
    write(root / "draft.md", "---\ndraft: true\n---\n# Draft\n")
    indexing.index(db)
    browser = page(Index(db))
    link = re.search(r'<a href="([^"]+)"[^>]*>true <span class="n">1</span>', browser.get("/ui/?c=notes").text)
    assert ids(browser.get(html.unescape(link[1]))) == ["notes/draft.md"]


def test_fields_of_any_name_filter(embedded):
    db, root = embedded
    write(root / "cited.md", "---\njournal-ref: Nature 12\n---\n# Cited\n")
    indexing.index(db)
    browser = page(Index(db))
    link = re.search(r'<a href="([^"]+)"[^>]*>Nature 12 <span', browser.get("/ui/?c=notes").text)
    assert ids(browser.get(html.unescape(link[1]))) == ["notes/cited.md"]


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


@pytest.mark.parametrize("path", ["robots", "/robots/", "robots//"])
def test_a_folder_is_a_folder_however_written(embedded, path):
    db, root = embedded
    write(root / "robots-old.md", "# Old robots\n")  # what "robots*" matched too
    indexing.index(db)
    response = page(Index(db)).get("/ui/", params={"c": "notes", "path": path})
    assert ids(response) == [BRUSH]
    assert ">/</a>" not in response.text  # no folder without a name


def test_folders_with_brackets_in_their_names(embedded):
    # Folder names went into a glob, where [old] matches one letter: o, l or d.
    db, root = embedded
    write(root / "drafts [old]" / "a.md", "# Old draft\n")
    write(root / "drafts o" / "b.md", "# Other\n")
    indexing.index(db)
    browser = page(Index(db))
    folder = re.search(r'<a href="([^"]+)"[^>]*>drafts \[old\]/</a>', browser.get("/ui/", params={"c": "notes"}).text)
    assert ids(browser.get(html.unescape(folder[1]))) == ["notes/drafts [old]/a.md"]


@pytest.mark.parametrize("k", ["²", "-3", "0", "many", "1e3", ""])
def test_odd_page_sizes_show_a_page(browser, k):
    response = browser.get("/ui/", params={"c": "notes", "k": k})
    assert response.status_code == 200
    assert len(ids(response)) == 3


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


def search_box(text):
    """The search box a part of the page carries, if any."""
    found = re.search(r'<input id="q"[^>]*>', text)
    return found and found[0]


def test_links_that_change_the_query_update_the_search_box(browser):
    # Similar leaves the query behind: the box showed it still, and the next change of mode brought it back.
    asked = {"HX-Request": "true", "HX-Target": "explore"}
    by_link = search_box(browser.get("/ui/", params={"similar": "notes/rl.md"}, headers=asked).text)
    assert 'value=""' in by_link
    assert 'hx-swap-oob="true"' in by_link
    typed = browser.get("/ui/", params={"q": "brus"}, headers=asked | {"HX-Trigger": "search"})
    assert search_box(typed.text) is None  # what is being typed stays as it is


def test_the_form_keeps_a_list_of_similar_documents(browser):
    # Sorting "papers like this one" by date went through the form, which knew nothing of them.
    assert '<input type="hidden" name="similar" value="notes/rl.md">' in browser.get("/ui/?similar=notes/rl.md").text
    typed = browser.get("/ui/", params={"similar": "notes/rl.md", "q": "brush"})  # a search typed there
    assert ids(typed) == [BRUSH]
    assert 'name="similar"' not in typed.text


def test_the_title_follows_the_search(browser):
    explore = browser.get("/ui/", params={"q": "brush"}, headers={"HX-Request": "true", "HX-Target": "explore"})
    assert "<title>brush · sift</title>" in explore.text


def test_the_facets_that_narrow_most_come_first(embedded):
    # A field most results share, such as a category, narrows them more than one whose values differ.
    db, root = embedded
    for n in range(3):
        write(root / "memos" / f"{n}.md", f"---\nauthor: A{n}\nkind: memo\n---\n# Memo {n}\n")
    indexing.index(db)
    side = page(Index(db)).get("/ui/", params={"c": "notes"}).text
    assert re.findall(r'<section class="facet">\s*<h2>(\w+)</h2>', side) == ["kind", "author", "categories", "year"]


@pytest.fixture
def log(embedded, monkeypatch):
    """A document of 100 lines, in a reader that shows 10 at a time."""
    monkeypatch.setattr(ui, "READER_LINES", 10)
    db, root = embedded
    write(root / "log.txt", "".join(f"entry {n}\n" for n in range(1, 101)))
    indexing.index(db)
    return page(Index(db))


def shown(text):
    return [int(n) for n in re.findall(r'id="L(\d+)"', text)]


def test_a_long_document_shows_in_parts(log):
    # A log of a million lines made a page of hundreds of megabytes, and the server took seconds to make it.
    reader = log.get("/ui/", params={"doc": "notes/log.txt"}).text
    assert shown(reader) == list(range(1, 11))
    assert 'class="earlier"' not in reader
    later = log.get(html.unescape(re.search(r'<a class="later" href="([^"]+)"', reader)[1])).text
    assert shown(later) == list(range(11, 21))
    assert "counter-reset: line 10" in later  # the numbers in the margin count on


def test_a_long_document_opens_at_its_hit(log):
    reader = log.get("/ui/", params={"doc": "notes/log.txt", "lines": "50-51"}).text
    assert shown(reader) == list(range(49, 59))  # from a line before the hit
    earlier = log.get(html.unescape(re.search(r'<a class="earlier" href="([^"]+)"', reader)[1])).text
    assert shown(earlier) == list(range(39, 49))
    end = log.get("/ui/", params={"doc": "notes/log.txt", "at": "95"}).text
    assert shown(end) == list(range(95, 101))
    assert 'class="later"' not in end


def test_the_browser_keeps_no_copies_of_documents(browser):
    # htmx kept the last pages in localStorage, where the text of private documents stayed on disk; going back asks
    # the server instead. Nor may it run scripts in what it swaps in, whatever slips through.
    config = re.search(r"<meta name=\"htmx-config\" content='([^']+)'>", browser.get("/ui/").text)
    assert json.loads(config[1]) == {"historyCacheSize": 0, "allowEval": False, "allowScriptTags": False}


def test_going_back_gets_the_whole_page(browser):
    headers = {"HX-Request": "true", "HX-Target": "explore", "HX-History-Restore-Request": "true"}
    assert "<html" in browser.get("/ui/", params={"q": "brush"}, headers=headers).text


@pytest.mark.parametrize(
    "params, message",
    [
        ({"q": "brush", "f": '"=1'}, "invalid filter field"),
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


def test_files_open_without_running_their_scripts(embedded):
    # A saved web page in a collection would otherwise run its scripts as the UI, free to read the whole index.
    db, root = embedded
    write(root / "saved.html", "<html><body><h1>Saved</h1><p>text</p><script>fetch('/ui/?q=a')</script></body></html>")
    indexing.index(db)
    response = page(Index(db)).get("/ui/file", params={"id": "notes/saved.html"})
    assert response.status_code == 200
    assert response.headers["content-security-policy"] == "sandbox"
    assert response.headers["x-content-type-options"] == "nosniff"


def test_a_pdf_opens_in_the_browsers_viewer(embedded):
    # Chromium's PDF viewer refuses sandboxed documents; it runs no scripts of the PDF as the page.
    db, root = embedded
    (root / "paper.pdf").write_bytes((SAMPLES / "sample.pdf").read_bytes())
    indexing.index(db)
    response = page(Index(db)).get("/ui/file", params={"id": "notes/paper.pdf"})
    assert response.headers["content-type"] == "application/pdf"
    assert "content-security-policy" not in response.headers
    assert response.headers["x-content-type-options"] == "nosniff"


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
        with pytest.raises(SystemExit, match=f"what answers on port {port} is no sift with a UI"):
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
