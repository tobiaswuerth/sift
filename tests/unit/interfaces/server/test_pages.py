import http.client
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlencode

import pytest

from sift.api.collections import AddCollection
from sift.api.embedding import EmbedCollections
from sift.api.errors import Problem, StoreUnavailableError
from sift.api.notices import Notice
from sift.api.search import SearchResult
from sift.app.collections import add
from sift.app.embedding import embed
from sift.interfaces.acts.acting import Acting
from sift.interfaces.server.handled import HANDLED, Handled
from sift.interfaces.server.serving import HomeServer
from sift.settings import Settings
from tests.damage import damage_table
from tests.files import write
from tests.homes import indexed, quiet
from tests.models import Hashed
from tests.serving import counted, running

type Got = tuple[int, Mapping[str, str], str]


def got(server: HomeServer, path: str, headers: Mapping[str, str] | None = None) -> Got:
    connection = http.client.HTTPConnection("127.0.0.1", server.endpoint.port, timeout=60)
    try:
        connection.request("GET", path, headers=dict(headers or {}))
        response = connection.getresponse()
        headers = {key.lower(): value for key, value in response.getheaders()}
        return response.status, headers, response.read().decode(errors="replace")
    finally:
        connection.close()


def page(server: HomeServer, **given: str) -> str:
    status, headers, text = got(server, "/ui?" + urlencode(given))
    assert (status, headers["content-type"]) == (200, "text/html; charset=utf-8"), text
    return text


@pytest.fixture
def home(settings: Settings, notes: Path, tmp_path: Path) -> Settings:
    write(notes / "2024" / "odd.md", "# Odd <b>\n\ntitle: x\n\nA <script>alert(1)</script> dark sky.\n")
    write(notes / "tagged.md", "---\ntag: [sky, sea]\nurl: https://example.com/a\n---\n# Tagged\n\nThe dark sea.\n")
    add(settings.home, AddCollection(str(notes)))
    write(tmp_path / "hidden" / "secret.md", "# Secret\n\nDark matter.\n")
    add(settings.home, AddCollection(str(tmp_path / "hidden"), agents=False))
    indexed(settings.home)
    return settings


def test_the_page_shows_the_collections_with_their_counts_and_no_script_may_run(home: Settings) -> None:
    with running(home) as server:
        status, headers, text = got(server, "/ui")
        leading = got(server, "/")
    assert status == 200
    assert "default-src 'none'" in headers["content-security-policy"]
    assert (headers["cache-control"], "<script" in text) == ("no-store", False)
    assert 'href="/ui?c=hidden">hidden</a><span class="count">1</span>' in text
    assert 'href="/ui?c=notes">notes</a><span class="count">4</span>' in text
    assert (leading[0], leading[1]["location"]) == (307, "/ui")


def test_a_search_lists_its_hits_with_its_words_marked_and_documents_escaped(home: Settings) -> None:
    with running(home) as server:
        text = page(server, q="dark")
    assert 'href="/ui?q=dark&amp;doc=notes%2F2024%2Fodd.md&amp;lines=1-5#l1">Odd &lt;b&gt;</a>' in text
    assert "<mark>dark</mark>" in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "Odd &lt;b&gt;" in text
    assert "<script>" not in text


def test_facets_count_the_values_of_the_hits_and_filter_by_them(home: Settings) -> None:
    with running(home) as server:
        listed = page(server, q="dark")
        narrowed = page(server, q="dark", f="tag=sea")
    assert 'href="/ui?q=dark&amp;f=tag%3Dsea">sea</a><span class="count">1</span>' in listed
    assert "notes/tagged.md" in narrowed
    assert "notes/2024/odd.md" not in narrowed
    assert 'class="chip" href="/ui?q=dark">tag: sea' in narrowed


def test_browsing_a_collection_shows_its_folders_and_documents(home: Settings) -> None:
    with running(home) as server:
        top = page(server, c="notes")
        inside = page(server, c="notes", path="2024")
    assert 'href="/ui?c=notes&amp;path=2024%2F">2024/</a>' in top
    assert "notes/brush.md" in top
    assert ("notes/2024/odd.md" in inside, "notes/brush.md" in inside) == (True, False)
    assert (
        '<p class="heading"><a href="/ui?c=notes">notes</a>/<a href="/ui?c=notes&amp;path=2024%2F">2024</a>/</p>'
        in inside
    )


def test_a_document_opens_in_the_reader_with_its_hit_marked_and_its_fields_as_filters(home: Settings) -> None:
    with running(home) as server:
        text = page(server, q="sea", doc="notes/tagged.md", lines="7-7")
    assert '<div class="hit" id="l7"><span class="n">7</span>The dark <mark>sea</mark>.</div>' in text
    assert '<a href="/ui?q=sea&amp;f=tag%3Dsky&amp;doc=notes%2Ftagged.md&amp;lines=7-7">sky</a>' in text
    assert '<a href="https://example.com/a" rel="noopener noreferrer">https://example.com/a</a>' in text
    assert 'href="/ui/file?id=notes%2Ftagged.md"' in text


def test_a_document_whose_last_line_has_no_line_break_shows_it_as_its_last_line(home: Settings, notes: Path) -> None:
    write(notes / "plain.txt", "First line.\nLast line, unbroken.")
    indexed(home.home)
    with running(home) as server:
        text = page(server, doc="notes/plain.txt")
    assert '<span class="n">2</span>Last line, unbroken.</div>' in text
    assert '<span class="n">3</span>' not in text


def test_errors_show_in_place(home: Settings) -> None:
    with running(home) as server:
        unknown = page(server, q="x", c="nope")
        missing = page(server, doc="notes/none.md")
    assert '<div class="error" role="alert"><p>no collection is named &#x27;nope&#x27;</p>' in unknown
    assert "no note or document &#x27;notes/none.md&#x27;" in missing


def test_a_bug_is_no_error_to_show_in_place(home: Settings) -> None:
    def broken(_acting: Acting, _data: object) -> object:
        raise RuntimeError("a bug")

    with running(home, {**HANDLED, "status": Handled(broken, writes=False)}) as server:
        status, _, _ = got(server, "/ui")
    assert status == 500


def test_the_original_file_opens_without_running_its_scripts(home: Settings, notes: Path) -> None:
    with running(home) as server:
        status, headers, text = got(server, "/ui/file?id=notes%2Fhalo.md")
        missing = got(server, "/ui/file?id=notes%2F..%2F..%2Fsecret")
    assert (status, text) == (200, (notes / "halo.md").read_bytes().decode())
    assert headers["content-security-policy"] == "sandbox; frame-ancestors 'none'"
    assert headers["x-content-type-options"] == "nosniff"
    assert missing[0] == 404


def test_the_ui_answers_local_hosts_alone(home: Settings) -> None:
    with running(home) as server:
        assert got(server, "/ui", {"Host": "attacker.example"})[0] == 400
        assert got(server, "/ui", {"Host": f"localhost:{server.endpoint.port}"})[0] == 200


def test_similar_documents_backlinks_and_links_are_one_click_away(home: Settings, notes: Path) -> None:
    write(notes / "links" / "a.md", "# A\n\nSee [[b]] and [[c]].\n")
    write(notes / "links" / "b.md", "# B\n\nBack to [[a]].\n")
    indexed(home.home)
    embed(home.home, EmbedCollections(), quiet, Hashed())
    with running(home) as server:
        read = page(server, doc="notes/links/a.md")
        linking = page(server, to="notes/links/a.md")
        linked = page(server, by="notes/links/a.md")
        alike = page(server, similar="notes/halo.md")
    backlinks = 'href="/ui?to=notes%2Flinks%2Fa.md&amp;doc=notes%2Flinks%2Fa.md">Backlinks</a>'
    assert (backlinks in read, "Links</a>" in read) == (True, True)
    assert (
        "Linking to A" in linking,
        'href="/ui?to=notes%2Flinks%2Fa.md&amp;doc=notes%2Flinks%2Fb.md">B</a>' in linking,
    ) == (True, True)
    assert ("Linked from A" in linked, ">B</a>" in linked, "<li>c</li>" in linked) == (True, True, True)
    assert "Like notes/halo.md" in alike
    assert 'class="chip" href="/ui">linking to notes/links/a.md' in linking


def test_a_search_within_a_collection_folder_and_order_keeps_them(home: Settings) -> None:
    with running(home) as server:
        text = page(server, q="dark", c="notes", path="2024", sort="-tag", to="", by="")
    assert ("notes/2024/odd.md" in text, "notes/tagged.md" in text) == (True, False)
    assert '<input type="hidden" name="c" value="notes"><input type="hidden" name="path" value="2024/">' in text


def test_a_long_document_shows_in_parts(home: Settings, notes: Path) -> None:
    write(notes / "long.md", "# Long\n\n" + "".join(f"line {n}\n" for n in range(3, 4_101)))
    indexed(home.home)
    with running(home) as server:
        first = page(server, doc="notes/long.md")
        at_hit = page(server, doc="notes/long.md", lines="3000-3000")
        last = page(server, doc="notes/long.md", at="4001")
    assert ("lines 1 to 2,000 of 4,100" in first, ">Later</a>" in first, ">Earlier</a>" in first) == (True, True, False)
    assert 'class="hit" id="l3000"' in at_hit
    assert "lines 2,800 to 4,100 of 4,100" in at_hit
    assert ("lines 4,001 to 4,100 of 4,100" in last, ">Later</a>" in last) == (True, False)


def test_a_pdf_opens_in_the_browser_s_own_viewer(home: Settings, notes: Path) -> None:
    (notes / "sample.pdf").write_bytes((Path(__file__).parents[3] / "samples" / "sample.pdf").read_bytes())
    indexed(home.home)
    with running(home) as server:
        status, headers, _ = got(server, "/ui/file?id=notes%2Fsample.pdf")
        style = got(server, "/ui/style.css")
    assert (status, headers["content-type"]) == (200, "application/pdf")
    assert headers["content-security-policy"] == "frame-ancestors 'none'"
    assert (style[0], style[1]["content-type"].startswith("text/css")) == (200, True)


def test_what_a_pane_cannot_show_it_says_in_its_place(home: Settings) -> None:
    def refused(_acting: Acting, _data: object) -> object:
        raise StoreUnavailableError("the store is busy", "ask again in a moment")

    def noted(_acting: Acting, _data: object) -> object:
        unsearched = (Notice(Problem.BUSY, "its store is busy", None, "notes"),)
        return SearchResult((), 0, unsearched, scored_by="keywords")

    failing = {**HANDLED, "status": Handled(refused, writes=False), "get": Handled(refused, writes=False)}
    failing |= {"links": Handled(refused, writes=False), "search": Handled(noted, writes=False)}
    with running(home, failing) as server:
        text = page(server, q="x", doc="notes/halo.md")
    assert ("Collections" in text, '<p class="note">notes: its store is busy</p>' in text) == (False, True)
    assert '<article class="reader"><div class="error" role="alert"><p>the store is busy</p>' in text


def test_a_long_front_matter_value_is_shown_as_text_and_one_of_many_values_no_facet(
    home: Settings, notes: Path
) -> None:
    for n in range(10):
        write(
            notes / "many" / f"{n}.md", f"---\nserial: s{n}\nabout: {'long words ' * 10}\n---\n# M{n}\n\nmany shades\n"
        )
    indexed(home.home)
    with running(home) as server:
        text = page(server, q="shades", doc="notes/many/1.md")
    assert ("<h3>serial</h3>" in text, "<h3>about</h3>" in text, "<h3>shades</h3>" in text) == (False, False, False)
    assert "<dd>long words long words" in text


def test_a_search_among_a_note_s_backlinks_keeps_the_note(home: Settings, notes: Path) -> None:
    write(notes / "links" / "a.md", "# A\n\nThe dark [[b]].\n")
    write(notes / "links" / "b.md", "# B\n\nBack to [[a]], dark.\n")
    indexed(home.home)
    with running(home) as server:
        linking = page(server, q="dark", to="notes/links/b.md")
        linked = page(server, q="dark", by="notes/links/b.md")
    assert ("notes/links/a.md" in linking, "notes/halo.md" in linking) == (True, False)
    assert ("notes/links/a.md" in linked, "notes/halo.md" in linked) == (True, False)


def test_a_note_whose_links_cannot_be_read_offers_none(home: Settings) -> None:
    def refused(_acting: Acting, _data: object) -> object:
        raise StoreUnavailableError("the store is busy", "ask again in a moment")

    def noted(_acting: Acting, _data: object) -> object:
        return SearchResult(
            (), 0, (), left_out=(Notice(Problem.BUSY, "some were left out", None),), scored_by="keywords"
        )

    failing = {**HANDLED, "links": Handled(refused, writes=False), "search": Handled(noted, writes=False)}
    with running(home, failing) as server:
        text = page(server, q="x", doc="notes/halo.md")
    assert (">Similar</a>" in text, ">Backlinks</a>" in text) == (True, False)
    assert '<p class="note">some were left out</p>' in text


def test_no_more_is_offered_past_the_most_a_page_holds(home: Settings) -> None:
    def many(_acting: Acting, _data: object) -> object:
        return SearchResult((), 0, (), more=True, scored_by="keywords")

    with running(home, {**HANDLED, "search": Handled(many, writes=False)}) as server:
        texts = [page(server, q="x", k=k) for k in ("30", "500")]
    assert ['class="more"' in text for text in texts] == [True, False]


def test_a_damaged_store_stops_no_page_of_the_ui(home: Settings) -> None:
    damage_table(home.home / "collections" / "hidden.sqlite", "counts")
    with running(home) as server:
        overview = got(server, "/ui")
        found = page(server, q="dark", c="notes")
    assert (overview[0], "hidden.sqlite is damaged: " in overview[2]) == (200, True)
    assert 'href="/ui?c=notes">notes</a><span class="count">4</span>' in overview[2]
    assert "doc=notes%2Ftagged.md" in found


def test_no_page_of_another_site_loads_or_frames_what_the_ui_answers(home: Settings, notes: Path) -> None:
    (notes / "sample.pdf").write_bytes((Path(__file__).parents[3] / "samples" / "sample.pdf").read_bytes())
    indexed(home.home)
    asked = ("/ui", "/ui/style.css", "/ui/file?id=notes%2Fhalo.md", "/ui/file?id=notes%2Fsample.pdf", "/ui/file?id=no")
    with running(home) as server:
        answered = [got(server, path) for path in asked]
    assert [status for status, _, _ in answered] == [200, 200, 200, 200, 404]
    assert [headers["cross-origin-resource-policy"] for _, headers, _ in answered] == ["same-origin"] * 5
    framed = [headers.get("content-security-policy", "") for _, headers, _ in answered]
    assert ["frame-ancestors 'none'" in policy for policy in framed] == [True, False, True, True, False]


NAVIGATED = {"Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "document"}
ASKED_BY_BROWSERS: tuple[tuple[Mapping[str, str], bool], ...] = (
    ({}, True),
    ({"Sec-Fetch-Site": "none", **NAVIGATED}, True),
    ({"Sec-Fetch-Site": "same-origin", **NAVIGATED}, True),
    ({"Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "no-cors", "Sec-Fetch-Dest": "style"}, True),
    ({"Sec-Fetch-Site": "cross-site", **NAVIGATED}, True),
    ({"Sec-Fetch-Site": "same-site", **NAVIGATED}, True),
    ({"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "navigate", "Sec-Fetch-Dest": "iframe"}, False),
    ({"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "no-cors", "Sec-Fetch-Dest": "image"}, False),
    ({"Sec-Fetch-Site": "cross-site", "Sec-Fetch-Mode": "cors", "Sec-Fetch-Dest": "empty"}, False),
    ({"Sec-Fetch-Site": "same-site", "Sec-Fetch-Mode": "no-cors", "Sec-Fetch-Dest": "script"}, False),
    ({"Sec-Fetch-Site": "cross-site"}, False),
)


@pytest.mark.parametrize("headers, let", ASKED_BY_BROWSERS)
def test_a_page_of_another_site_reaches_the_ui_by_a_link_alone_and_makes_it_search_nothing(
    home: Settings, headers: Mapping[str, str], *, let: bool
) -> None:
    asked = list[str]()
    with running(home, counted(asked)) as server:
        answered = [got(server, path, headers)[0] for path in ("/ui?q=dark", "/ui/file?id=notes%2Fhalo.md", "/")]
        style = got(server, "/ui/style.css", headers)[0]
    assert (answered, style) == (([200, 200, 307], 200) if let else ([403, 403, 403], 403))
    assert ("search" in asked) is let
