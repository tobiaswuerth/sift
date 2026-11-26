import dataclasses

from sift.api.errors import Problem, StoreUnavailableError
from sift.api.links import LinkIn, LinkingLine, LinkOut, Links, Target, Unresolved
from sift.api.notices import Notice
from sift.api.search import Hit, SearchResult
from sift.interfaces.cli.shown.links import links as shown_links
from sift.interfaces.cli.shown.links import unresolved as shown_unresolved
from sift.interfaces.cli.shown.outcome import failure, notice_lines
from sift.interfaces.cli.shown.searching import hits
from sift.interfaces.console.style import PLAIN


def test_a_partial_search_says_better_matches_may_be_missing() -> None:
    found = SearchResult(
        (Hit("notes/a.md", "notes", "a.md", "A", 1, 2, 1.5, None),), 1, (), partial=True, scored_by="keywords"
    )
    dot = "  \N{MIDDLE DOT}  "
    said = dot.join(("1 result", "keyword", "partial: better matches may be missing", "1 collection", "250 ms"))
    assert hits(found, 0.25, "linux", PLAIN).messages == (said,)
    assert "partial" not in hits(dataclasses.replace(found, partial=False), 0.25, "linux", PLAIN).messages[0]


def test_what_a_search_left_out_of_a_collection_is_noted_with_its_remedy_before_the_count() -> None:
    hit = Hit("notes/a.md", "notes", "a.md", "A", 1, 2, 1.5, None)
    left = Notice(Problem.NOT_EMBEDDED, "3 of its passages hold no vector yet", "embed the collection", "notes")
    messages = hits(SearchResult((hit,), 1, (), left_out=(left,), scored_by="keywords"), 0.25, "linux", PLAIN).messages
    assert messages[:3] == (
        " note  notes: 3 of its passages hold no vector yet",
        "  fix  embed the collection",
        " next  sift embed notes",
    )
    assert messages[3].startswith("1 result")


def test_stores_ruled_out_by_the_links_are_said_apart_from_those_searched() -> None:
    dot = "  \N{MIDDLE DOT}  "
    found = SearchResult((), 1, (), ruled_out=2, scored_by="keywords")
    said = dot.join(("no matches", "keyword", "1 collection, 2 ruled out by the links", "250 ms"))
    assert hits(found, 0.25, "linux", PLAIN).messages == (said,)
    assert hits(dataclasses.replace(found, searched=0), 0.25, "linux", PLAIN).messages[0].startswith("no matches")


def test_a_hit_names_its_copies_in_other_collections_under_its_place() -> None:
    hit = Hit("notes/halo.md", "notes", "halo.md", "Halo", 1, 3, 3.25, None, ("more/copy.md", "papers/halo.md"))
    shown = hits(SearchResult((hit,), 3, (), scored_by="reranker"), 0.25, "linux", PLAIN)
    assert shown.results == (
        "1  3.25  Halo",
        "         notes/halo.md:1-3",
        "         also more/copy.md, papers/halo.md",
    )
    assert " hybrid " in shown.messages[-1]


def test_a_score_too_small_for_two_decimals_is_shown_in_full() -> None:
    found = SearchResult(
        (Hit("n/a.md", "n", "a.md", "A", 1, 1, 1.6e-06, None), Hit("n/b.md", "n", "b.md", "B", 1, 1, 0.0, None)),
        1,
        (),
        scored_by="keywords",
    )
    assert hits(found, 0.25, "linux", PLAIN).results[::2] == ("1  1.6e-06  A", "2     0.00  B")


def test_a_hit_shows_the_front_matter_asked_for_under_its_place() -> None:
    hit = Hit(
        "notes/halo.md",
        "notes",
        "halo.md",
        "Halo",
        1,
        3,
        3.25,
        None,
        fields={"created": "2024-06-15", "tags": ["a", "é"], "n": 2},
    )
    shown = hits(SearchResult((hit,), 1, (), scored_by="keywords"), 0.25, "linux", PLAIN)
    assert shown.results == (
        "1  3.25  Halo",
        "         notes/halo.md:1-3",
        '         created: 2024-06-15  tags: ["a", "é"]  n: 2',
    )
    assert (
        len(
            hits(
                SearchResult((dataclasses.replace(hit, fields={}),), 1, (), scored_by="keywords"), 0.25, "linux", PLAIN
            ).results
        )
        == 2
    )


def test_links_show_where_each_leads_and_the_lines_linking_in() -> None:
    found = Links(
        "v/Offer.md",
        "Offer",
        (
            LinkOut(3, "Hans", "org", "Team", "alias", "v/Hans.md", "Hans", (), None),
            LinkOut(4, "Dup", None, None, "ambiguous", None, None, ("v/A/Dup.md", "v/B/Dup.md"), None),
            LinkOut(5, "Gone", None, None, "unresolved", None, None, (), "v/Notes.md"),
        ),
        (LinkIn("v/Notes.md", "Notes", (LinkingLine(3, None, "[[Offer]] again"),), None),),
    )
    assert shown_links(found, "linux", PLAIN).results == (
        "Offer (v/Offer.md) links to:",
        "     3  org: Hans#Team  ->  v/Hans.md (alias)",
        "     4  Dup  ->  ambiguous: v/A/Dup.md, v/B/Dup.md",
        "     5  Gone  ->  unresolved  (through v/Notes.md)",
        "linking to Offer (v/Offer.md):",
        "  v/Notes.md  Notes",
        "     3  [[Offer]] again",
    )
    alone = Links("v/A.md", "A", (), None)
    assert shown_links(alone, "linux", PLAIN).results == ("A (v/A.md) links to: nothing",)


def test_unresolved_targets_show_how_many_link_them_and_a_few_of_those() -> None:
    found = Unresolved((Target("Missing", 7, ("v/a.md", "v/b.md")), Target("Gone", 1, ("v/c.md",))))
    assert shown_unresolved(found, "linux", PLAIN).results == ("7  Missing  v/a.md, v/b.md, ...", "1  Gone     v/c.md")
    assert shown_unresolved(Unresolved(()), "linux", PLAIN).messages == (" none  every link leads to a note",)


def test_unresolved_targets_of_a_store_that_could_not_be_read_are_said_to_be_missing() -> None:
    found = Unresolved(
        (Target("Gone", 1, ("v/c.md",)),),
        (Notice(Problem.STORE_DAMAGED, "the store is damaged", "remove the collection", "w"),),
    )
    assert shown_unresolved(found, "linux", PLAIN).messages[0] == "error  w: the store is damaged"
    alone = shown_unresolved(Unresolved((), found.unsearched), "linux", PLAIN)
    assert (alone.results, alone.messages[1:]) == ((), ("  fix  remove the collection", " next  sift rm w"))


def test_a_next_step_names_the_collection_or_is_left_out_where_it_needs_one_and_none_is_known() -> None:
    def steps(code: Problem, collection: str | None) -> tuple[str, ...]:
        return notice_lines((Notice(code, "said", None, collection),), PLAIN)[1:]

    assert steps(Problem.INACTIVE_COLLECTION, "notes") == (" next  sift activate notes",)
    assert steps(Problem.OTHER_MODEL, "notes") == (" next  sift embed --reset notes",)
    assert steps(Problem.KEYWORDS_REMADE, None) == (" next  sift index",)
    for code in (Problem.INACTIVE_COLLECTION, Problem.STORE_MISSING, Problem.STORE_DAMAGED, Problem.OTHER_MODEL):
        assert steps(code, None) == ()
    refused = failure(StoreUnavailableError("no vectors yet", "embed it", code=Problem.NOT_EMBEDDED), PLAIN)
    assert refused.messages == ("error  no vectors yet", "  fix  embed it", " next  sift embed")
    missing = failure(StoreUnavailableError("its store is gone", "remove it", code=Problem.STORE_MISSING), PLAIN)
    assert missing.messages == ("error  its store is gone", "  fix  remove it")
    assert failure(StoreUnavailableError("x", "y", code=Problem.CATALOG_DAMAGED), PLAIN).messages[2:] == ()


def test_links_of_a_note_found_while_a_store_could_not_be_read_say_so() -> None:
    found = Links(
        "v/A.md",
        "A",
        (),
        None,
        (Notice(Problem.STORE_DAMAGED, "the store is damaged", "remove the collection", "w"),),
    )
    shown_found = shown_links(found, "linux", PLAIN)
    assert shown_found.results == ("A (v/A.md) links to: nothing",)
    assert shown_found.messages[0] == "error  w: the store is damaged"


def test_a_search_with_more_results_than_shown_says_more_follow() -> None:
    hit = Hit("notes/a.md", "notes", "a.md", "A", 1, 2, 1.5, None)
    shown = hits(SearchResult((hit,), 1, (), more=True, scored_by="vectors"), 0.25, "linux", PLAIN)
    assert shown.messages[-1].startswith("1 result, more follow  ")
    assert " semantic " in shown.messages[-1]


def test_a_loosened_search_says_no_passage_held_every_word() -> None:
    hit = Hit("notes/a.md", "notes", "a.md", "A", 1, 2, 1.5, None)
    shown = hits(SearchResult((hit,), 1, (), loosened=True, scored_by="keywords"), 0.25, "linux", PLAIN)
    assert shown.messages[0] == " note  no passage holds every word of the query: these hold some of them"


def test_links_and_unresolved_targets_cut_at_k_say_more_follow() -> None:
    cut = Links("notes/a.md", "A", (), None, more=True)
    assert shown_links(cut, "linux", PLAIN).messages == (" more  more links follow: ask for more with -k",)
    targets = Unresolved((Target("Missing", 2, ("notes/a.md", "notes/c.md")),), more=True)
    assert shown_unresolved(targets, "linux", PLAIN).messages == (" more  more targets follow: ask for more with -k",)


def test_a_note_of_the_whole_search_is_said_without_a_collection() -> None:
    cut = Notice(Problem.TOO_MANY_WORDS, "the question holds 1,500 words: its keywords are the first 1,000", "ask less")
    shown = hits(SearchResult((), 1, (), left_out=(cut,), scored_by="reranker"), 0.25, "linux", PLAIN)
    assert shown.messages[:2] == (
        " note  the question holds 1,500 words: its keywords are the first 1,000",
        "  fix  ask less",
    )
