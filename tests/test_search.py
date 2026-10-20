import numpy as np
import pytest

from sift import embed, indexing, search, store
from sift.search import Index, _any_word

from .conftest import PretendGpu, write


def ids(hits):
    return [h["id"] for h in hits]


# --- keyword search ----------------------------------------------------------


def test_keyword_search(corpus):
    index = Index(corpus[0])
    assert ids(index.search("reinforcement")) == ["notes/rl.md"]
    assert ids(index.search('"brush strokes"')) == ["notes/robots/brush.md"]
    assert ids(index.search("Kubelka-Munk")) == ["notes/robots/brush.md"]  # not valid FTS5, falls back to phrases
    assert ids(index.search("title: shaping")) == ["notes/rl.md"]  # column filter
    assert index.search("nonexistentword") == []


@pytest.mark.parametrize("mode", ["search", "vsearch", "query"])
@pytest.mark.parametrize("query", ["", "  \n "])
def test_blank_queries_find_nothing(embedded, mode, query):
    # The embedding of nothing is close to arbitrary chunks; returning those would look like an answer.
    assert getattr(Index(embedded[0]), mode)(query) == []


def test_unusable_query_is_a_clear_error(corpus, monkeypatch):
    monkeypatch.setattr(search, "_phrases", lambda q: "(")
    with pytest.raises(ValueError, match="invalid search query"):
        Index(corpus[0]).search("NEAR(")


@pytest.mark.parametrize("k", [0, -1, "3", 1.5])
def test_k_must_be_positive_integer(corpus, k):
    with pytest.raises(ValueError, match="k must be"):
        Index(corpus[0]).search("reward", k=k)


def test_hit_fields(corpus):
    hit = Index(corpus[0]).search("reward")[0]
    assert hit == {
        "id": "notes/rl.md",
        "title": "Reward shaping",
        "lines": "1-3",
        "score": hit["score"],
        "text": "# Reward shaping\n\nReward shaping speeds up reinforcement learning.",
    }
    assert hit["score"] > 0


def test_filters(corpus):
    index = Index(corpus[0])
    assert ids(index.search("brush", filters={"categories": "cs.GR"})) == ["notes/robots/brush.md"]
    assert index.search("brush", filters={"categories": "cs.LG"}) == []
    assert ids(index.search("brush", filters={"year": 2024})) == ["notes/robots/brush.md"]
    assert ids(index.search("reward OR brush", path="robots/*")) == ["notes/robots/brush.md"]
    assert ids(index.search("reward", collection="notes")) == ["notes/rl.md"]


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"filters": {"bad key": 1}}, ValueError),
        ({"filters": {"categories": ["cs.RO"]}}, ValueError),
        ({"filters": {"categories": None}}, ValueError),
        ({"collection": "missing"}, LookupError),
    ],
)
def test_bad_filters_are_clear_errors(corpus, kwargs, error):
    with pytest.raises(error):
        Index(corpus[0]).search("brush", **kwargs)


def test_nan_in_front_matter_does_not_break_filters(corpus):
    db, root = corpus
    write(root / "odd.md", "---\nscore: NaN\nlimit: Infinity\n---\n# Odd\nbrush\n")
    indexing.index(db)
    assert ids(Index(db).search("brush", filters={"categories": "cs.RO"})) == ["notes/robots/brush.md"]
    assert ids(Index(db).search("brush", filters={"score": "NaN"})) == ["notes/odd.md"]


# --- semantic and hybrid search ----------------------------------------------


def test_vsearch_needs_embeddings(corpus, fake_models):
    with pytest.raises(LookupError, match="sift embed"):
        Index(corpus[0]).vsearch("anything")


def test_vsearch(embedded):
    index = Index(embedded[0])
    assert ids(index.vsearch("painting robot brush strokes", k=1)) == ["notes/robots/brush.md"]
    assert ids(index.vsearch("reward shaping", k=1, path="*.md")) == ["notes/rl.md"]
    assert ids(index.vsearch("anything", filters={"categories": "cs.RO"})) == ["notes/robots/brush.md"]
    assert index.vsearch("anything", filters={"categories": "none"}) == []
    assert len(index.vsearch("anything", k=100)) == 3


def test_broken_vectors_do_not_hide_the_others(embedded):
    # PyTorch ranks NaN above every number, so as many NaN vectors as results asked for left vsearch with none.
    db, _ = embedded
    nan = np.full(64, np.nan, np.float16).tobytes()
    db.execute("UPDATE vectors SET vec = ? WHERE chunk IN (SELECT chunk FROM vectors ORDER BY chunk LIMIT 2)", (nan,))
    store.touch(db)
    db.commit()
    assert len(Index(db).vsearch("anything", k=2)) == 1  # the one intact vector of three


def test_vsearch_reloads_after_index_changes(embedded):
    db, root = embedded
    index = Index(db)
    assert ids(index.vsearch("reward shaping", k=1)) == ["notes/rl.md"]
    (root / "rl.md").unlink()
    indexing.index(db)  # deletes rl.md and its vector; the cached matrix must not serve it
    assert "notes/rl.md" not in ids(index.vsearch("reward shaping"))


def test_vsearch_after_interrupted_reset_asks_for_embedding(embedded, fake_models, monkeypatch):
    db, _ = embedded
    index = Index(db)
    index.vsearch("reward")  # vectors cached

    def interrupted(texts, query=False):
        raise KeyboardInterrupt

    monkeypatch.setattr(fake_models, "encode", interrupted)
    with pytest.raises(KeyboardInterrupt):
        embed.embed_pending(db, reset=True)
    with pytest.raises(LookupError, match="sift embed"):
        index.vsearch("reward")


class Constant:
    """Models that embed every text alike and score every pair alike: what search does with any models."""

    def encode(self, texts, query=False):
        return np.ones((len(texts), 64), np.float16)

    def rerank(self, query, texts):
        return np.arange(len(texts), dtype=np.float32)  # the later a candidate, the better

    def vectors(self, ids, matrix):
        from sift.vectors import InRam

        return InRam(ids, matrix)


def test_search_uses_the_models_it_is_given(embedded):
    index = Index(embedded[0], models=Constant())
    assert len(index.vsearch("anything")) == 3
    hits = index.query("anything")
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)


def test_hits_skip_chunks_removed_meanwhile(corpus):
    assert Index(corpus[0])._hits([(10**9, 1.0)]) == []


def test_query(embedded):
    hits = Index(embedded[0]).query("how does a painting robot plan brush strokes", k=2)
    assert ids(hits)[0] == "notes/robots/brush.md"
    assert len(hits) == 2
    assert hits[0]["score"] >= hits[1]["score"]


def test_query_with_only_stopwords_uses_semantic_candidates(embedded):
    assert len(Index(embedded[0]).query("what is the", k=5)) == 3


def test_query_without_matches(embedded):
    assert Index(embedded[0]).query("brush", filters={"categories": "none"}) == []


def test_any_word_drops_stopwords_and_duplicates():
    assert _any_word("How do robots mix the pigments, robots?") == '"robots" OR "mix" OR "pigments"'
    assert _any_word("the of a") == ""


# --- documents and status ----------------------------------------------------


def test_get(corpus):
    index = Index(corpus[0])
    assert index.get("notes/rl.md").startswith("# Reward shaping")
    for bad in ("notes/../outside.md", "missing/rl.md", "notes", "notes/"):
        with pytest.raises(LookupError, match="unknown document"):
            index.get(bad)


def test_get_a_folder_is_no_document(corpus):
    with pytest.raises(LookupError, match="unknown document"):
        Index(corpus[0]).get("notes/robots")


def test_status(embedded):
    status = Index(embedded[0]).status()
    assert status["index"].endswith("sift.db")
    assert status["model"]
    fields = ["categories", "title", "year"]
    path = status["collections"][0]["path"]
    assert status["collections"] == [
        {"name": "notes", "path": path, "docs": 3, "chunks": 3, "embedded": 3, "fields": fields}
    ]


def test_vectors_reload_while_on_the_gpu(embedded, make_local):
    db, root = embedded
    index = Index(db, models=make_local(accelerator=PretendGpu))
    assert "notes/rl.md" in ids(index.vsearch("reward shaping"))
    (root / "rl.md").unlink()
    indexing.index(db)
    assert "notes/rl.md" not in ids(index.vsearch("reward shaping"))


# --- edge cases ----------------------------------------------------------------


@pytest.mark.parametrize("mode", ["search", "vsearch", "query"])
def test_huge_k_is_fine(embedded, mode):
    # SQLite takes 64-bit integers only: a larger k raised OverflowError, which reached agents as a crash.
    assert ids(getattr(Index(embedded[0]), mode)("reward", k=10**30))[0] == "notes/rl.md"


def test_huge_numbers_in_filters_are_clear_errors(corpus):
    with pytest.raises(ValueError, match="too large"):
        Index(corpus[0]).search("brush", filters={"year": 10**30})


def test_query_returns_k_hits_beyond_its_candidate_count(embedded):
    # Candidates came from a fixed pool of 40 per mode, so k=100 could return fewer hits than there are.
    assert len(Index(embedded[0]).query("brush", k=3, candidates=1)) == 3


def test_get_serves_indexed_documents_only(corpus):
    # Ids come from search results; `get` must not read other files under a collection's folder.
    index = Index(corpus[0])
    for other in ("notes/ignored.py", "notes/.hidden/secret.md", "notes/node_modules/dep.md"):
        with pytest.raises(LookupError, match="unknown document"):
            index.get(other)


def test_get_reads_utf16_documents(corpus):
    db, root = corpus
    (root / "utf16.txt").write_bytes("\ufeff# Héllo\r\nworld\r\n".encode("utf-16-le"))
    indexing.index(db)
    assert Index(db).get("notes/utf16.txt") == "# Héllo\nworld\n"


def test_get_returns_lines_as_indexed(corpus):
    # \n line endings, as in hits, so that `get(id, hit["lines"])` gives the hit's text
    db, root = corpus
    (root / "crlf.md").write_bytes(b"# One\r\n\r\ntwo\r\n")
    indexing.index(db)
    index = Index(db)
    assert index.get("notes/crlf.md") == "# One\n\ntwo\n"
    assert index.get("notes/crlf.md", "3") == "two\n"


# --- what hits carry: front-matter fields, compact hits ----------------------


def test_hits_carry_requested_front_matter_fields(corpus):
    # Agents read dates, links and ids from these instead of guessing them from file paths.
    hit = Index(corpus[0]).search("brush", fields=["year", "categories", "missing"])[0]
    assert hit["fields"] == {"year": 2024, "categories": ["cs.RO", "cs.GR"]}  # absent fields are left out


def test_all_front_matter_fields(corpus):
    hit = Index(corpus[0]).search("brush", fields=["*"])[0]
    assert hit["fields"] == {"categories": ["cs.RO", "cs.GR"], "year": 2024}  # the title is a hit field already


def test_fields_of_documents_without_front_matter(corpus):
    assert Index(corpus[0]).search("reward", fields=["year"])[0]["fields"] == {}


def test_no_fields_unless_asked(corpus):
    assert "fields" not in Index(corpus[0]).search("brush")[0]


@pytest.mark.parametrize("fields", ["year", [1], [""], [None]])
def test_bad_fields_are_clear_errors(corpus, fields):
    with pytest.raises(ValueError, match="fields"):
        Index(corpus[0]).search("brush", fields=fields)


@pytest.mark.parametrize("mode", ["search", "vsearch", "query"])
def test_compact_hits_leave_out_the_text(embedded, mode):
    # Ten chunks of text per call fill an agent's context; to survey, ids, titles and scores are enough,
    # and `get` reads the documents that matter.
    hits = getattr(Index(embedded[0]), mode)("brush strokes", compact=True, fields=["year"])
    assert hits
    assert all(set(h) == {"id", "title", "lines", "score", "fields"} for h in hits)


def test_compact_query_ranks_like_the_full_one(embedded, fake_models):
    index = Index(embedded[0])
    full, compact = index.query("painting robot", k=3), index.query("painting robot", k=3, compact=True)
    assert [(h["id"], h["score"]) for h in compact] == [(h["id"], h["score"]) for h in full]  # reranked on the text


# --- range filters -------------------------------------------------------------

DAYS = ["2023-12-31", "2024-01-01", "2024-06-15", "2024-12-31", "2025-01-01"]


@pytest.fixture
def dated(embedded, fake_models):
    db, root = embedded
    for day in DAYS:
        write(root / "dated" / f"{day}.md", f'---\ncreated: "{day}"\n---\n# Paper of {day}\npigment\n')
    indexing.index(db)
    embed.embed_pending(db)
    return Index(db)


def days(hits):
    return sorted(h["id"].rsplit("/", 1)[1].removesuffix(".md") for h in hits)


@pytest.mark.parametrize(
    "condition, expected",
    [
        ({">=": "2024"}, DAYS[1:]),
        ({"<": "2024"}, DAYS[:1]),
        ({"<=": "2024"}, DAYS[:4]),  # dates compare at the precision given: all of 2024
        ({">": "2024"}, DAYS[4:]),  # after 2024, not after its first day
        ({">=": "2024-06", "<": "2025"}, DAYS[2:4]),
        ({">": "2024-06-15"}, DAYS[3:]),
        ({"=": "2024-06-15"}, DAYS[2:3]),
        ({">=": 2024, "<": 2025}, DAYS[1:4]),  # a number compares with the number a date starts with
    ],
)
def test_range_filters(dated, condition, expected):
    assert days(dated.search("pigment", k=20, filters={"created": condition})) == expected


@pytest.mark.parametrize("mode", ["search", "vsearch", "query"])
def test_range_filters_in_every_mode(dated, mode):
    hits = getattr(dated, mode)("pigment", k=20, filters={"created": {">=": "2024-06", "<": "2025"}})
    assert days(hits) == DAYS[2:4]


def test_range_filters_on_numbers(corpus):
    index = Index(corpus[0])
    assert ids(index.search("brush", filters={"year": {">": 2023, "<=": 2024}})) == ["notes/robots/brush.md"]
    assert index.search("brush", filters={"year": {">": 2024}}) == []
    assert ids(index.search("brush", filters={"year": {"<": "2025"}})) == ["notes/robots/brush.md"]  # given as text


def test_range_on_a_list_needs_one_element_in_range(corpus):
    db, root = corpus
    write(root / "multi.md", "---\nyears: [2019, 2030]\n---\n# Multi\nbrush\n")
    indexing.index(db)
    assert Index(db).search("brush", filters={"years": {">": 2020, "<": 2025}}) == []


@pytest.mark.parametrize("condition", [{}, {"=>": "2024"}, {">": ["2024"]}, {">": None}, {">": 10**30}])
def test_bad_range_filters_are_clear_errors(corpus, condition):
    with pytest.raises(ValueError, match="'created'"):
        Index(corpus[0]).search("x", filters={"created": condition})


# --- keyword search falls back to any term ---------------------------------------


@pytest.mark.parametrize(
    "query, expected",
    [
        ('"brush strokes" quantum', ["notes/robots/brush.md"]),  # a phrase and one more word
        ("reward pigment", ["notes/rl.md", "notes/robots/brush.md"]),
        ("brus* quantum", ["notes/robots/brush.md"]),  # prefixes stay prefixes
        ("Kubelka-Munk quantum", ["notes/robots/brush.md"]),  # not FTS5 syntax
        ('"brush quantum', ["notes/robots/brush.md"]),  # an unbalanced quote
    ],
)
def test_search_falls_back_to_any_term(corpus, query, expected):
    # Every term is required, so one word too many returned nothing at all. With no document holding all
    # terms, those holding any of them are better than nothing; BM25 ranks those with more of them first.
    assert sorted(ids(Index(corpus[0]).search(query))) == expected


@pytest.mark.parametrize("query", ["brush AND quantum", '"brush strokes" NOT robot', "quantum", '"no such phrase"'])
def test_no_fallback_for_explicit_operators_or_one_term(corpus, query):
    assert Index(corpus[0]).search(query) == []


def test_fallback_keeps_the_filters(corpus):
    assert Index(corpus[0]).search("reward pigment", filters={"year": 2024}) == Index(corpus[0]).search(
        "pigment", filters={"year": 2024}
    )


# --- reading part of a document ------------------------------------------------


@pytest.mark.parametrize(
    "lines, text",
    [
        ("1", "# Reward shaping\n"),
        ("2-3", "\nReward shaping speeds up reinforcement learning.\n"),
        ("3-99", "Reward shaping speeds up reinforcement learning.\n"),
        ("9", ""),
        (1, "# Reward shaping\n"),  # a line number, as agents also send it
    ],
)
def test_get_lines(corpus, lines, text):
    assert Index(corpus[0]).get("notes/rl.md", lines) == text


def test_get_the_lines_of_a_hit(corpus):
    # A hit's `lines` count from the top of the file, front matter included, so `get` returns the chunk.
    index = Index(corpus[0])
    hit = index.search("Kubelka")[0]
    assert index.get(hit["id"], hit["lines"]).strip() == hit["text"]


@pytest.mark.parametrize("lines", ["0", "3-2", "a", "1-", "-3", " 1", 0, -1, True, 1.5])
def test_bad_line_ranges_are_clear_errors(corpus, lines):
    with pytest.raises(ValueError, match="lines"):
        Index(corpus[0]).get("notes/rl.md", lines)


@pytest.mark.parametrize("short", ["notes/brush.md", "notes/robots/brush.md"])
def test_get_finds_a_document_by_its_file_name(corpus, short):
    # Agents shorten ids, e.g. arxiv/2603.29315.md for arxiv/2026/03/2603.29315.md; that failed.
    index = Index(corpus[0])
    assert index.get(short) == index.get("notes/robots/brush.md")
    assert index.get(short, "1") == "---\n"


def test_get_by_a_file_name_several_documents_have(corpus):
    db, root = corpus
    for folder in ("a", "b"):
        write(root / folder / "same.md", f"# {folder}\n")
    indexing.index(db)
    with pytest.raises(LookupError, match=r"'notes/same\.md' matches notes/a/same\.md, notes/b/same\.md;"):
        Index(db).get("notes/same.md")


@pytest.mark.parametrize(
    "bad", ["notes/rush.md", "notes/s/brush.md", "missing/brush.md", "notes/%.md", "notes/_rush.md"]
)
def test_get_by_file_name_needs_the_whole_name(corpus, bad):
    with pytest.raises(LookupError, match="unknown document"):
        Index(corpus[0]).get(bad)


def test_get_by_a_file_name_many_documents_have(corpus):
    db, root = corpus
    for folder in "abcdef":
        write(root / folder / "same.md", f"# {folder}\n")
    indexing.index(db)
    with pytest.raises(LookupError, match=r"notes/e/same\.md, \.\.\.; give the whole id"):
        Index(db).get("notes/same.md")
