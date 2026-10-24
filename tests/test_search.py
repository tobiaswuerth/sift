"""Searching the index: keyword, semantic and hybrid search, filters and sorting, documents like one, `get`, and
browsing by folder."""

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


def test_a_document_about_a_term_outranks_one_that_names_it_in_its_title(corpus):
    # The title is indexed twice already, as the text's heading and as its own column; weighted double on top, a
    # title word outranked a whole abstract about it.
    db, root = corpus
    write(
        root / "named.md",
        "# A transformer for music\n\nWe compose melodies with a sequence model of notes and chords.\n",
    )
    write(
        root / "about.md",
        "# Attention models\n\nA transformer replaces recurrence with attention. Each transformer layer attends to "
        "every token, and stacked transformer blocks train in parallel.\n",
    )
    indexing.index(db)
    assert ids(Index(db).search("transformer", k=2)) == ["notes/about.md", "notes/named.md"]


@pytest.mark.parametrize("mode", ["search", "vsearch", "query"])
@pytest.mark.parametrize("query", ["", "  \n "])
def test_blank_queries_find_nothing(embedded, mode, query):
    # The embedding of nothing is close to arbitrary chunks; returning those would look like an answer.
    assert getattr(Index(embedded[0]), mode)(query) == []


@pytest.mark.parametrize("query", ["title: speeds learning", "{title meta}: speeds up", "meta:speeds learning"])
def test_a_column_filter_holds_when_nothing_matches(corpus, query):
    # When no passage matched, sift looked for any of the terms, and "title:" became a word like the others: passages
    # with "speeds" in their text came back for a search of titles.
    assert Index(corpus[0]).search(query) == []


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


BRUSH, RL = "notes/robots/brush.md", "notes/rl.md"


def test_a_list_of_values_means_any_of_them(corpus):
    index = Index(corpus[0])
    assert ids(index.search("brush OR reward", filters={"categories": ["cs.LG", "cs.GR"]})) == [BRUSH]
    assert index.search("brush", filters={"categories": ["cs.LG", "cs.CV"]}) == []
    assert ids(index.search("brush", filters={"year": {"=": [2023, 2024]}})) == [BRUSH]


@pytest.mark.parametrize(
    "condition, expected",
    [
        ({"not": "cs.GR"}, [RL]),  # a document without the field has none of its values
        ({"not": ["cs.LG", "cs.CV"]}, [RL, BRUSH]),
        ({"not": ["cs.LG", "cs.RO"]}, [RL]),
    ],
)
def test_not_filters(corpus, condition, expected):
    hits = Index(corpus[0]).search("brush OR reward", filters={"categories": condition})
    assert sorted(ids(hits)) == expected


def test_not_with_a_range(corpus):
    index = Index(corpus[0])
    assert ids(index.search("brush OR reward", filters={"year": {">=": 2020, "not": 2023}})) == [BRUSH]
    assert index.search("brush OR reward", filters={"year": {">=": 2020, "not": 2024}}) == []


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"filters": {'bad"key': 1}}, ValueError),
        ({"filters": {"": 1}}, ValueError),
        ({"filters": {"categories": []}}, ValueError),
        ({"filters": {"categories": [None]}}, ValueError),
        ({"filters": {"categories": None}}, ValueError),
        ({"filters": {"categories": {"not": []}}}, ValueError),
        ({"filters": {"categories": {"not": None}}}, ValueError),
        ({"collection": "missing"}, LookupError),
    ],
)
def test_bad_filters_are_clear_errors(corpus, kwargs, error):
    with pytest.raises(error):
        Index(corpus[0]).search("brush", **kwargs)


def test_filters_and_sort_take_any_field_name(corpus):
    # Front matter and JSON Lines name fields freely, e.g. "journal-ref" or "date created".
    db, root = corpus
    write(root / "cited.md", "---\njournal-ref: Nature 12\ndate created: 2024-05-01\n---\n# Cited\nbrush\n")
    indexing.index(db)
    index = Index(db)
    assert ids(index.search("brush", filters={"journal-ref": "Nature 12"})) == ["notes/cited.md"]
    assert ids(index.search("brush", filters={"date created": {">=": "2024-05"}})) == ["notes/cited.md"]
    assert ids(index.search("brush", filters={"journal-ref": {"not": "Nature 12"}})) == ["notes/robots/brush.md"]
    assert ids(index.search("brush", sort="-date created"))[0] == "notes/cited.md"
    assert ids(index.browse(sort="-date created"))[0] == "notes/cited.md"


def test_nan_in_front_matter_does_not_break_filters(corpus):
    db, root = corpus
    write(root / "odd.md", "---\nscore: NaN\nlimit: Infinity\n---\n# Odd\nbrush\n")
    indexing.index(db)
    assert ids(Index(db).search("brush", filters={"categories": "cs.RO"})) == ["notes/robots/brush.md"]
    assert ids(Index(db).search("brush", filters={"score": "NaN"})) == ["notes/odd.md"]


# --- front matter in keyword search ------------------------------------------

PAPER = (
    '---\ntitle: "Attention"\nauthors: ["Ashish Vaswani", "Noam Shazeer"]\nid: "1706.03762"\n---\n'
    "# Attention\n\nSequence transduction with attention only.\n"
)


@pytest.fixture
def paper(corpus):
    db, root = corpus
    write(root / "paper.md", PAPER)
    indexing.index(db)
    return Index(db)


@pytest.mark.parametrize("query", ["Vaswani", "1706.03762", "Shazeer attention"])
def test_keyword_search_finds_front_matter_values(paper, query):
    # Authors and ids live in front matter, which the keyword index left out: on 3.2 million arXiv abstracts,
    # "Vaswani transformer paper" missed "Attention Is All You Need", and searching an arXiv id found nothing.
    assert ids(paper.search(query)) == ["notes/paper.md"]


def test_a_changed_front_matter_value_is_no_longer_found(paper, corpus):
    db, root = corpus
    write(root / "paper.md", PAPER.replace("Ashish Vaswani", "Someone Else"))
    indexing.index(db)
    assert paper.search("Vaswani") == []
    assert ids(paper.search("Someone")) == ["notes/paper.md"]
    db.execute("INSERT INTO fts(fts) VALUES('integrity-check')")  # raises if deleting missed what inserting added


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


class Alike(Constant):
    """Constant models whose vectors are all alike, so every cosine with a query is the same."""

    def vectors(self, ids, matrix):
        return super().vectors(ids, np.ones_like(matrix))


def test_query_scores_are_the_rerankers_raw_scores_when_the_cosines_agree(embedded):
    # Through a sigmoid they ran from 0 to 1, but saturated: Ettin scores close matches 8 to 12, so every hit showed
    # 0.9999, and an off-topic query's best still 0.98. The raw scores tell strong hits from weak ones; the cosine
    # shifts them only where the candidates differ in meaning.
    hits = Index(embedded[0], models=Alike()).query("anything")
    assert [h["score"] for h in hits] == [2.0, 1.0, 0.0]


@pytest.fixture
def embedded_paper(embedded, fake_models):
    """The notes, a paper with authors, and an abstract that cites one of them."""
    db, root = embedded
    write(root / "paper.md", PAPER)
    write(root / "survey.md", "# Survey\n\nSince Vaswani et al., attention is everywhere.\n")
    indexing.index(db)
    embed.embed_pending(db)
    return Index(db)


def test_query_finds_a_paper_by_its_author(embedded_paper):
    # The reranker saw nothing of "Vaswani" in "Attention Is All You Need", only in abstracts that cite it: with the
    # author in the keyword index alone, the paper still ranked 34th among 3.2 million arXiv abstracts.
    assert ids(embedded_paper.query("Ashish Vaswani attention", k=1)) == ["notes/paper.md"]


class Recording(Constant):
    """Constant models that keep what the reranker is given."""

    def __init__(self):
        self.texts = []

    def rerank(self, query, texts):
        self.texts += texts
        return super().rerank(query, texts)


def test_the_reranker_reads_only_the_front_matter_values_a_query_names(embedded_paper):
    # All of them (links, licences, 30 authors) diluted the passage: topical queries ranked worse (known-item
    # evaluation, MRR 0.92 to 0.75), while values the query names lifted author and id queries (0.69 to 0.92 overall).
    models = Recording()
    Index(embedded_paper.db, models=models).query("Shazeer attention")
    assert "Attention\nNoam Shazeer\n\n# Attention\n\nSequence transduction with attention only." in models.texts
    models.texts.clear()
    Index(embedded_paper.db, models=models).query("sequence transduction")
    assert "Attention\n\n# Attention\n\nSequence transduction with attention only." in models.texts


# --- query: the reranker, and the passage's meaning ---------------------------------


def test_blend_is_the_rerankers_score_when_the_cosines_agree():
    rerank = np.array([3.0, 1.0, 2.0])
    assert search._blend(rerank, np.full(3, 0.8), np.zeros(3, bool)).tolist() == rerank.tolist()


def test_blend_lifts_a_passage_closer_in_meaning():
    # The reranker alone ranked scientific claims worse than the embedding did (BEIR SciFact nDCG@10 0.760 against
    # 0.840): the cosine has a say too.
    # The reranker prefers the second slightly (its third candidate it clearly dislikes); the first is far closer.
    blended = search._blend(np.array([1.0, 1.05, 0.0]), np.array([0.9, 0.5, 0.5]), np.zeros(3, bool))
    assert blended[0] > blended[1] > blended[2]


def test_blend_fits_any_rerankers_scale():
    # Rerankers score on scales of their own; the cosine counts in units of their spread, so one weight fits all.
    rerank, cosine, named = np.array([1.0, 1.1, 0.4]), np.array([0.9, 0.5, 0.7]), np.zeros(3, bool)
    assert np.allclose(search._blend(10 * rerank, cosine, named), 10 * search._blend(rerank, cosine, named))


def test_blend_leaves_front_matter_matches_to_the_reranker():
    # The embedding never sees front matter: a paper found by its author must not lose for a low cosine.
    blended = search._blend(np.array([1.1, 1.0]), np.array([0.5, 0.9]), np.array([True, False]))
    assert blended[0] > blended[1]


class Disagreeing(Recording):
    """Models that embed as the fakes do (bag of words) and rerank like a reranker that slightly prefers one passage
    to the one closest in meaning, and clearly dislikes a third."""

    def __init__(self, fakes, preferred, disliked):
        super().__init__()
        self.fakes, self.preferred, self.disliked = fakes, preferred, disliked

    def encode(self, texts, query=False):
        return self.fakes.encode(texts, query)

    def rerank(self, query, texts):
        self.texts += texts
        return np.array([1.01 if self.preferred in t else 0.0 if self.disliked in t else 1.0 for t in texts])


def test_query_weighs_meaning_as_well_as_the_reranker(embedded, fake_models):
    models = Disagreeing(fake_models, preferred="Reward", disliked="Colour")
    hits = Index(embedded[0], models=models).query("painting robot brush strokes", k=1)
    assert ids(hits) == ["notes/robots/brush.md"]  # far closer in meaning; the reranker's preference is slight


def test_query_finds_documents_not_embedded_yet(embedded):
    # After `sift index` and before `sift embed`, a new document has no vector: keyword search still finds it, and
    # its cosine counts as the others' mean, neither for it nor against it.
    db, root = embedded
    write(root / "new.md", "# Pigment chart\n\npigment pigment\n")
    indexing.index(db)
    assert "notes/new.md" in ids(Index(db).query("pigment chart"))


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


def test_get_by_the_file_s_absolute_path(corpus):
    # An agent that has a file's path, as glean's reviews have each email's, reads it without working out its id.
    db, root = corpus
    index = Index(db)
    assert index.get(str(root / "rl.md")) == index.get("notes/rl.md")
    assert index.document(str(root / "robots" / "brush.md"))["id"] == "notes/robots/brush.md"
    for outside in (root.parent / "rl.md", root / "brush.md"):  # in no collection; not by its file name alone
        with pytest.raises(LookupError, match="unknown document"):
            index.get(str(outside))


def test_document_with_front_matter_fields(corpus):
    # An agent reading a paper wants its link and date with it, without a search to find them.
    index = Index(corpus[0])
    document = index.document("notes/brush.md", fields=["year", "missing"])
    assert document == {
        "id": "notes/robots/brush.md",  # the whole id, also when asked by file name
        "title": "Brush stroke planning",
        "fields": {"year": 2024},
        "text": index.get("notes/robots/brush.md"),
    }


def test_document_lines_without_fields(corpus):
    document = Index(corpus[0]).document("notes/rl.md", 1)
    assert document == {"id": "notes/rl.md", "title": "Reward shaping", "text": "# Reward shaping\n"}


def test_document_with_all_fields(corpus):
    assert Index(corpus[0]).document("notes/brush.md", fields=["*"])["fields"] == {
        "categories": ["cs.RO", "cs.GR"],
        "year": 2024,
    }


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
        | {"links": 0, "unresolved": 0}
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


# --- one result per document -----------------------------------------------------

LONG = "notes/long.md"


@pytest.fixture
def long(embedded, fake_models):
    """The notes, and a document of several passages, all about pigment mixing, each with a word of its own."""
    db, root = embedded
    parts = [f"## Part {i}\n\n" + f"pigment mixing part{i} " * 150 for i in range(3)]
    write(root / "long.md", "# Long\n\n" + "\n\n".join(parts) + "\n")
    indexing.index(db)
    embed.embed_pending(db)
    assert (
        db.execute("SELECT count(*) FROM chunks c JOIN docs d ON d.id = c.doc WHERE d.path = 'long.md'").fetchone()[0]
        > 2
    )
    return Index(db)


@pytest.mark.parametrize("mode", ["search", "vsearch", "query"])
def test_one_result_per_document(long, mode):
    # The passages of one long document filled the results, and the other documents were never seen.
    hits = getattr(long, mode)("pigment mixing part0 part1 part2", k=2)
    assert len(hits) == 2
    assert ids(hits)[0] == LONG
    assert len(set(ids(hits))) == 2


@pytest.mark.parametrize("mode", ["search", "vsearch", "query"])
def test_all_passages(long, mode):
    hits = getattr(long, mode)("pigment mixing part0 part1 part2", k=10, all_passages=True)
    assert ids(hits).count(LONG) > 2


@pytest.mark.parametrize("mode", ["search", "query"])
def test_a_document_shows_its_best_passage(long, mode):
    hit = getattr(long, mode)("part2", k=1)[0]
    assert (hit["id"], "part2" in hit["text"]) == (LONG, True)


def test_vsearch_looks_further_for_other_documents(long):
    # When the best passages all come from one document, vsearch looks further down for more documents.
    hits = long.vsearch("pigment mixing part0 part1 part2", k=4, filters={"categories": {"not": "cs.LG"}})
    assert len(set(ids(hits))) == 4


# --- ordering by a front-matter field --------------------------------------------


def day(hit):
    return hit["id"].rsplit("/", 1)[1].removesuffix(".md")


@pytest.mark.parametrize("sort, expected", [("created", DAYS), ("-created", DAYS[::-1])])
def test_sort_by_a_front_matter_field(dated, sort, expected):
    # "The newest papers on X": the best results, newest first.
    hits = dated.search("pigment", k=20, sort=sort)
    assert [day(h) for h in hits] == [*expected, "brush"]  # those without the field come last


@pytest.mark.parametrize("mode", ["vsearch", "query"])
def test_sort_in_every_mode(dated, mode):
    hits = getattr(dated, mode)("pigment", k=20, sort="-created", filters={"created": {">=": "2024"}})
    assert [day(h) for h in hits] == DAYS[1:][::-1]


def test_sort_puts_numbers_before_text(corpus):
    db, root = corpus
    for name, rank in (("a", "2"), ("b", '"x"'), ("c", "10"), ("d", "null")):
        write(root / "ranked" / f"{name}.md", f"---\nrank: {rank}\n---\n# {name}\nranked\n")
    indexing.index(db)
    assert [day(h) for h in Index(db).search("ranked", sort="rank")] == ["a", "c", "b", "d"]


@pytest.mark.parametrize("sort", ["", "-", 'bad"key', 1, ["created"]])
def test_bad_sort_is_a_clear_error(corpus, sort):
    with pytest.raises(ValueError, match="sort"):
        Index(corpus[0]).search("brush", sort=sort)


# --- documents like a document ----------------------------------------------------


def test_similar_finds_documents_like_one(dated):
    # "More papers like this one", without putting the paper into words first.
    hits = dated.similar("notes/dated/2024-01-01.md", k=4)
    assert len(hits) == 4
    assert all(h["id"].startswith("notes/dated/") for h in hits)  # the papers with the same words
    assert "notes/dated/2024-01-01.md" not in ids(hits)  # not the document itself
    assert hits[0]["score"] >= hits[-1]["score"]


def test_similar_takes_what_searches_take(dated):
    hits = dated.similar(
        "notes/2024-01-01.md", filters={"created": {">=": "2024-06"}}, fields=["created"], compact=True, sort="created"
    )
    assert [(day(h), h["fields"]["created"], "text" in h) for h in hits] == [
        (d, d, False) for d in ("2024-06-15", "2024-12-31", "2025-01-01")
    ]


def test_similar_is_one_result_per_document(long):
    assert len(set(ids(long.similar("notes/robots/brush.md", k=10)))) == len(
        long.similar("notes/robots/brush.md", k=10)
    )


def test_similar_needs_embeddings(corpus, fake_models):
    with pytest.raises(LookupError, match="sift embed"):
        Index(corpus[0]).similar("notes/rl.md")


def test_similar_to_a_document_not_embedded_yet(embedded):
    db, root = embedded
    write(root / "new.md", "# New\nfresh\n")
    indexing.index(db)
    with pytest.raises(LookupError, match=r"'notes/new\.md' has no embeddings yet, run `sift embed`"):
        Index(db).similar("notes/new.md")


def test_similar_to_an_unknown_document(embedded):
    with pytest.raises(LookupError, match="unknown document"):
        Index(embedded[0]).similar("notes/missing.md")


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
        (1, "# Reward shaping\n"),  # a line number, as agents also send it
    ],
)
def test_get_lines(corpus, lines, text):
    assert Index(corpus[0]).get("notes/rl.md", lines) == text


def test_get_returns_the_indexed_text_until_the_next_index(corpus):
    # Hits and get always agree: both come from the index, not from the file as it is now.
    db, root = corpus
    write(root / "rl.md", "# Rewritten\n")
    assert Index(db).get("notes/rl.md").startswith("# Reward shaping")
    (root / "rl.md").unlink()
    indexing.index(db)
    with pytest.raises(LookupError, match="unknown document"):
        Index(db).get("notes/rl.md")


def test_lines_past_the_end_are_a_clear_error(corpus):
    # An empty answer read to agents as an empty passage.
    with pytest.raises(ValueError, match=r"notes/rl\.md has 3 lines; line 9 is past its end"):
        Index(corpus[0]).get("notes/rl.md", "9-12")


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


def test_browse_lists_documents_without_a_query(corpus):
    docs = Index(corpus[0]).browse()
    assert ids(docs) == ["notes/plain.txt", "notes/rl.md", "notes/robots/brush.md"]  # by path
    assert docs[1]["title"] == "Reward shaping"
    assert docs[1]["text"].startswith("# Reward shaping")  # the document's start, as a hit's passage
    assert docs[2]["text"].startswith("# Brush stroke planning")  # after its front matter, as its first passage


def test_browse_lists_documents_of_front_matter_alone(corpus):
    db, root = corpus
    write(root / "empty.md", "---\ntitle: Only metadata\n---\n")
    indexing.index(db)
    # its passage is its front matter, as for search
    assert {"id": "notes/empty.md", "title": "Only metadata", "text": "title: Only metadata"} in Index(db).browse()
    assert [h["id"] for h in Index(db).search("metadata")] == ["notes/empty.md"]


def test_browse_takes_what_searches_take(corpus):
    index = Index(corpus[0])
    assert ids(index.browse(path="robots/*")) == ["notes/robots/brush.md"]
    assert ids(index.browse(collection="notes", k=1)) == ["notes/plain.txt"]
    assert index.browse(filters={"year": 2024}, fields=["year"], compact=True) == [
        {"id": "notes/robots/brush.md", "title": "Brush stroke planning", "fields": {"year": 2024}}
    ]
    assert ids(index.browse(sort="-year"))[0] == "notes/robots/brush.md"  # those without the field come last
    with pytest.raises(LookupError, match="no collection named 'nope'"):
        index.browse(collection="nope")


def test_folders_are_those_directly_inside_a_path(corpus):
    db, root = corpus
    for path in ("robots/arms/grip.md", "robots/arms/lift.md", "robots/legs/walk.md", "z/deep/er/x.md"):
        write(root / path, "# x\n")
    indexing.index(db)
    index = Index(db)
    assert index.folders("notes") == ["robots", "z"]
    assert index.folders("notes", "robots/") == ["arms", "legs"]
    assert index.folders("notes", "robots/arms/") == []
    for written in ("robots", "/robots", "robots//"):  # as a person might write the folder
        assert index.folders("notes", written) == ["arms", "legs"]
    assert index.folders("notes", "/") == ["robots", "z"]
    with pytest.raises(LookupError, match="no collection named 'nope'"):
        index.folders("nope")


def test_the_file_a_document_came_from(corpus, tmp_path):
    db, root = corpus
    write(tmp_path / "qa" / "faq.jsonl", '{"id": "q1", "text": "a record"}\n')
    store.add_collection(db, tmp_path / "qa", "qa")
    indexing.index(db)
    index = Index(db)
    assert index.file("notes/brush.md") == root / "robots" / "brush.md"  # ids as for get
    assert index.file("qa/faq/q1") is None  # a record is a part of its file, not one of its own


def test_get_by_a_file_name_many_documents_have(corpus):
    db, root = corpus
    for folder in "abcdef":
        write(root / folder / "same.md", f"# {folder}\n")
    indexing.index(db)
    with pytest.raises(LookupError, match=r"notes/e/same\.md, \.\.\.; give the whole id"):
        Index(db).get("notes/same.md")
