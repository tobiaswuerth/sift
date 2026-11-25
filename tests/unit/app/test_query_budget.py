from pathlib import Path

from sift.api.collections import AddCollection
from sift.api.embedding import EmbedCollections
from sift.api.indexing import IndexCollections
from sift.api.search import SearchHybrid
from sift.app.collections import add
from sift.app.embedding import embed
from sift.app.indexing import index
from sift.app.query import RERANK_PART, query
from sift.app.searching import Watching
from tests.files import write
from tests.models import Hashed


def collection(home: Path, root: Path) -> None:
    add(home, AddCollection(str(root)))
    index(home, IndexCollections(), lambda _: None)
    embed(home, EmbedCollections((root.name,)), lambda _: None, Hashed())


def reranking_takes_ten_seconds(models: Hashed) -> Watching:
    return Watching(clock=lambda: 10.0 * len(models.reranked))


def test_a_query_out_of_time_after_a_rerank_widens_no_further_and_is_partial(tmp_path: Path) -> None:
    home = tmp_path / "home"
    for name in ("a", "b", "c"):
        for n in range(4):
            text = f"---\ndoi: 10.1000/w{n}\n---\n# Work {n}\n\n{'quasar ' * (4 - n)}light.\n"
            write(tmp_path / name / f"w{n}.md", text)
        collection(home, tmp_path / name)
    request = SearchHybrid("quasar", k=4, candidates=4)
    unhurried = Hashed()
    assert (query(home, request, unhurried).partial, len(unhurried.reranked) > 1) == (False, True)
    models = Hashed()
    found = query(
        home,
        SearchHybrid("quasar", k=4, candidates=4, budget=1.0),
        models,
        watching=reranking_takes_ten_seconds(models),
    )
    assert (found.partial, len(models.reranked)) == (True, 1)


def test_a_query_out_of_time_midway_through_its_rerank_keeps_the_part_reranked(tmp_path: Path) -> None:
    home, many = tmp_path / "home", RERANK_PART + 8
    for n in range(many):
        write(tmp_path / "notes" / f"n{n:02}.md", f"# Note {n}\n\nA halo, number {n}.\n")
    collection(home, tmp_path / "notes")
    unhurried = Hashed()
    assert len(query(home, SearchHybrid("halo", k=many, candidates=many), unhurried).hits) == many
    assert [len(texts) for _, texts in unhurried.reranked] == [RERANK_PART, 8]
    models = Hashed()
    request = SearchHybrid("halo", k=many, candidates=many, budget=1.0)
    found = query(home, request, models, watching=reranking_takes_ten_seconds(models))
    assert (found.partial, len(found.hits), [len(texts) for _, texts in models.reranked]) == (
        True,
        RERANK_PART,
        [RERANK_PART],
    )
