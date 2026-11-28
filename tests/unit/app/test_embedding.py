import contextlib
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from sift.api.collections import AddCollection, DeactivateCollections
from sift.api.embedding import CollectionEmbedded, EmbedCollections, Embedded
from sift.api.errors import InvalidRequestError, NotFoundError, Problem
from sift.api.events import Advanced, CollectionStarted, Event, Noted, Phase, PhaseDone, PhaseStarted
from sift.api.indexing import IndexCollections, Skipped
from sift.app.collections import add, deactivate
from sift.app.embedding import BROKEN_REMEDY, FORGOTTEN, FORGOTTEN_REMEDY, Embedding, embed, models_held
from sift.app.indexing import index
from sift.domain.codec.codes import Floats, Halves
from sift.inference.prepared import ModelsPrepared
from sift.storage.catalog import known_collections
from sift.storage.store_files import CATALOG
from sift.storage.vectors import EMBED_REMEDY
from tests.damage import damage_table
from tests.files import write
from tests.homes import home_of, indexed, indexed_home, quiet
from tests.later import LATER
from tests.models import BROKEN, Hashed

LONG = "\n\n".join(["Pruning skips what cannot win. " * 20] * 8)


def counted(store: Path, sql: str) -> int:
    with contextlib.closing(sqlite3.connect(store)) as database:
        (found,) = cast("tuple[int]", database.execute(sql).fetchone())
        return found


@pytest.fixture
def notes(tmp_path: Path) -> Path:
    root = tmp_path / "notes"
    write(root / "pruning.md", f"# Block-Max Pruning\n\nRanges are scored best bound first.\n\n## More\n\n{LONG}\n")
    write(root / "halo.md", "# Halo\n\nDark matter halo models.\n")
    return root


@pytest.fixture
def home(tmp_path: Path, notes: Path) -> Path:
    return indexed_home(tmp_path, notes)


def store(home: Path, name: str = "notes") -> Path:
    return home / "collections" / f"{name}.sqlite"


def chunks(home: Path, name: str = "notes") -> int:
    return counted(store(home, name), "SELECT count(*) FROM chunks")


def test_every_chunk_is_embedded_once_with_its_progress(home: Path) -> None:
    said: list[Event] = []
    result = embed(home, EmbedCollections(), said.append, Hashed(), Embedding(chunks_a_call=2))
    total = chunks(home)
    assert result.collections == (CollectionEmbedded("notes", Embedded(total, 0)),)
    assert said[1:4] == [PhaseStarted(Phase.LOAD, None), PhaseDone(1), PhaseStarted(Phase.EMBED, total)]
    assert [event.done for event in said if isinstance(event, Advanced)] == [*range(2, total, 2), total]
    assert said[-1] == PhaseDone(total)
    assert counted(store(home), "SELECT count(*) FROM vectors") == total
    again: list[Event] = []
    assert embed(home, EmbedCollections(), again.append, Hashed()).collections[0].outcome == Embedded(0, 0)
    assert not [event for event in again if isinstance(event, PhaseStarted)]


def sized_notes(tmp_path: Path) -> Path:
    for n in range(12):
        write(tmp_path / "notes" / f"n{n:02}.md", f"# N{n}\n\n" + "word " * (5 + 37 * n % 11 * 9))
    return indexed_home(tmp_path, tmp_path / "notes")


def test_passages_read_together_are_embedded_longest_first_a_call_at_a_time(tmp_path: Path) -> None:
    home, models, said = sized_notes(tmp_path), Hashed(), list[Event]()
    embed(home, EmbedCollections(), said.append, models, Embedding(chunks_a_call=2, chunks_a_read=8))
    told = [[len(text.split()) for text in texts] for texts, _ in models.asked]
    assert told == [[88, 79], [61, 52], [43, 25], [16, 7], [97, 70], [34, 7]]
    assert [event.done for event in said if isinstance(event, Advanced)] == [2, 4, 6, 8, 10, 12]
    assert counted(store(home), "SELECT count(*) FROM vectors") == 12


def test_an_embedding_cut_short_keeps_every_call_before_it_and_is_finished_by_embedding_again(
    tmp_path: Path,
) -> None:
    home, sizes = sized_notes(tmp_path), Embedding(chunks_a_call=2, chunks_a_read=8)
    with pytest.raises(KeyboardInterrupt):
        embed(home, EmbedCollections(), quiet, Stopping(Hashed(), 3), sizes)
    assert counted(store(home), "SELECT count(*) FROM vectors") == 6
    assert embed(home, EmbedCollections(), quiet, Hashed(), sizes).collections[0].outcome == Embedded(6, 0)
    assert counted(store(home), "SELECT count(DISTINCT chunk) FROM vectors") == 12


def test_the_model_is_compiled_once_before_a_collection_with_many_passages_to_embed(tmp_path: Path) -> None:
    home, models, said = sized_notes(tmp_path), Hashed(), list[Event]()
    write(tmp_path / "few" / "a.md", "# A\n\nA note.\n")
    add(home, AddCollection(str(tmp_path / "few")))
    indexed(home)
    embed(home, EmbedCollections(), said.append, models, Embedding(compile_from=12))
    compiling = [index for index, event in enumerate(said) if event == PhaseStarted(Phase.COMPILE, None)]
    assert (models.compiled, len(compiling)) == ([True], 1)
    assert said[compiling[0] + 1 : compiling[0] + 4] == [PhaseDone(1), Noted("compiled"), PhaseStarted(Phase.EMBED, 12)]
    fewer = Hashed()
    embed(home, EmbedCollections(reset=True), quiet, fewer, Embedding(compile_from=13))
    assert fewer.compiled == []


def test_chunks_after_the_first_are_embedded_with_their_title_and_documents_as_documents(home: Path) -> None:
    models = Hashed()
    embed(home, EmbedCollections(), quiet, models)
    texts = [text for asked, query in models.asked for text in asked if not query]
    assert "# Halo\n\nDark matter halo models." in texts
    pruning = [text for text in texts if "Pruning skips" in text]
    assert pruning[0].startswith("# Block-Max Pruning\n\nRanges are scored best bound first.")
    assert pruning[1].startswith("Block-Max Pruning\n\nPruning skips what cannot win.")
    assert all(not query for _, query in models.asked)


def test_a_chunk_the_model_gives_no_vector_for_is_counted_and_stays_to_embed(home: Path, notes: Path) -> None:
    write(notes / "odd.md", f"# Odd\n\nAn {BROKEN} passage.\n")
    indexed(home)
    said: list[Event] = []
    outcome = embed(home, EmbedCollections(), said.append, Hashed()).collections[0].outcome
    assert outcome == Embedded(chunks(home) - 1, 1)
    assert said[-2:] == [
        PhaseDone(chunks(home) - 1),
        Noted("1 chunk could not be embedded: the model gave no vector", BROKEN_REMEDY),
    ]
    assert embed(home, EmbedCollections(), quiet, Hashed()).collections[0].outcome == Embedded(0, 1)


def test_a_collection_of_another_model_is_skipped_until_every_one_is_embedded_again(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "more" / "a.md", "# A\n\nAnother note.\n")
    add(home, AddCollection(str(tmp_path / "more")))
    indexed(home)
    embed(home, EmbedCollections(), quiet, Hashed())
    with contextlib.closing(sqlite3.connect(home / CATALOG)) as catalog, catalog:
        catalog.execute("UPDATE collections SET active = 0 WHERE name = 'more'")
    other = Hashed(name="other-model", width=32)
    (skipped,) = embed(home, EmbedCollections(), quiet, other).collections
    assert other.asked == []
    assert skipped == CollectionEmbedded(
        "notes",
        Skipped(
            Problem.OTHER_MODEL,
            "the store holds vectors of hashed-words, not of other-model",
            EMBED_REMEDY.format(model="other-model"),
        ),
    )
    with pytest.raises(InvalidRequestError, match="embeds every collection again"):
        embed(home, EmbedCollections(("notes",), reset=True), quiet, other)
    (redone,) = embed(home, EmbedCollections(reset=True), quiet, other).collections
    assert redone == CollectionEmbedded("notes", Embedded(chunks(home), 0))
    assert counted(store(home, "more"), "SELECT count(*) FROM vectors") == 0
    assert embed(home, EmbedCollections(("more",)), quiet, other).collections[0].outcome == Embedded(1, 0)


def test_a_collection_whose_store_is_gone_is_skipped_and_others_embedded(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "more" / "a.md", "# A\n\nAnother note.\n")
    add(home, AddCollection(str(tmp_path / "more")))
    indexed(home)
    store(home).unlink()
    found = {each.name: each.outcome for each in embed(home, EmbedCollections(), quiet, Hashed()).collections}
    assert isinstance(found["notes"], Skipped)
    assert found["more"] == Embedded(1, 0)


def test_the_collections_named_alone_are_embedded_and_one_unknown_embeds_none(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "more" / "a.md", "# A\n\nAnother note.\n")
    add(home, AddCollection(str(tmp_path / "more")))
    indexed(home)
    models = Hashed()
    with pytest.raises(NotFoundError, match="no collection is named 'nope'"):
        embed(home, EmbedCollections(("more", "nope")), quiet, models)
    held = [counted(store(home, name), "SELECT count(*) FROM vectors") for name in ("notes", "more")]
    assert (held, models.asked) == ([0, 0], [])
    done = embed(home, EmbedCollections(("more",)), quiet, models).collections
    assert (done, counted(store(home), "SELECT count(*) FROM vectors")) == (
        (CollectionEmbedded("more", Embedded(1, 0)),),
        0,
    )


def test_a_reset_says_which_store_it_could_not_clear(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "more" / "a.md", "# A\n\nAnother note.\n")
    add(home, AddCollection(str(tmp_path / "more")))
    indexed(home)
    with contextlib.closing(sqlite3.connect(home / CATALOG)) as catalog, catalog:
        catalog.execute("UPDATE collections SET active = 0 WHERE name = 'more'")
    store(home, "more").unlink()
    said: list[Event] = []
    embed(home, EmbedCollections(reset=True), said.append, Hashed())
    assert next(event.text for event in said if isinstance(event, Noted)).startswith(
        "its vectors could not be forgotten:"
    )


def test_a_reset_says_an_inactive_collection_whose_vectors_it_forgot_is_left_unembedded(
    home: Path, tmp_path: Path
) -> None:
    write(tmp_path / "more" / "a.md", "# A\n\nAnother note.\n")
    add(home, AddCollection(str(tmp_path / "more")))
    indexed(home)
    embed(home, EmbedCollections(), quiet, Hashed())
    deactivate(home, DeactivateCollections(("more",)))
    said: list[Event] = []
    (redone,) = embed(home, EmbedCollections(reset=True), said.append, Hashed()).collections
    assert (redone.name, counted(store(home, "more"), "SELECT count(*) FROM vectors")) == ("notes", 0)
    assert [(each.text, each.remedy) for each in said if isinstance(each, Noted)] == [(FORGOTTEN, FORGOTTEN_REMEDY)]
    assert CollectionStarted("more", str(tmp_path / "more")) in said


@dataclass(slots=True)
class Stopping:
    models: Hashed
    calls_left: int

    @property
    def name(self) -> str:
        return self.models.name

    def prepare(self) -> ModelsPrepared:
        if self.calls_left < 0:
            raise RuntimeError("the model would not load")
        return self.models.prepare()

    def prepare_many(self) -> ModelsPrepared:
        return self.models.prepare_many()

    def prepare_ranker(self) -> ModelsPrepared:
        return self.models.prepare_ranker()

    def encode(self, texts: Sequence[str], *, query: bool) -> Halves:
        if not self.calls_left:
            raise KeyboardInterrupt
        self.calls_left -= 1
        return self.models.encode(texts, query=query)

    def rerank(self, query: str, texts: Sequence[str]) -> Floats:
        return self.models.rerank(query, texts)


def test_a_reset_cut_short_is_finished_by_embedding_again(home: Path, tmp_path: Path) -> None:
    write(tmp_path / "more" / "a.md", "# A\n\nAnother note.\n")
    add(home, AddCollection(str(tmp_path / "more")))
    indexed(home)
    embed(home, EmbedCollections(), quiet, Hashed())
    other = Hashed(name="other-model", width=32)
    with pytest.raises(KeyboardInterrupt):
        embed(home, EmbedCollections(reset=True), quiet, Stopping(other, 0))
    assert other.prepared == [True]
    resumed = embed(home, EmbedCollections(), quiet, other).collections
    assert {each.name: each.outcome for each in resumed} == {
        "more": Embedded(1, 0),
        "notes": Embedded(chunks(home), 0),
    }


def test_a_reset_whose_model_will_not_load_forgets_nothing(home: Path) -> None:
    embed(home, EmbedCollections(), quiet, Hashed())
    with pytest.raises(RuntimeError, match="would not load"):
        embed(home, EmbedCollections(reset=True), quiet, Stopping(Hashed(name="other-model"), -1))
    assert counted(store(home), "SELECT count(*) FROM vectors") == chunks(home)


def test_a_store_taken_by_another_model_midway_is_skipped_saying_so(home: Path) -> None:
    said: list[Event] = []
    busy: list[Skipped | Embedded] = []

    def taken(event: Event) -> None:
        said.append(event)
        if isinstance(event, Advanced) and event.done == 2:
            other = embed(home, EmbedCollections(reset=True), quiet, Hashed(name="other-model"))
            busy.extend(each.outcome for each in other.collections)
            with contextlib.closing(sqlite3.connect(store(home))) as database:
                database.execute("UPDATE meta SET value = 'other-model' WHERE key = 'vector_model'")
                database.commit()

    sizes = Embedding(chunks_a_call=1, chunks_a_read=1)
    (outcome,) = embed(home, EmbedCollections(), taken, Hashed(), sizes).collections
    assert isinstance(outcome.outcome, Skipped)
    assert outcome.outcome.message == "the store holds vectors of other-model, not of hashed-words"
    assert [event for event in said if isinstance(event, PhaseStarted)][-1].phase is Phase.EMBED
    assert not isinstance(said[-2], PhaseDone)
    assert said[-1] == Noted(outcome.outcome.message, outcome.outcome.remedy, Problem.OTHER_MODEL)
    (refused,) = busy
    assert isinstance(refused, Skipped)
    assert "another sift writes it now (an embedding, process " in refused.message


def test_a_store_a_reset_could_not_clear_is_skipped_saying_why(home: Path) -> None:
    embed(home, EmbedCollections(), quiet, Hashed())
    damage_table(store(home), "vectors")
    said: list[Event] = []
    (outcome,) = embed(home, EmbedCollections(reset=True), said.append, Hashed(name="other-model")).collections
    assert isinstance(outcome.outcome, Skipped)
    assert len([event for event in said if isinstance(event, Noted)]) == 1
    assert outcome.outcome.message.startswith("its vectors could not be forgotten: ")
    assert "damaged" in outcome.outcome.message


@pytest.mark.parametrize("reset", [False, True])
def test_the_model_loads_once_however_many_collections_wait(home: Path, tmp_path: Path, *, reset: bool) -> None:
    write(tmp_path / "more" / "comets.md", "# Comets\n\nIce and dust.\n")
    add(home, AddCollection(str(tmp_path / "more")))
    indexed(home)
    said: list[Event] = []
    models = Hashed()
    embed(home, EmbedCollections(reset=reset), said.append, models)
    assert models.prepared == [True]
    assert [event for event in said if isinstance(event, PhaseStarted)].count(PhaseStarted(Phase.LOAD, None)) == 1


def test_a_note_changed_in_one_chunk_needs_that_chunk_embedded_alone(home: Path, notes: Path) -> None:
    embed(home, EmbedCollections(), quiet, Hashed())
    pruning = notes / "pruning.md"
    pruning.write_text(pruning.read_text(encoding="utf-8").replace("Ranges are", "Blocks are"), encoding="utf-8")
    index(home, IndexCollections(), quiet, readers=LATER)
    assert chunks(home) > 2
    assert counted(store(home), "SELECT chunks - vectors FROM counts") == 1


def test_a_collection_whose_store_is_out_of_reach_counts_as_embedded_by_no_model(tmp_path: Path) -> None:
    home = home_of(tmp_path, {"notes/a.md": "Note: alpha.\n"}, embedded=Hashed())
    assert models_held(home, ("notes",)) == {"notes": Hashed().name}
    (one,) = known_collections(home)
    one.store.unlink()
    assert models_held(home, ("notes",)) == {}
