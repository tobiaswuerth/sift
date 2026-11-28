from sift.api.errors import Problem
from sift.api.importing import Brought, CollectionImported, HomeImported
from sift.api.indexing import Skipped
from sift.interfaces.cli.commands.importing import imported_table


def brought(documents: int, passages: int, vectors: int) -> Brought:
    return Brought("a store", documents, documents, passages, passages, passages, vectors, "a-model")


def test_the_counts_of_an_import_stay_apart_however_many_digits_they_have() -> None:
    done = HomeImported(
        "a home",
        (
            CollectionImported("papers", brought(3_252_359, 3_252_534, 3_252_534)),
            CollectionImported("gone", Skipped(Problem.FOLDER_MISSING, "its folder is missing", "bring it back")),
            CollectionImported("laws", brought(5_339, 75_851, 75_837)),
        ),
    )
    assert imported_table(done) == (
        "collection               documents                passages                 vectors",
        "papers      3,252,359 -> 3,252,359  3,252,534 -> 3,252,534  3,252,534 -> 3,252,534",
        "laws                5,339 -> 5,339        75,851 -> 75,851        75,851 -> 75,837",
    )


def test_an_import_that_brought_nothing_shows_no_table() -> None:
    skipped = Skipped(Problem.FOLDER_MISSING, "its folder is missing", "bring it back")
    assert imported_table(HomeImported("a home", (CollectionImported("gone", skipped),))) == ()
    assert imported_table(HomeImported("a home", ())) == ()
