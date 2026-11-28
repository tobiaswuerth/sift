import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.api.ids import MOST_COLLECTIONS, ROW_BITS, CollectionId, first_row, owner

collections = st.integers(1, MOST_COLLECTIONS).map(CollectionId)


def test_a_collection_s_rows_start_at_its_id_shifted() -> None:
    assert first_row(CollectionId(1)) == 1 << ROW_BITS
    assert first_row(CollectionId(3)) == 3 << ROW_BITS


@given(collections, st.integers(0, (1 << ROW_BITS) - 1))
def test_every_row_of_a_collection_names_it_as_its_owner(collection: CollectionId, offset: int) -> None:
    assert owner(first_row(collection) + offset) == collection


def test_the_last_row_of_the_last_collection_fits_sqlite_s_integers() -> None:
    assert first_row(CollectionId(MOST_COLLECTIONS)) + (1 << ROW_BITS) - 1 == (1 << 63) - 1


@pytest.mark.parametrize("collection", [0, -1, MOST_COLLECTIONS + 1])
def test_a_collection_id_out_of_range_is_refused(collection: int) -> None:
    with pytest.raises(ValueError, match=f"between 1 and {MOST_COLLECTIONS}"):
        first_row(CollectionId(collection))
