import pytest

from sift.domain.plan.keyword_state import KeywordState, keyword_state, remaking


@pytest.mark.parametrize(
    "found, state",
    [
        ("abc", KeywordState.CURRENT),
        (remaking("abc"), KeywordState.REMAKING),
        (remaking("other"), KeywordState.TAKEN_OVER),
        ("other", KeywordState.STALE),
        (None, KeywordState.STALE),
    ],
)
def test_the_state_of_a_keyword_index_is_read_from_its_mark_and_this_sift_s_fingerprint(
    found: str | None, state: KeywordState
) -> None:
    assert keyword_state(found, "abc") is state
