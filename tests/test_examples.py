from hypothesis import settings

from tests.examples import EXPLORE, PLAIN, few, more


def test_few_and_more_widen_under_the_explore_profile_alone() -> None:
    assert (few(3).max_examples, more(200).max_examples) == (3, 200)
    settings.load_profile(EXPLORE)
    try:
        assert (few(3).max_examples, few(300).max_examples, more(200).max_examples) == (200, 300, 2000)
    finally:
        settings.load_profile(PLAIN)
