from hypothesis import settings

from tests.examples import EXPLORE, PLAIN, few, more

RUN_UNDER = "the profile this run began under"


def test_few_and_more_widen_under_the_explore_profile_alone_whichever_profile_the_run_is_under() -> None:
    settings.register_profile(RUN_UNDER, settings.default)
    try:
        settings.load_profile(PLAIN)
        assert (few(3).max_examples, more(200).max_examples) == (3, 200)
        settings.load_profile(EXPLORE)
        assert (few(3).max_examples, few(300).max_examples, more(200).max_examples) == (200, 300, 2000)
    finally:
        settings.load_profile(RUN_UNDER)
