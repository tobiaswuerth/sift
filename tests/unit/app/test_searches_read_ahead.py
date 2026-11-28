import threading
from contextlib import ExitStack
from pathlib import Path

from sift.api.collections import Audience
from sift.api.search import SearchKeywords
from sift.app import searching
from sift.storage.columns import TOUCH_OWNERS
from sift.storage.readers import FEWEST, Ahead
from tests.homes import home_of
from tests.waits import until


def test_each_store_a_search_opens_is_read_ahead_beside_it_and_its_readers_go_with_it(tmp_path: Path) -> None:
    home = home_of(tmp_path, {"notes/a.md": "# A\n\nPigment and binder.\n"})
    narrowing = searching.narrowed(home, SearchKeywords("pigment"), Audience.PEOPLE)
    with ExitStack() as cleanup:
        kept = searching.opened(cleanup, narrowing, searching.Wants())
        (each,) = kept.searching
        ahead = each.transaction.ahead
        assert isinstance(ahead, Ahead)
        assert ahead.path == each.collection.store
        ahead.touched(TOUCH_OWNERS, list(range(1, 2 * FEWEST)))
        assert ahead.held is not None
    assert ahead.held is None
    assert until(lambda: not any(thread.name.startswith("sift-ahead") for thread in threading.enumerate()))
