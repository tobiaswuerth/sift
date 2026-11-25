import os
from pathlib import Path

import pytest
from hypothesis import settings

from tests.examples import EXPLORE, EXPLORED, PLAIN

settings.register_profile(PLAIN, derandomize=True, database=None, deadline=None, print_blob=True)
settings.register_profile(EXPLORE, max_examples=EXPLORED, database=None, deadline=None, print_blob=True)
settings.load_profile(PLAIN)


@pytest.fixture(autouse=True, scope="session")
def no_home_but_a_temporary_one(tmp_path_factory: pytest.TempPathFactory) -> Path:
    unused = tmp_path_factory.mktemp("no-test-uses-this")
    os.environ["SIFT_HOME"] = str(unused / "home")
    for variable in ("LOCALAPPDATA", "XDG_STATE_HOME"):
        os.environ[variable] = str(unused / "state")
    return unused / "home"
