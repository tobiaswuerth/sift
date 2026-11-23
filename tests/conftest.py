import os
from pathlib import Path

import pytest
from hypothesis import settings

settings.register_profile("sift", derandomize=True, database=None, deadline=None, print_blob=True)
settings.register_profile("explore", max_examples=2000, database=None, deadline=None, print_blob=True)
settings.load_profile("sift")


@pytest.fixture(autouse=True, scope="session")
def no_home_but_a_temporary_one(tmp_path_factory: pytest.TempPathFactory) -> Path:
    home = tmp_path_factory.mktemp("no-test-uses-this") / "home"
    os.environ["SIFT_HOME"] = str(home)
    return home
