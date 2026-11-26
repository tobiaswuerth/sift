from pathlib import Path

import pytest

from sift.api.collections import AddCollection, Audience
from sift.api.embedding import EmbedCollections
from sift.app.collections import add
from sift.app.embedding import embed
from sift.interfaces.acts.acting import Acting
from sift.settings import Settings
from tests.agent_tools import Direct
from tests.files import write
from tests.homes import collection, indexed, quiet
from tests.models import Hashed


@pytest.fixture
def direct(tmp_path: Path) -> Direct:
    home = tmp_path / "home"
    plan = "---\ncreated: 2026-01-02\n---\n# Plan\n\nNote: the plan of Kestrel names [[Secret]].\n"
    collection(tmp_path, home, "open", {"Plan.md": plan, "Other.md": "Note: an osprey over the lake.\n"})
    write(tmp_path / "hidden" / "Secret.md", "Note: Kestrel is the secret project, see [[Plan]].\n")
    add(home, AddCollection(str(tmp_path / "hidden"), agents=False))
    indexed(home)
    embed(home, EmbedCollections(), quiet, Hashed())
    return Direct(Acting(Settings(home=home, user=None, cwd=tmp_path), quiet, float, Hashed, Audience.AGENTS))
