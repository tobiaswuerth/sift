from pathlib import Path

from tests.files import write
from tests.surface.process import sift, through_stdio
from tests.surface.stdio import STDIO, answers_of, found_ids, stdio_search


def test_an_agent_s_stdio_starts_a_home_process_that_answers_it(tmp_path: Path) -> None:
    home = tmp_path / "home"
    write(tmp_path / "notes" / "halo.md", "# Halo\n\nDark matter halo models.\n")
    assert sift(home, "add", str(tmp_path / "notes")).returncode == 0
    assert sift(home, "index").returncode == 0
    done = through_stdio(home, STDIO, stdio_search("halo"))
    answers = answers_of(done.stdout)
    assert (done.returncode, sorted(answers), found_ids(answers[2])) == (0, [1, 2], ["notes/halo.md"]), done.stderr
    assert "started a home process for agents on port" in done.stderr
    logs = sorted(log.name.partition("-")[0] for log in (home / "logs").glob("*.log"))
    assert logs == ["add", "index", "serve", "stdio"]
