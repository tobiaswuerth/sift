"""The commands that change the index or serve it keep a log of their own."""

import datetime
import logging
import pathlib

import pytest

import sift
from sift import cli, console, logs

NOW = datetime.datetime(2026, 10, 23, 21, 5, 9)


def lines(path):
    return [line.split(" ", 2)[2] for line in path.read_text(encoding="utf-8").splitlines()]


def serve_and_fail(kept):
    with logs.kept("serve", NOW) as path:
        kept.append(path)
        console.say("loading models and vectors into RAM ...")
        with console.Progress(2, "files", "index: ") as progress:
            progress.update(2)
        logging.getLogger("mcp.server").warning("a library's record")
        logging.getLogger("uvicorn.access").propagate = False  # as uvicorn sets it up
        logs.attach("uvicorn.access")
        logging.getLogger("uvicorn.access").info('"POST /mcp HTTP/1.1" 200')
        raise SystemExit("error: the port is taken")


def test_a_log_holds_what_sift_said_the_libraries_logged_and_how_it_ended(home, capsys):
    kept = []
    with pytest.raises(SystemExit):
        serve_and_fail(kept)
    (path,) = kept
    assert path == home / "logs" / "serve-2026-10-23_210509.log"
    found = lines(path)
    assert found[:2] == [f"INFO sift.console: sift {sift.__version__}, serve",
                         "INFO sift.console: loading models and vectors into RAM ..."]  # fmt: skip
    assert found[2].startswith("INFO sift.console: index: [####################] 100% 2/2 files, ")
    assert found[3:] == [
        "WARNING mcp.server: a library's record",
        'INFO uvicorn.access: "POST /mcp HTTP/1.1" 200',
        "ERROR sift.console: error: the port is taken",
    ]
    console.say("after")
    assert "after" not in path.read_text(encoding="utf-8")


def test_said_once_even_where_the_root_logger_prints(home, capsys):
    printing = logging.StreamHandler()  # as an MCP server's logging set-up adds one
    logging.getLogger().addHandler(printing)
    try:
        with logs.kept("index", NOW):
            console.say("3 added")
    finally:
        logging.getLogger().removeHandler(printing)
    assert capsys.readouterr().err.count("3 added") == 1


@pytest.mark.parametrize("raised, logged", [(KeyboardInterrupt, "stopped"), (ValueError("bad"), "ValueError: bad")])
def test_how_a_command_ended(home, raised, logged):
    with pytest.raises(type(raised) if isinstance(raised, Exception) else raised), logs.kept("embed", NOW) as path:
        raise raised
    assert lines(path)[-1] == f"ERROR sift.console: {logged}"


def test_a_command_that_ends_well_logs_no_error(home):
    with pytest.raises(SystemExit), logs.kept("rm", NOW) as path:
        raise SystemExit(0)
    assert "ERROR" not in path.read_text(encoding="utf-8")


def test_two_in_one_second_get_two_files_and_every_log_is_kept(home):
    with logs.kept("index", NOW) as first, logs.kept("index", NOW) as second:
        logs.attach("uvicorn")
    assert (first.name, second.name) == ("index-2026-10-23_210509.log", "index-2026-10-23_210509-2.log")
    older = [f"index-2026-10-0{day}_120000.log" for day in range(1, 10)]
    for name in older:
        (home / "logs" / name).write_text("x", encoding="utf-8")
    with logs.kept("index", NOW):
        pass
    assert len(list((home / "logs").glob("index-*.log"))) == len(older) + 3
    logs.attach("uvicorn")  # no command running: nothing to attach


def test_a_log_file_is_claimed_at_once(home, monkeypatch):
    # Two commands started in the same second each found the name free, and wrote into one file.
    taken = home / "logs" / "index-2026-10-23_210509.log"
    taken.parent.mkdir(parents=True)
    taken.write_text("another command's\n", encoding="utf-8")
    monkeypatch.setattr(pathlib.Path, "exists", lambda self: False)  # as a name looks free just before it is taken
    with logs.kept("index", NOW) as path:
        pass
    assert path.name == "index-2026-10-23_210509-2.log"
    assert taken.read_text(encoding="utf-8") == "another command's\n"


def test_commands_that_change_or_serve_the_index_keep_logs(notes, home, capsys):
    cli.main(["add", str(notes)])
    cli.main(["index"])
    cli.main(["search", "anything"])
    found = sorted(p.name.split("-20")[0] for p in (home / "logs").glob("*.log"))
    assert found == ["add", "index"]
