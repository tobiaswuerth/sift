import io
import json
import signal
import threading

import pytest

from sift.api.events import Event, Listening, Noted
from sift.interfaces.cli.commands.serving import opened_once_listening
from sift.interfaces.cli.main import Host, in_browser, run
from sift.interfaces.server.endpoint import found
from sift.settings import Settings
from tests.serving import running
from tests.surface.browser import BrowserAskedError
from tests.surface.commands import ran
from tests.surface.test_serve_cli import answers
from tests.waits import until


def opened_by(settings: Settings, opened: list[str], *argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = run(argv, settings, out, err, Host(browse=opened.append))
    return code, out.getvalue(), err.getvalue()


def test_sift_ui_opens_the_page_of_the_home_process_that_runs(settings: Settings) -> None:
    opened = list[str]()
    with running(settings) as server:
        code, out, err = opened_by(settings, opened, "ui")
        listed = ran(settings, "ui", "--no-browser", "--json")
    url = f"http://127.0.0.1:{server.endpoint.port}/ui"
    assert (code, out, err, opened) == (0, "", f"   ui  {url}\n", [url])
    assert (listed[0], json.loads(listed[1])) == (0, {"url": url, "served": False})


def test_sift_ui_serves_the_home_here_when_none_runs_and_opens_its_page_once_it_answers(settings: Settings) -> None:
    opened, answered = list[str](), list[bool]()

    def interrupted_once_answering() -> None:
        try:
            answered.append(until(lambda: answers(settings)))
        finally:
            signal.raise_signal(signal.SIGINT)

    threading.Thread(target=interrupted_once_answering, daemon=True).start()
    code, _, err = opened_by(settings, opened, "ui")
    assert (code, answered, len(opened), found(settings)) == (0, [True], 1, None)
    assert opened[0].startswith("http://127.0.0.1:")
    assert opened[0].endswith("/ui")
    assert err.endswith(f" done  stopped serving {opened[0]}\n")


def test_only_the_line_saying_where_the_home_process_listens_opens_the_page() -> None:
    heard, opened = list[Event](), list[str]()
    said = opened_once_listening(heard.append, opened.append)
    said(Noted("loading"))
    said(Listening("http://127.0.0.1:9", "/home"))
    assert (heard, opened) == ([Noted("loading"), Listening("http://127.0.0.1:9", "/home")], ["http://127.0.0.1:9/ui"])


def test_the_page_opens_in_a_new_tab_of_the_system_s_browser() -> None:
    with pytest.raises(BrowserAskedError) as asked:
        in_browser("http://127.0.0.1:9/ui", "tests.surface.browser")
    assert asked.value.args == ("http://127.0.0.1:9/ui",)
