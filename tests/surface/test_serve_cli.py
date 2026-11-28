import http.client
import signal
import threading
from pathlib import Path

from sift.interfaces.console.progress import STOP_HINT
from sift.interfaces.server.endpoint import found, own_endpoint
from sift.settings import Settings
from tests.surface import process
from tests.surface.commands import ran
from tests.waits import until


def answers(settings: Settings) -> bool:
    endpoint = found(settings)
    if endpoint is None:
        return False
    connection = http.client.HTTPConnection("127.0.0.1", endpoint.port, timeout=10)
    try:
        connection.request("POST", "/api/status", b"{}", {"Authorization": f"Bearer {endpoint.token}"})
        return connection.getresponse().status == http.client.OK
    except OSError:
        return False
    finally:
        connection.close()


def interrupted_once_answering(settings: Settings, answered: list[bool]) -> None:
    try:
        answered.append(until(lambda: answers(settings)))
    finally:
        signal.raise_signal(signal.SIGINT)


def test_ctrl_c_stops_sift_serve_saying_so_and_leaves_nothing_behind(settings: Settings) -> None:
    answered: list[bool] = []
    threading.Thread(target=interrupted_once_answering, args=(settings, answered), daemon=True).start()
    code, out, err = ran(settings, "serve", "--port", "0")
    assert answered == [True], "sift serve did not answer within 30 s"
    url = err.split("  ")[1]
    started = f"serve  {url}  {settings.home}\n stop  {STOP_HINT}\n"
    assert (code, out, err) == (0, "", f"{started} done  stopped serving {url}\n")
    assert (found(settings), own_endpoint(settings).exists()) == (None, False)
    said = next((settings.home / "logs").glob("serve-*.log")).read_text(encoding="utf-8")
    assert said.endswith(" INFO sift: ended with exit code 0\n")
    assert " INFO uvicorn.error: Started server process [" in said


def test_ctrl_c_stops_the_home_process_and_takes_its_endpoint_away(tmp_path: Path) -> None:
    home = tmp_path / "home"
    settings = process.settings_of(home)
    with process.served(home) as running:
        assert own_endpoint(settings).exists()
    assert (running.returncode, own_endpoint(settings).exists(), found(settings)) == (0, False, None)


def test_serve_over_stdio_takes_no_port_and_says_so(settings: Settings) -> None:
    code, out, err = ran(settings, "serve", "--stdio", "--port", "5")
    assert (code, out, err.startswith("error  --port: sift serve --stdio talks to the home process")) == (2, "", True)
