import socket
import ssl
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from sift.api.errors import InvalidRequestError, ModelUnavailableError, Problem, StoreUnavailableError
from sift.api.events import Event, Lending, Phase, PhaseDone, PhaseStarted
from sift.api.serving import ALL_INTERFACES, WORKER_PORT, Served, ServeWorker, WorkerAddress
from sift.inference.loading import Deferred
from sift.interfaces.server.serving import bound
from sift.interfaces.worker.keys import worker_key
from sift.interfaces.worker.serving import LEND, Lend, WorkerServer, lend, lent
from sift.settings import Settings
from tests.models import Hashed
from tests.serving import VERSION
from tests.workers import LOOPBACK, QUICK, offered, working

HERE = ServeWorker(port=0, host=LOOPBACK)


def unheard(_event: Event) -> None:
    return None


def quick(models: Hashed | None = None) -> Lend:
    return replace(QUICK, models=offered(models or Hashed()), machine=lambda: "workstation")


def on_the_loopback(port: int, _host: str) -> socket.socket:
    return bound(port, LOOPBACK)


def test_the_worker_loads_its_models_then_says_where_and_with_which_key_it_is_reached(settings: Settings) -> None:
    models, ran, heard = Hashed(), list[WorkerServer](), list[Event]()
    how = replace(quick(models), run=ran.append, bind=on_the_loopback)
    served = lend(settings, ServeWorker(port=0), VERSION, heard.append, how)
    (running,) = ran
    key = worker_key(settings.home)
    assert heard == [PhaseStarted(Phase.LOAD, None), PhaseDone(2), Lending("workstation", running.port, key.key)]
    assert (served, models.prepared, models.ranker_prepared) == (
        Served(f"https://workstation:{running.port}"),
        [True],
        [True],
    )
    config = running.server.config
    assert (config.ssl_certfile, config.ssl_keyfile) == (str(key.certificate), str(key.private_key))


def test_the_worker_answers_on_every_interface_of_its_machine_at_its_own_port_unless_told() -> None:
    assert (ServeWorker().host, ServeWorker().port, socket.inet_aton(ALL_INTERFACES)) == (
        ALL_INTERFACES,
        WORKER_PORT,
        bytes(4),
    )


def test_the_worker_answers_where_it_is_told_and_says_that_address(settings: Settings) -> None:
    hosts, heard = list[str](), list[Event]()

    def bound(running: WorkerServer) -> None:
        hosts.append(cast("tuple[str, int]", running.socket.getsockname())[0])

    served = lend(settings, HERE, VERSION, heard.append, replace(quick(), run=bound))
    assert (hosts, served.url.startswith(f"https://{LOOPBACK}:")) == ([LOOPBACK], True)
    assert isinstance(heard[-1], Lending)
    assert heard[-1].machine == LOOPBACK


def test_the_worker_serves_its_own_machine_s_models_even_where_it_would_send_them_elsewhere(tmp_path: Path) -> None:
    elsewhere = Settings(tmp_path, None, tmp_path, embed_remote=WorkerAddress("elsewhere", WORKER_PORT, "t", "p"))
    own = LEND.models(elsewhere)
    assert isinstance(own, Deferred)
    assert (own.model, own.reranker, own.built) == (elsewhere.embed_model, elsewhere.rerank_model, None)


def test_ctrl_c_while_the_worker_serves_stops_it_and_lets_another_start(settings: Settings) -> None:
    def interrupted(_running: WorkerServer) -> None:
        raise KeyboardInterrupt

    how = replace(quick(), run=interrupted)
    for _ in range(2):
        assert lend(settings, HERE, VERSION, unheard, how).url.startswith(f"https://{LOOPBACK}:")


def test_a_server_error_comes_out_as_it_is(settings: Settings) -> None:
    def failing(_running: WorkerServer) -> None:
        raise OSError("the network went away")

    with pytest.raises(OSError, match="the network went away"):
        lend(settings, HERE, VERSION, unheard, replace(quick(), run=failing))


def test_one_worker_lends_a_home_s_models_at_a_time(settings: Settings) -> None:
    with (
        lent(settings, HERE, VERSION, unheard, quick()),
        pytest.raises(StoreUnavailableError) as refused,
    ):
        lend(settings, HERE, VERSION, unheard, quick())
    assert (refused.value.code, "sift worker" in refused.value.message) == (Problem.BUSY, True)


def test_a_port_another_program_holds_is_refused(settings: Settings) -> None:
    with socket.create_server((LOOPBACK, 0)) as taken:
        port = cast("tuple[str, int]", taken.getsockname())[1]
        with pytest.raises(InvalidRequestError, match=f"port {port} cannot be had"):
            lend(settings, ServeWorker(port=port, host=LOOPBACK), VERSION, unheard, quick())


def test_a_worker_whose_private_key_cannot_be_read_is_refused_saying_to_make_a_new_one(settings: Settings) -> None:
    worker_key(settings.home).private_key.write_bytes(b"no key")
    with pytest.raises(StoreUnavailableError, match=r"the worker's key in .*worker cannot be read: ") as refused:
        lend(settings, HERE, VERSION, unheard, quick())
    assert refused.value.remedy == "delete that folder: the worker makes a new key, to copy to the machines that use it"


def test_the_worker_takes_no_tls_older_than_1_2(settings: Settings) -> None:
    with working(settings, Hashed()) as worker:
        tls = worker.running.server.config.ssl
        assert tls is not None
        assert tls.minimum_version == ssl.TLSVersion.TLSv1_2


def test_a_worker_whose_model_is_named_at_a_branch_is_refused_before_any_model_is_loaded(settings: Settings) -> None:
    heard = list[Event]()
    branch = replace(settings, embed_model="owner/model@main")
    how = replace(QUICK, models=LEND.models, bind=on_the_loopback)
    with pytest.raises(ModelUnavailableError, match="names main, no whole commit"):
        lend(branch, HERE, VERSION, heard.append, how)
    assert heard == []
