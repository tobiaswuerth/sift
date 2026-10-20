"""Which models a process uses: those of this machine, or a `sift worker` set in SIFT_EMBED_REMOTE. Both offer the
same interface, so the rest of sift never asks which it has."""

import inspect

import pytest

from sift import models
from sift.local import Local
from sift.models import Models
from sift.remote import Remote


def test_without_a_remote_the_models_are_this_machines(fake_models):
    assert models.default() is fake_models
    assert models.default() is models.default()  # chosen once


def test_with_a_remote_the_models_are_the_workers(fake_models, monkeypatch, configure):
    configure(remote="https://t.p@workstation:8766")
    worker = models.default()
    assert isinstance(worker, Remote)
    assert worker.address == "https://workstation:8766"
    assert models.default() is worker  # one per worker, so its answers are kept


MEMBERS = sorted({*Models.__annotations__, *(n for n in vars(Models) if not n.startswith("_"))})


@pytest.mark.parametrize("make", [lambda make_local: make_local(), lambda make_local: Remote("https://t.p@pc:8766")])
def test_both_kinds_offer_the_same_interface(make_local, make):
    # Looked up statically: asking a Remote for its name asks the worker.
    models = make(make_local)
    assert [m for m in MEMBERS if inspect.getattr_static(models, m, None) is None] == []
    assert len(MEMBERS) == 9


def test_a_worker_manages_its_own_gpu():
    worker = Remote("https://t.p@pc:8766")
    assert worker.release_idle() is False  # nothing on this machine to hand back
    worker.prepare(lambda: pytest.fail("a worker sizes its batches itself"))


def test_choosing_the_models_loads_none_yet(monkeypatch, configure):
    # Loading a model takes seconds; that waits until something is embedded or reranked.
    configure(remote=None)
    monkeypatch.setattr(models, "_local", models._local.__wrapped__)  # not cached: build a fresh one
    chosen = models.default()
    assert isinstance(chosen, Local)
    assert chosen._loaded == {}
