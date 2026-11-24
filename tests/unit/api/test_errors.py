import copy

import pytest

from sift.api.errors import (
    InvalidRequestError,
    ModelUnavailableError,
    NotFoundError,
    Problem,
    SiftError,
    StoreUnavailableError,
)


def test_an_error_says_what_happened_and_what_to_do() -> None:
    error = NotFoundError("no collection named 'notes'", "the collections are: papers")
    assert (str(error), error.message, error.remedy) == (
        "no collection named 'notes'",
        "no collection named 'notes'",
        "the collections are: papers",
    )


@pytest.mark.parametrize("kind", [SiftError, InvalidRequestError, NotFoundError, StoreUnavailableError])
def test_an_error_is_copied_whole_by_the_protocol_processes_pass_it_with(kind: type[SiftError]) -> None:
    copied = copy.deepcopy(kind("what", "remedy"))
    assert (type(copied), copied.message, copied.remedy) == (kind, "what", "remedy")


@pytest.mark.parametrize("kind", [InvalidRequestError, NotFoundError, StoreUnavailableError])
def test_every_error_of_sift_is_a_sift_error(kind: type[SiftError]) -> None:
    assert issubclass(kind, SiftError)


@pytest.mark.parametrize(
    "kind, code",
    [
        (InvalidRequestError, Problem.INVALID_REQUEST),
        (NotFoundError, Problem.NOT_FOUND),
        (StoreUnavailableError, Problem.STORE_UNAVAILABLE),
        (ModelUnavailableError, Problem.MODEL_UNAVAILABLE),
    ],
)
def test_an_error_carries_a_code_a_surface_may_branch_on_its_kind_s_unless_given_one(
    kind: type[SiftError], code: Problem
) -> None:
    assert kind("what", "remedy").code == code
    given = copy.deepcopy(kind("what", "remedy", code=Problem.STORE_DAMAGED))
    assert (type(given), given.code, given.message) == (kind, Problem.STORE_DAMAGED, "what")


def test_a_refused_request_keeps_its_field_and_code_when_copied() -> None:
    copied = copy.deepcopy(InvalidRequestError("what", "remedy", "k", code=Problem.INVALID_REQUEST))
    assert (copied.field, copied.code) == ("k", Problem.INVALID_REQUEST)
