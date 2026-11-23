import copy

import pytest

from sift.api.errors import InvalidRequestError, NotFoundError, SiftError, StoreUnavailableError


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
