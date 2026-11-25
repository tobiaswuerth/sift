import copy
import errno

import pytest

from sift.api.errors import (
    CODE_KINDS,
    OS_REMEDY,
    InvalidRequestError,
    ModelUnavailableError,
    NotFoundError,
    Problem,
    SiftError,
    StoreUnavailableError,
    system_failure,
)


def test_an_error_says_what_happened_and_what_to_do() -> None:
    error = NotFoundError("no collection named 'notes'", "the collections are: papers")
    assert (str(error), error.message, error.remedy) == (
        "no collection named 'notes'",
        "no collection named 'notes'",
        "the collections are: papers",
    )


@pytest.mark.parametrize("kind", [InvalidRequestError, NotFoundError, StoreUnavailableError])
def test_an_error_is_copied_whole_by_the_protocol_processes_pass_it_with(kind: type[SiftError]) -> None:
    copied = copy.deepcopy(kind("what", "remedy"))
    assert (type(copied), copied.message, copied.remedy) == (kind, "what", "remedy")


@pytest.mark.parametrize("kind", [InvalidRequestError, NotFoundError, StoreUnavailableError])
def test_every_error_of_sift_is_a_sift_error(kind: type[SiftError]) -> None:
    assert issubclass(kind, SiftError)


@pytest.mark.parametrize(
    "kind, code, other",
    [
        (InvalidRequestError, Problem.INVALID_REQUEST, Problem.AMBIGUOUS),
        (NotFoundError, Problem.NOT_FOUND, Problem.UNKNOWN_NOTE),
        (StoreUnavailableError, Problem.STORE_UNAVAILABLE, Problem.STORE_DAMAGED),
        (ModelUnavailableError, Problem.MODEL_UNAVAILABLE, Problem.MODEL_UNAVAILABLE),
    ],
)
def test_an_error_carries_a_code_a_surface_may_branch_on_its_kind_s_unless_given_one(
    kind: type[SiftError], code: Problem, other: Problem
) -> None:
    assert kind("what", "remedy").code == code
    given = copy.deepcopy(kind("what", "remedy", code=other))
    assert (type(given), given.code, given.message) == (kind, other, "what")


def test_a_refused_request_keeps_its_field_and_code_when_copied() -> None:
    copied = copy.deepcopy(InvalidRequestError("what", "remedy", "k", code=Problem.INVALID_REQUEST))
    assert (copied.field, copied.code) == ("k", Problem.INVALID_REQUEST)


def test_every_code_belongs_to_one_kind_and_an_error_of_another_kind_is_a_bug() -> None:
    assert set(CODE_KINDS) == set(Problem)
    assert {kind.usual for kind in set(CODE_KINDS.values())} <= set(CODE_KINDS)
    assert [code for code, kind in CODE_KINDS.items() if kind(code.value, "r", code=code).code is not code] == []
    with pytest.raises(TypeError, match="InvalidRequestError cannot carry the code busy, a StoreUnavailableError's"):
        InvalidRequestError("m", "r", code=Problem.BUSY)
    assert (CODE_KINDS[Problem.AMBIGUOUS], CODE_KINDS[Problem.TAKEN]) == (InvalidRequestError, InvalidRequestError)


@pytest.mark.parametrize(
    "error, said, remedy",
    [
        (
            FileExistsError(errno.EEXIST, "Cannot create a file when that file already exists", "C:\\Users\\a b\\x"),
            "Cannot create a file when that file already exists: C:\\Users\\a b\\x",
            "move what is there away, or choose another path",
        ),
        (
            NotADirectoryError(errno.ENOTDIR, "Not a directory", "/home/a/file/x"),
            "Not a directory: /home/a/file/x",
            "check that each folder of the path is a folder, not a file",
        ),
        (
            OSError(errno.EINVAL, "Invalid argument", "a:b?"),
            "Invalid argument: a:b?",
            'check that the path names no character the system refuses, such as : * ? " < > |',
        ),
        (
            OSError(errno.ENAMETOOLONG, "File name too long", "x" * 300),
            f"File name too long: {'x' * 300}",
            "choose a shorter path",
        ),
        (
            OSError(errno.EROFS, "Read-only file system", "/mnt/x"),
            "Read-only file system: /mnt/x",
            "choose a place that may be written",
        ),
        (
            OSError(errno.EXDEV, "Invalid cross-device link", "/a", None, "/b"),
            "Invalid cross-device link: /a -> /b",
            OS_REMEDY,
        ),
        (OSError(errno.ENOSPC, "No space left on device"), "No space left on device", "free space on the disk"),
        (OSError(errno.EIO, "Input/output error", b"/raw\xff"), "Input/output error: /raw\\xff", OS_REMEDY),
        (OSError("no number at all"), "no number at all", OS_REMEDY),
    ],
    ids=["exists", "no folder", "invalid", "too long", "read-only", "two paths", "no path", "bytes", "no errno"],
)
def test_an_error_of_the_system_says_what_it_said_of_which_path_and_what_to_do(
    error: OSError, said: str, remedy: str
) -> None:
    found = system_failure(error)
    assert (found.message, found.remedy) == (said, remedy)
