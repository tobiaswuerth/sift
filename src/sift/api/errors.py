import errno
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, cast, override


class Problem(StrEnum):
    INVALID_REQUEST = "invalid_request"
    AMBIGUOUS = "ambiguous"
    TAKEN = "taken"
    TOO_MANY_WORDS = "too_many_words"
    WIDENED_ENOUGH = "widened_enough"
    FIELD_NOT_HELD = "field_not_held"
    NOT_FOUND = "not_found"
    FOLDER_MISSING = "folder_missing"
    UNKNOWN_COLLECTION = "unknown_collection"
    UNKNOWN_DOCUMENT = "unknown_document"
    UNKNOWN_NOTE = "unknown_note"
    INACTIVE_COLLECTION = "inactive_collection"
    STORE_UNAVAILABLE = "store_unavailable"
    STORE_MISSING = "store_missing"
    STORE_DAMAGED = "store_damaged"
    STORE_NEWER = "store_newer"
    SQLITE_TOO_OLD = "sqlite_too_old"
    KEYWORDS_REMADE = "keywords_remade"
    NOT_EMBEDDED = "not_embedded"
    OTHER_MODEL = "other_model"
    OUT_OF_TIME = "out_of_time"
    BUSY = "busy"
    MODEL_UNAVAILABLE = "model_unavailable"
    UNAUTHORIZED = "unauthorized"
    OS = "os"
    STOPPED = "stopped"
    INTERNAL = "internal"


class SiftError(Exception):
    usual: ClassVar[Problem] = Problem.INVALID_REQUEST
    message: str
    remedy: str
    code: Problem
    field: str | None

    def __init__(self, message: str, remedy: str, field: str | None = None, *, code: Problem | None = None) -> None:
        super().__init__(message)
        chosen = type(self).usual if code is None else code
        kind = CODE_KINDS[chosen]
        if not isinstance(self, kind):
            raise TypeError(f"{type(self).__name__} cannot carry the code {chosen.value}, a {kind.__name__}'s")
        self.message = message
        self.remedy = remedy
        self.code = chosen
        self.field = field

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return _rebuilt, (type(self), self.message, self.remedy, self.code, self.field)


class InvalidRequestError(SiftError):
    usual: ClassVar[Problem] = Problem.INVALID_REQUEST


class NotFoundError(SiftError):
    usual: ClassVar[Problem] = Problem.NOT_FOUND


class StoreUnavailableError(SiftError):
    usual: ClassVar[Problem] = Problem.STORE_UNAVAILABLE


class ModelUnavailableError(SiftError):
    usual: ClassVar[Problem] = Problem.MODEL_UNAVAILABLE


class InternalError(SiftError):
    usual: ClassVar[Problem] = Problem.INTERNAL


class SystemFailureError(SiftError):
    usual: ClassVar[Problem] = Problem.OS


class JobStoppedError(SiftError):
    usual: ClassVar[Problem] = Problem.STOPPED


CODE_KINDS: Mapping[Problem, type[SiftError]] = MappingProxyType(
    {
        Problem.INVALID_REQUEST: InvalidRequestError,
        Problem.AMBIGUOUS: InvalidRequestError,
        Problem.TAKEN: InvalidRequestError,
        Problem.TOO_MANY_WORDS: InvalidRequestError,
        Problem.WIDENED_ENOUGH: InvalidRequestError,
        Problem.FIELD_NOT_HELD: InvalidRequestError,
        Problem.INACTIVE_COLLECTION: InvalidRequestError,
        Problem.UNAUTHORIZED: InvalidRequestError,
        Problem.NOT_FOUND: NotFoundError,
        Problem.FOLDER_MISSING: NotFoundError,
        Problem.UNKNOWN_COLLECTION: NotFoundError,
        Problem.UNKNOWN_DOCUMENT: NotFoundError,
        Problem.UNKNOWN_NOTE: NotFoundError,
        Problem.STORE_UNAVAILABLE: StoreUnavailableError,
        Problem.STORE_MISSING: StoreUnavailableError,
        Problem.STORE_DAMAGED: StoreUnavailableError,
        Problem.STORE_NEWER: StoreUnavailableError,
        Problem.SQLITE_TOO_OLD: StoreUnavailableError,
        Problem.KEYWORDS_REMADE: StoreUnavailableError,
        Problem.NOT_EMBEDDED: StoreUnavailableError,
        Problem.OTHER_MODEL: StoreUnavailableError,
        Problem.OUT_OF_TIME: StoreUnavailableError,
        Problem.BUSY: StoreUnavailableError,
        Problem.MODEL_UNAVAILABLE: ModelUnavailableError,
        Problem.OS: SystemFailureError,
        Problem.STOPPED: JobStoppedError,
        Problem.INTERNAL: InternalError,
    }
)
OS_REMEDIES: Mapping[int, str] = MappingProxyType(
    {
        errno.ENOENT: "check that the path exists",
        errno.EACCES: "check that you may read and write the path",
        errno.EPERM: "check that you may read and write the path",
        errno.ENOSPC: "free space on the disk",
        errno.EEXIST: "move what is there away, or choose another path",
        errno.ENOTDIR: "check that each folder of the path is a folder, not a file",
        errno.EISDIR: "give a file, not a folder",
        errno.EINVAL: 'check that the path names no character the system refuses, such as : * ? " < > |',
        errno.ENAMETOOLONG: "choose a shorter path",
        errno.EROFS: "choose a place that may be written",
    }
)
OS_REMEDY = "check the path and its permissions"
PATHS_JOINED = " -> "


def _path_text(path: object) -> str:
    return path.decode("utf-8", "backslashreplace") if isinstance(path, bytes) else str(path)


def _system_said(error: OSError) -> str:
    said = error.strerror
    if said is None:
        return str(error)
    named = cast("tuple[object, object]", (error.filename, error.filename2))
    paths = [_path_text(path) for path in named if path is not None]
    return f"{said}: {PATHS_JOINED.join(paths)}" if paths else said


def system_failure(error: OSError) -> SystemFailureError:
    return SystemFailureError(_system_said(error), OS_REMEDIES.get(error.errno or 0, OS_REMEDY))


def _rebuilt(kind: type[SiftError], message: str, remedy: str, code: Problem, field: str | None) -> SiftError:
    return kind(message, remedy, field, code=code)


@contextmanager
def field_at_fault(field: str) -> Generator[None]:
    try:
        yield
    except (InvalidRequestError, NotFoundError) as error:
        if error.field is None:
            error.field = field
        raise
