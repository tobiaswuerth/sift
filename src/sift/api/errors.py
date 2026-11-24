import errno
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import ClassVar, override


class Problem(StrEnum):
    INVALID_REQUEST = "invalid_request"
    NOT_FOUND = "not_found"
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

    def __init__(self, message: str, remedy: str, *, code: Problem | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy
        self.code = type(self).usual if code is None else code

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return _rebuilt, (type(self), self.message, self.remedy, self.code, None)


class InvalidRequestError(SiftError):
    field: str | None

    def __init__(self, message: str, remedy: str, field: str | None = None, *, code: Problem | None = None) -> None:
        super().__init__(message, remedy, code=code)
        self.field = field

    @override
    def __reduce__(self) -> tuple[object, ...]:
        return _rebuilt, (type(self), self.message, self.remedy, self.code, self.field)


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


OS_REMEDIES: Mapping[int, str] = MappingProxyType(
    {
        errno.ENOENT: "check that the path exists",
        errno.EACCES: "check that you may read and write the path",
        errno.ENOSPC: "free space on the disk",
    }
)
OS_REMEDY = "check the path and its permissions"


def system_failure(error: OSError) -> SystemFailureError:
    return SystemFailureError(str(error), OS_REMEDIES.get(error.errno or 0, OS_REMEDY))


def _rebuilt(kind: type[SiftError], message: str, remedy: str, code: Problem, field: str | None) -> SiftError:
    if issubclass(kind, InvalidRequestError):
        return kind(message, remedy, field, code=code)
    return kind(message, remedy, code=code)
