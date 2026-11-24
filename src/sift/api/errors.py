from enum import StrEnum
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


def _rebuilt(kind: type[SiftError], message: str, remedy: str, code: Problem, field: str | None) -> SiftError:
    if issubclass(kind, InvalidRequestError):
        return kind(message, remedy, field, code=code)
    return kind(message, remedy, code=code)
