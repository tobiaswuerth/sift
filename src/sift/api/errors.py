from typing import override


class SiftError(Exception):
    message: str
    remedy: str

    def __init__(self, message: str, remedy: str) -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy

    @override
    def __reduce__(self) -> tuple[type["SiftError"], tuple[str | None, ...]]:
        return type(self), (self.message, self.remedy)


class InvalidRequestError(SiftError):
    field: str | None

    def __init__(self, message: str, remedy: str, field: str | None = None) -> None:
        super().__init__(message, remedy)
        self.field = field

    @override
    def __reduce__(self) -> tuple[type["SiftError"], tuple[str | None, ...]]:
        return type(self), (self.message, self.remedy, self.field)


class NotFoundError(SiftError):
    pass


class StoreUnavailableError(SiftError):
    pass


class ModelUnavailableError(SiftError):
    pass
