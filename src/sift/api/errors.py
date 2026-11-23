from typing import override


class SiftError(Exception):
    message: str
    remedy: str

    def __init__(self, message: str, remedy: str) -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy

    @override
    def __reduce__(self) -> tuple[type["SiftError"], tuple[str, str]]:
        return type(self), (self.message, self.remedy)


class InvalidRequestError(SiftError):
    pass


class NotFoundError(SiftError):
    pass


class StoreUnavailableError(SiftError):
    pass


class ModelUnavailableError(SiftError):
    pass
