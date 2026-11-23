class SiftError(Exception):
    message: str
    remedy: str | None

    def __init__(self, message: str, remedy: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.remedy = remedy


class InvalidRequestError(SiftError):
    pass


class NotFoundError(SiftError):
    pass


class StoreUnavailableError(SiftError):
    pass


class ModelUnavailableError(SiftError):
    pass
