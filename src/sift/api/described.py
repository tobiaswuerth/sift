from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Help:
    text: str


@dataclass(frozen=True, slots=True)
class AbsolutePath:
    pass


@dataclass(frozen=True, slots=True)
class FileOrId:
    pass


@dataclass(frozen=True, slots=True)
class Composed:
    pass


@dataclass(frozen=True, slots=True)
class Bounds:
    low: int
    high: int
    grouped: bool = True

    def shown(self, number: int) -> str:
        return f"{number:,}" if self.grouped else str(number)


@dataclass(frozen=True, slots=True)
class Length:
    low: int
    high: int


@dataclass(frozen=True, slots=True)
class Within:
    low: float
    high: float


MOST_LISTED = 1_000
PATH_LENGTH = Length(1, 32_767)
NAME_LENGTH = Length(1, 200)
FIELD_LENGTH = Length(1, 1_000)
TEXT_LENGTH = Length(1, 10_000)
LISTED = Bounds(0, MOST_LISTED)
BUDGET_SECONDS = Within(0.001, 86_400.0)
