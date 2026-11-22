from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Term:
    text: str


@dataclass(frozen=True, slots=True)
class Phrase:
    tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Prefix:
    text: str


@dataclass(frozen=True, slots=True)
class And:
    parts: "tuple[Expression, ...]"


@dataclass(frozen=True, slots=True)
class Or:
    parts: "tuple[Expression, ...]"


@dataclass(frozen=True, slots=True)
class Not:
    kept: "Expression"
    dropped: "Expression"


@dataclass(frozen=True, slots=True)
class Nothing:
    pass


type Leaf = Term | Phrase | Prefix
type Expression = Leaf | And | Or | Not | Nothing

NOTHING = Nothing()


def leaves(expression: Expression) -> tuple[Leaf, ...]:
    match expression:
        case Term() | Phrase() | Prefix():
            return (expression,)
        case And(parts) | Or(parts):
            return tuple(leaf for part in parts for leaf in leaves(part))
        case Not(kept, dropped):
            return (*leaves(kept), *leaves(dropped))
        case _:
            return ()
