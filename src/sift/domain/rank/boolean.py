from collections.abc import Mapping

from sift.domain.arrays import Mask, either, every
from sift.domain.query.expressions import And, Expression, Leaf, Not, Or, Phrase, Prefix, Term

type Counting = tuple[tuple[Leaf, Mask], ...]

UNSEARCHED = "nothing is searched for: a query that matches nothing is answered before it gets here"


def counted(expression: Expression, present: Mapping[Leaf, Mask]) -> tuple[Mask, Counting]:
    match expression:
        case Term() | Phrase() | Prefix():
            return present[expression], ((expression, present[expression]),)
        case Not(kept, dropped):
            held, counting = counted(kept, present)
            held = held & ~counted(dropped, present)[0]
            return held, tuple((leaf, where & held) for leaf, where in counting)
        case Or(parts):
            found = [counted(part, present) for part in parts]
            held = either([part for part, _ in found])
            return held, tuple(each for _, counting in found for each in counting)
        case And(parts):
            found = [counted(part, present) for part in parts]
            held = every([part for part, _ in found])
            return held, tuple((leaf, where & held) for _, counting in found for leaf, where in counting)
        case _:
            raise ValueError(UNSEARCHED)


def may_hold(expression: Expression, present: Mapping[Leaf, Mask]) -> Mask:
    match expression:
        case Term() | Phrase() | Prefix():
            return present[expression]
        case Not(kept, _):
            return may_hold(kept, present)
        case Or(parts):
            return either([may_hold(part, present) for part in parts])
        case And(parts):
            return every([may_hold(part, present) for part in parts])
        case _:
            raise ValueError(UNSEARCHED)
