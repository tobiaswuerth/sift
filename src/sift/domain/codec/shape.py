from dataclasses import dataclass

FORMAT = 1
MOST_RANGE_SHIFT = 8
MOST_BLOCK = 1 << 12
MOST_LENGTHS_SHIFT = 20


@dataclass(frozen=True, slots=True)
class Shape:
    range_shift: int = 7
    block: int = 4096
    long: int = 4096
    lengths_shift: int = 10

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (self.range_shift, self.block, self.long, self.lengths_shift)):
            raise TypeError("a shape is whole numbers")
        if not 0 <= self.range_shift <= MOST_RANGE_SHIFT:
            raise ValueError(
                f"a range holds 2**range_shift places, at most 2**{MOST_RANGE_SHIFT}: its counts are bytes"
            )
        if min(self.block, self.long, self.lengths_shift) < 1:
            raise ValueError("blocks, long lists and blocks of lengths hold one entry at least")
        if max(self.block, self.long) > MOST_BLOCK or self.lengths_shift > MOST_LENGTHS_SHIFT:
            raise ValueError(
                f"blocks hold at most {MOST_BLOCK:,} entries and blocks of lengths 2**{MOST_LENGTHS_SHIFT}"
            )


def shaped(shape: Shape) -> tuple[int, int, int, int]:
    return shape.range_shift, shape.block, shape.long, shape.lengths_shift


SHAPE = Shape()
