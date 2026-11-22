import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.interfaces.console.units import size


@pytest.mark.parametrize(
    "stored, shown",
    [
        (0, "0 B"),
        (999, "999 B"),
        (1_000, "1.0 kB"),
        (1_449, "1.4 kB"),
        (9_960, "10 kB"),
        (999_600, "1.0 MB"),
        (412_300_000, "412 MB"),
        (1_234_567_890_123, "1.2 TB"),
        (7 * 10**18, "7000 PB"),
    ],
)
def test_a_size_reads_in_the_unit_that_keeps_it_short(stored: int, shown: str) -> None:
    assert size(stored) == shown


@given(st.integers(min_value=0, max_value=10**17))
def test_a_size_has_at_most_three_digits_before_its_unit(stored: int) -> None:
    number, _ = size(stored).split(" ")
    assert len(number.replace(".", "")) <= 3
