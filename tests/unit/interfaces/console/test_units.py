import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.interfaces.console.units import duration, rate, size


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


@pytest.mark.parametrize(
    "seconds, shown",
    [
        (0, "0.0 s"),
        (0.44, "0.4 s"),
        (9.94, "9.9 s"),
        (9.96, "10 s"),
        (41.4, "41 s"),
        (59.6, "1 m 0 s"),
        (250, "4 m 10 s"),
        (3725, "1 h 2 m"),
        (40 * 3600, "40 h 0 m"),
    ],
)
def test_a_duration_reads_in_the_units_that_keep_it_short(seconds: float, shown: str) -> None:
    assert duration(seconds) == shown


@pytest.mark.parametrize(
    "done, seconds, shown", [(5, 2.0, "2.5/s"), (310, 1.0, "310/s"), (12_345, 1.0, "12,345/s"), (3, 0.0, "0.0/s")]
)
def test_a_rate_is_whole_from_ten_a_second_on(done: int, seconds: float, shown: str) -> None:
    assert rate(done, seconds) == shown
