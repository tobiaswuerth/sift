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
        (0, "0 ms"),
        (0.0004, "0 ms"),
        (0.044, "44 ms"),
        (0.44, "440 ms"),
        (0.9996, "1.0 s"),
        (9.94, "9.9 s"),
        (9.96, "10 s"),
        (41.4, "41 s"),
        (59.6, "1 m 0 s"),
        (250, "4 m 10 s"),
        (3725, "1 h 2 m"),
        (23 * 3600 + 59 * 60, "23 h 59 m"),
        (40 * 3600, "1 d 16 h"),
        (198 * 3600, "8 d 6 h"),
    ],
)
def test_a_duration_reads_in_the_units_that_keep_it_short(seconds: float, shown: str) -> None:
    assert duration(seconds) == shown


@pytest.mark.parametrize(
    "done, seconds, unit, shown",
    [
        (5, 2.0, "files", "2.5 files/s"),
        (310, 1.0, "files", "310 files/s"),
        (12_345, 1.0, "chunks", "12,345 chunks/s"),
        (400, 1.0, "MB", "400 MB/s"),
        (3, 0.0, "files", "0.0 files/s"),
    ],
)
def test_a_rate_is_whole_from_ten_a_second_on_and_says_its_unit(
    done: int, seconds: float, unit: str, shown: str
) -> None:
    assert rate(done, seconds, unit) == shown


@pytest.mark.parametrize(
    "done, seconds, shown",
    [
        (9, 10.0, "54 files/min"),
        (84, 78 * 60.0, "1.1 files/min"),
        (13, 6 * 3600.0, "2.2 files/h"),
        (3, 10 * 3600.0, "0.3 files/h"),
        (1, 400 * 3600.0, "0.0 files/h"),
    ],
)
def test_a_rate_below_one_a_second_is_told_by_the_minute_and_below_one_a_minute_by_the_hour(
    done: int, seconds: float, shown: str
) -> None:
    assert rate(done, seconds, "files") == shown
