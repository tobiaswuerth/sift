import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import pytest

from sift.api.collections import AddCollection, ShowStatus, TitleFrom
from sift.api.described import Help
from sift.api.errors import InvalidRequestError
from sift.api.filters import Filter
from sift.api.indexing import IndexCollections
from sift.api.jobs import ListJobs
from sift.api.search import SearchHybrid, SearchKeywords
from sift.interfaces.cli.arguments import arguments, flags, options, request_of
from sift.interfaces.cli.main import Parser, UsageError
from sift.interfaces.cli.reading import ReadDocuments
from sift.interfaces.cli.written import written

WHOLE = Path("/whole").absolute()


def whole(text: str) -> str:
    return str(WHOLE / text)


def parsed[R](request: type[R], argv: list[str]) -> R:
    parser = Parser(prog="sift add")
    options(parser, request, whole)
    return request_of(request, parser.parse_args(argv))


def test_a_field_without_a_default_is_given_by_its_place_and_the_others_by_name() -> None:
    assert parsed(AddCollection, ["notes"]) == AddCollection(whole("notes"))
    given = ["notes", "--name", "n", "--store", "s", "--title-from", "name", "--nested"]
    assert parsed(AddCollection, given) == AddCollection(
        whole("notes"), name="n", store=whole("s"), title_from=TitleFrom.NAME, nested=True
    )


def test_a_list_is_given_by_its_places_and_may_be_empty_if_it_has_a_default() -> None:
    assert parsed(ShowStatus, []) == ShowStatus()
    assert parsed(IndexCollections, ["a", "b"]) == IndexCollections(("a", "b"))
    assert parsed(ReadDocuments, ["x", "y", "--lines", "3"]) == ReadDocuments(("x", "y"), "3")
    with pytest.raises(UsageError, match="the following arguments are required: ids"):
        parsed(ReadDocuments, [])


@dataclass(frozen=True, slots=True)
class Picked:
    title_from: Annotated[TitleFrom, Help("where titles come from")]


def test_a_choice_may_be_given_by_its_place() -> None:
    assert parsed(Picked, ["name"]) == Picked(TitleFrom.NAME)


@dataclass(frozen=True, slots=True)
class Toggled:
    folder: Annotated[str, Help("a folder")]
    follow: Annotated[bool | None, Help("follow its links")] = None


@dataclass(frozen=True, slots=True)
class ToggledOn:
    follow: Annotated[bool | None, Help("follow its links")] = False


def test_a_yes_or_no_that_may_be_left_out_is_given_by_its_name_or_its_name_with_no() -> None:
    assert parsed(Toggled, ["x"]) == Toggled("x")
    assert parsed(Toggled, ["x", "--follow"]) == Toggled("x", follow=True)
    assert parsed(Toggled, ["x", "--no-follow"]) == Toggled("x", follow=False)
    assert flags(Toggled) == {"folder": "folder", "follow": "--follow"}


def test_a_choice_not_offered_is_refused_with_the_choices_there_are() -> None:
    with pytest.raises(UsageError, match=r"invalid choice: 'title' \(choose from '?heading'?, '?name'?\)"):
        parsed(AddCollection, ["notes", "--title-from", "title"])


def test_each_argument_says_what_it_is_for() -> None:
    parser = argparse.ArgumentParser(prog="sift add")
    options(parser, AddCollection, whole)
    shown = " ".join(parser.format_help().split())
    usage = (
        "sift add [-h] [--name NAME] [--store STORE] [--title-from {heading,name}] [--nested]"
        " [--follow-links | --no-follow-links] [--agents | --no-agents]"
        " [--id-field ID_FIELD] [--title-field TITLE_FIELD] [--text-fields TEXT_FIELDS] folder"
    )
    assert usage in shown
    assert "folder the folder whose files the collection holds" in shown
    assert "--nested add a folder inside a collection" in shown
    assert "it changes that collection" in shown


@dataclass(frozen=True, slots=True)
class Ratio:
    ratio: Annotated[complex, Help("a share")]


@dataclass(frozen=True, slots=True)
class Undescribed:
    name: str


@dataclass(frozen=True, slots=True)
class OnByDefault:
    loud: Annotated[bool, Help("loud")] = True


@dataclass(frozen=True, slots=True)
class OptionalWithAValue:
    name: Annotated[str | None, Help("a name")] = "x"


@dataclass(frozen=True, slots=True)
class ShortPosition:
    k: Annotated[int, Help("how many")]


@dataclass(frozen=True, slots=True)
class Either:
    value: Annotated[str | int, Help("a value")]


@dataclass(frozen=True, slots=True)
class Numbers:
    values: Annotated[tuple[int, ...], Help("numbers")]


@dataclass(frozen=True, slots=True)
class FilledList:
    names: Annotated[tuple[str, ...], Help("names")] = ("a",)


@dataclass(frozen=True, slots=True)
class Tagged:
    filters: Annotated[tuple[Filter, ...], Help("conditions")] = ()
    fields: Annotated[tuple[str, ...], Help("names")] = ()


@dataclass(frozen=True, slots=True)
class ReadAlone:
    filters: Annotated[str, Help("a condition")] = ""


@dataclass(frozen=True, slots=True)
class ReadFilled:
    fields: Annotated[tuple[str, ...], Help("names")] = ("url",)


def test_a_list_read_by_a_function_is_an_option_given_as_often_as_wanted() -> None:
    assert parsed(Tagged, []) == Tagged()
    given = ["-f", "year=2024", "--filter", "tags!=x", "--fields", "url,created", "--fields", "doi"]
    assert parsed(Tagged, given) == Tagged((written("year=2024"), written("tags!=x")), ("url", "created", "doi"))
    with pytest.raises(InvalidRequestError, match="'x' is no KEY=VALUE"):
        parsed(Tagged, ["-f", "x"])


@pytest.mark.parametrize(
    "request_type, refusal",
    [
        (int, "int is no dataclass"),
        (Numbers, r"Numbers.values: the command line takes no tuple\[int, \.\.\.\]"),
        (FilledList, r"FilledList.names: a list given by place has no default but \(\)"),
        (Ratio, "Ratio.ratio: the command line takes no complex"),
        (Either, r"Either.value: the command line takes no str \| int"),
        (ShortPosition, "ShortPosition.k: only an option given by name has a short flag"),
        (Undescribed, "Undescribed.name: no Help"),
        (OnByDefault, "OnByDefault.loud: a switch is off by default"),
        (ToggledOn, "ToggledOn.follow: a yes or no that may be left out has no default but None"),
        (OptionalWithAValue, "OptionalWithAValue.name: an option that may be left out has no default but None"),
        (ReadAlone, "ReadAlone.filters: only a list is read by a function"),
        (ReadFilled, r"ReadFilled.fields: a list read by a function has no default but \(\)"),
    ],
)
def test_a_field_the_command_line_cannot_take_is_refused(request_type: type[object], refusal: str) -> None:
    with pytest.raises(TypeError, match=refusal):
        options(argparse.ArgumentParser(), request_type, whole)


@dataclass(frozen=True, slots=True)
class Counted:
    query: Annotated[str, Help("what to find")]
    k: Annotated[int, Help("how many")] = 10
    order: Annotated[TitleFrom, Help("an order")] = TitleFrom.NAME


def test_a_field_with_a_value_for_default_is_an_option_and_may_have_a_short_flag() -> None:
    assert parsed(Counted, ["x"]) == Counted("x", 10, TitleFrom.NAME)
    assert parsed(Counted, ["x", "-k", "3", "--order", "heading"]) == Counted("x", 3, TitleFrom.HEADING)
    with pytest.raises(UsageError, match="unrecognized arguments: --k"):
        parsed(Counted, ["x", "--k", "4"])


@pytest.mark.parametrize("written", ["many", "1_000", " 3", "3.0", "+3", chr(0x663), "1" * 19, ""])
def test_a_number_written_other_than_in_digits_alone_is_refused(written: str) -> None:
    with pytest.raises(UsageError, match="is no whole number: write up to 18 digits alone"):
        parsed(Counted, ["x", f"-k={written}"])


@dataclass(frozen=True, slots=True)
class Timed:
    query: Annotated[str, Help("what to find")]
    budget: Annotated[float | None, Help("seconds")] = None


def test_a_decimal_option_takes_digits_with_a_point() -> None:
    assert parsed(Timed, ["x"]).budget is None
    assert parsed(Timed, ["x", "--budget", "0.25"]).budget == 0.25
    assert parsed(Timed, ["x", "--budget=-3"]).budget == -3.0


@pytest.mark.parametrize("written", ["1e3", "nan", "inf", ".5", "1.", "0x1", "1_0.5", ""])
def test_a_decimal_written_otherwise_is_refused(written: str) -> None:
    with pytest.raises(UsageError, match="is no number"):
        parsed(Timed, ["x", f"--budget={written}"])


def test_a_whole_number_may_be_negative_and_have_18_digits() -> None:
    assert parsed(Counted, ["x", "-k=-3"]).k == -3
    assert parsed(Counted, ["x", "-k", "9" * 18]).k == int("9" * 18)


def test_paths_are_made_whole_by_the_function_given(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    options(parser, AddCollection, lambda text: str(tmp_path / text))
    assert request_of(AddCollection, parser.parse_args(["x"])).folder == str(tmp_path / "x")


@dataclass(frozen=True, slots=True)
class Clashing:
    json: Annotated[bool, Help("clashes with --json")] = False


@dataclass(frozen=True, slots=True)
class Commanding:
    command: Annotated[str, Help("clashes with the command's name")]


@dataclass(frozen=True, slots=True)
class TwiceMany:
    first: Annotated[tuple[str, ...], Help("some")] = ()
    second: Annotated[tuple[str, ...], Help("more")] = ()


@pytest.mark.parametrize(
    "request_type, refusal",
    [
        (Clashing, "Clashing.json: sift's own option is named so"),
        (Commanding, "Commanding.command: sift's own option is named so"),
        (TwiceMany, "TwiceMany.second: only one list is given by its places"),
    ],
)
def test_a_request_the_command_line_would_misread_is_refused_when_its_parser_is_built(
    request_type: type[object], refusal: str
) -> None:
    with pytest.raises(TypeError, match=refusal):
        arguments(request_type)


def test_the_flags_of_a_search_are_named_for_one_value_given_again() -> None:
    assert flags(SearchKeywords) == {
        "query": "query",
        "k": "-k",
        "collections": "--collection",
        "path": "--path",
        "filters": "--filter",
        "fields": "--fields",
        "sort": "--sort",
        "links_to": "--links-to",
        "linked_from": "--linked-from",
        "all_passages": "--all-passages",
        "compact": "--compact",
        "adaptive": "--adaptive",
        "offset": "--offset",
        "budget": "--budget",
    }


def test_a_whole_number_s_range_and_default_are_shown_from_its_field() -> None:
    told = {each.name: each.help for each in arguments(SearchHybrid)}
    assert (told["k"], told["candidates"].endswith(" if fewer (1 to 10,000; default: 40)")) == (
        "how many results (1 to 10,000; default: 10)",
        True,
    )
    assert {each.name: each.help for each in arguments(ListJobs)}["stop"].endswith(")") is False


def test_options_the_command_line_reads_its_own_way_say_how_to_write_them() -> None:
    told = {each.name: each.help for each in arguments(SearchKeywords)}
    assert told["filters"].endswith("; written KEY=VALUE, KEY!=VALUE, or a range such as created>=2024-06")
    assert told["fields"].endswith("of them; comma-separated, e.g. url,created")
