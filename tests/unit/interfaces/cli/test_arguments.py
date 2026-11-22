import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import pytest

from sift.api.collections import AddCollection, ShowStatus, TitleFrom
from sift.api.described import Help
from sift.api.documents import GetDocuments
from sift.api.indexing import IndexCollections
from sift.interfaces.cli.arguments import options, request_of


def whole(text: str) -> str:
    return f"/whole/{text}"


def parsed[R](request: type[R], argv: list[str]) -> R:
    parser = argparse.ArgumentParser(prog="sift add", exit_on_error=False)
    options(parser, request, whole)
    return request_of(request, parser.parse_args(argv))


def test_a_field_without_a_default_is_given_by_its_place_and_the_others_by_name() -> None:
    assert parsed(AddCollection, ["notes"]) == AddCollection("/whole/notes")
    given = ["notes", "--name", "n", "--store", "s", "--title-from", "name", "--nested"]
    assert parsed(AddCollection, given) == AddCollection(
        "/whole/notes", name="n", store="/whole/s", title_from=TitleFrom.NAME, nested=True
    )


def test_a_list_is_given_by_its_places_and_may_be_empty_if_it_has_a_default() -> None:
    assert parsed(ShowStatus, []) == ShowStatus()
    assert parsed(IndexCollections, ["a", "b"]) == IndexCollections(("a", "b"))
    assert parsed(GetDocuments, ["x", "y", "--lines", "3"]) == GetDocuments(("x", "y"), "3")
    with pytest.raises(argparse.ArgumentError, match="the following arguments are required: ids"):
        parsed(GetDocuments, [])


@dataclass(frozen=True, slots=True)
class Picked:
    title_from: Annotated[TitleFrom, Help("where titles come from")]


def test_a_choice_may_be_given_by_its_place() -> None:
    assert parsed(Picked, ["name"]) == Picked(TitleFrom.NAME)


def test_a_choice_not_offered_is_refused_with_the_choices_there_are() -> None:
    with pytest.raises(argparse.ArgumentError, match=r"invalid choice: 'title' \(choose from '?heading'?, '?name'?\)"):
        parsed(AddCollection, ["notes", "--title-from", "title"])


def test_each_argument_says_what_it_is_for() -> None:
    parser = argparse.ArgumentParser(prog="sift add")
    options(parser, AddCollection, whole)
    shown = " ".join(parser.format_help().split())
    usage = (
        "sift add [-h] [--name NAME] [--store STORE] [--title-from {heading,name}] [--nested] [--follow-links] folder"
    )
    assert usage in shown
    assert "folder the folder whose files the collection holds" in shown
    assert "--nested add a folder inside a collection" in shown


@dataclass(frozen=True, slots=True)
class Ratio:
    ratio: Annotated[float, Help("a share")]


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
class Defaulted:
    name: Annotated[str, Help("a name")] = "x"


@dataclass(frozen=True, slots=True)
class Either:
    value: Annotated[str | int, Help("a value")]


@dataclass(frozen=True, slots=True)
class Numbers:
    values: Annotated[tuple[int, ...], Help("numbers")]


@dataclass(frozen=True, slots=True)
class FilledList:
    names: Annotated[tuple[str, ...], Help("names")] = ("a",)


@pytest.mark.parametrize(
    "request_type, refusal",
    [
        (int, "int is no dataclass"),
        (Numbers, r"Numbers.values: the command line takes no tuple\[int, \.\.\.\]"),
        (FilledList, r"FilledList.names: a list given by place has no default but \(\)"),
        (Ratio, "Ratio.ratio: the command line takes no float"),
        (Either, r"Either.value: the command line takes no str \| int"),
        (Defaulted, "Defaulted.name: an argument given by its place has no default"),
        (Undescribed, "Undescribed.name: no Help"),
        (OnByDefault, "OnByDefault.loud: a switch is off by default"),
        (OptionalWithAValue, "OptionalWithAValue.name: an option that may be left out has no default but None"),
    ],
)
def test_a_field_the_command_line_cannot_take_is_refused(request_type: type[object], refusal: str) -> None:
    with pytest.raises(TypeError, match=refusal):
        options(argparse.ArgumentParser(), request_type, whole)


def test_paths_are_made_whole_by_the_function_given(tmp_path: Path) -> None:
    parser = argparse.ArgumentParser()
    options(parser, AddCollection, lambda text: str(tmp_path / text))
    assert request_of(AddCollection, parser.parse_args(["x"])).folder == str(tmp_path / "x")
