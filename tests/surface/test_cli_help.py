import re
from typing import cast

import pytest

from sift.interfaces.cli.entries import ENTRIES
from sift.settings import Settings
from tests.surface.commands import ran

ORDER = [
    "add",
    "index",
    "embed",
    "search",
    "vsearch",
    "query",
    "get",
    "links",
    "mentions",
    "similar",
    "unresolved",
    "status",
    "activate",
    "deactivate",
    "move",
    "rm",
    "serve",
    "jobs",
]
FIELD_NAMED = re.compile(r"\b(COLLECTIONS|FILTERS|FILES|IDS|NAMES|LINKS_TO|LINKED_FROM|TEXT_FIELDS|FIELDS|ALSO)\b")


def test_the_commands_are_listed_in_the_order_they_are_used_each_said_as_what_it_does(settings: Settings) -> None:
    code, out, _ = ran(settings, "--help")
    listed = cast("list[tuple[str, str]]", re.findall(r"^    (\w+) +(\S+)", out, re.MULTILINE))
    assert (code, [name for name, _ in listed]) == (0, ORDER)
    assert all(verb[0].islower() and not verb.startswith(("the", "link", "what")) for _, verb in listed)


def test_help_ends_with_examples_exit_codes_and_settings(settings: Settings) -> None:
    out = ran(settings, "--help")[1]
    examples, codes, variables = (out.index(each) for each in ("examples:", "exit codes:", "settings, read from"))
    assert examples < codes < variables
    assert '  sift search "dark matter" -k 5\n' in out
    assert "  0 done, 1 failed, 2 refused as asked, 130 stopped\n" in out


def test_a_command_s_help_names_what_each_option_takes_and_shows_an_example(settings: Settings) -> None:
    out = ran(settings, "search", "--help")[1]
    for written in (
        "--collection NAME",
        "--filter KEY=VALUE",
        "--fields FIELD,...",
        "--links-to NOTE",
        "--linked-from NOTE",
        "--path GLOB",
        "--sort FIELD",
        "--offset N",
        "-k N",
        "--budget SECONDS",
        "--sort=-created",
        "QUERY",
    ):
        assert written in out, written
    assert out.rstrip("\n").endswith('e.g. sift search "dark matter" -k 5')


@pytest.mark.parametrize("command", list(ENTRIES))
def test_no_command_names_a_value_by_its_field(settings: Settings, command: str) -> None:
    out = ran(settings, command, "--help")[1]
    assert FIELD_NAMED.findall(out.split("\n\n", 1)[0]) == []


def test_a_mistyped_command_is_told_the_nearest_one(settings: Settings) -> None:
    code, _, err = ran(settings, "searchh", "x")
    assert (code, "  fix  did you mean sift search? see sift --help\n" in err) == (2, True)
