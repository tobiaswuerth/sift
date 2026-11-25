from typing import cast, get_type_hints

from sift.interfaces.acts import collections, indexing, jobs, links, searching
from sift.interfaces.acts.acting import Act
from sift.interfaces.server.handled import HANDLED

ACTS: list[Act[object, object]] = [
    value
    for module in (collections, indexing, jobs, links, searching)
    for value in cast("dict[str, object]", vars(module)).values()
    if isinstance(value, Act)
]


def test_every_act_served_names_the_answer_its_run_gives() -> None:
    assert sorted(act.name for act in ACTS) == sorted(HANDLED)
    assert [act.name for act in ACTS if act.answer != get_type_hints(act.run)["return"]] == []
