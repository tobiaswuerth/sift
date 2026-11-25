from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

import pytest

from sift.interfaces.mcp.schemas import schema_of


@dataclass(frozen=True, slots=True)
class Point:
    x: int


@dataclass(frozen=True, slots=True)
class Shapes:
    kind: Literal["a", "b"]
    extra: Mapping[str, str]
    at: Point | int
    many: tuple[str, ...] = field(default_factory=tuple[str, ...])


@dataclass(frozen=True, slots=True)
class Odd:
    when: complex


def test_literals_objects_unions_and_made_defaults_have_their_schemas() -> None:
    assert schema_of(Shapes) == {
        "type": "object",
        "properties": {
            "kind": {"enum": ["a", "b"]},
            "extra": {"type": "object"},
            "at": {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {"x": {"type": "integer"}},
                        "required": ["x"],
                        "additionalProperties": False,
                    },
                    {"type": "integer"},
                ]
            },
            "many": {"type": "array", "items": {"type": "string"}, "default": []},
        },
        "required": ["kind", "extra", "at"],
        "additionalProperties": False,
    }


def test_a_kind_json_has_no_schema_for_is_a_mistake_said_at_once() -> None:
    with pytest.raises(TypeError, match="no JSON schema is made of <class 'complex'>"):
        schema_of(Odd)
