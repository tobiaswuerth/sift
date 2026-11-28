import dataclasses
import types
from collections.abc import Mapping
from types import MappingProxyType
from typing import Literal, TypeAliasType, cast, get_args, get_origin, get_type_hints

from sift.api.described import Bounds, Help, Within
from sift.api.requests import annotated_parts, choices_of
from sift.api.wire import Json, encoded

type Schema = dict[str, Json]

SCALARS: Mapping[object, str] = MappingProxyType(
    {str: "string", int: "integer", float: "number", bool: "boolean", types.NoneType: "null"}
)


def _united(members: tuple[object, ...]) -> Schema:
    found = [_value(member) for member in members]
    plain = [each["type"] for each in found if set(each) == {"type"} and isinstance(each["type"], str)]
    if len(plain) == len(found):
        return {"type": list[Json](plain)}
    return {"anyOf": list[Json](found)}


def _shaped(kind: object) -> Schema:
    origin, args = get_origin(kind), cast("tuple[object, ...]", get_args(kind))
    if origin is Literal:
        return {"enum": [encoded(each) for each in args]}
    if origin is tuple and args[1:] == (Ellipsis,):
        return {"type": "array", "items": _value(args[0])}
    if origin is Mapping:
        return {"type": "object"}
    choices = choices_of(kind)
    if choices is not None:
        return {"type": "string", "enum": [member.value for member in choices]}
    if isinstance(kind, type) and dataclasses.is_dataclass(kind):
        return schema_of(kind)
    raise TypeError(f"no JSON schema is made of {kind!r}")


def _value(kind: object) -> Schema:
    if isinstance(kind, TypeAliasType):
        return _value(cast("object", kind.__value__))
    if get_origin(kind) is types.UnionType:
        return _united(cast("tuple[object, ...]", get_args(kind)))
    scalar = SCALARS.get(kind)
    return {"type": scalar} if scalar is not None else _shaped(kind)


def _marked(schema: Schema, mark: object) -> Schema:
    listed = schema.get("type") == "array"
    match mark:
        case Help(text=text):
            return {**schema, "description": text}
        case Bounds(low=low, high=high):
            return {**schema, **({"minItems": low, "maxItems": high} if listed else {"minimum": low, "maximum": high})}
        case Within(low=low, high=high):
            return {**schema, "minimum": low, "maximum": high}
        case _:
            return schema


def property_of(hint: object, default: object = dataclasses.MISSING) -> Schema:
    kind, marks = annotated_parts(hint)
    schema = _value(kind)
    for mark in marks:
        schema = _marked(schema, mark)
    return schema if default is dataclasses.MISSING else {**schema, "default": encoded(default)}


def _default(field: dataclasses.Field[object]) -> object:
    if field.default_factory is not dataclasses.MISSING:
        return field.default_factory()
    return field.default


def schema_of(kind: type) -> Schema:
    hints = cast("dict[str, object]", get_type_hints(kind, include_extras=True))
    fields = dataclasses.fields(kind)
    properties: Schema = {field.name: property_of(hints[field.name], _default(field)) for field in fields}
    required: list[Json] = [field.name for field in fields if _default(field) is dataclasses.MISSING]
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}
