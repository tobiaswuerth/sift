from collections.abc import Mapping, Sequence

type FieldValue = str | int | float | bool | Sequence[FieldValue] | Mapping[str, FieldValue] | None
type Fields = Mapping[str, FieldValue]


FIELDS = 'the front matter fields to show, e.g. url and created; "*" for all of them'
