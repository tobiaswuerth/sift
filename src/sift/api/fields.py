from collections.abc import Mapping, Sequence

type FieldValue = str | int | float | bool | Sequence[FieldValue] | Mapping[str, FieldValue] | None
type Fields = Mapping[str, FieldValue]
