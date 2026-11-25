from typing import NamedTuple


class ModelsPrepared(NamedTuple):
    notices: tuple[str, ...]
    built: bool
