from importlib import import_module
from typing import cast


def late[P](name: str, _protocol: type[P]) -> P:
    module: object = import_module(name)
    return cast("P", module)
