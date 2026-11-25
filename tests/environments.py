import os
from collections.abc import Mapping

from sift.settings import HOME, STATE_VARIABLE, STATE_VARIABLES, VARIABLES

SYSTEM = frozenset(
    {"PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP", "TMPDIR", "HOME", "LANG"}
)
KEPT = SYSTEM | {"USERPROFILE", "PYTHONPATH", HOME, STATE_VARIABLE, *STATE_VARIABLES.values()}
KEPT_PREFIXES = ("COVERAGE_",)


def clean_environment(inherited: Mapping[str, str] = os.environ, **given: str) -> dict[str, str]:
    kept = {name: value for name, value in inherited.items() if name in KEPT or name.startswith(KEPT_PREFIXES)}
    defaults = {name: "" for name in VARIABLES if name != HOME}
    return {**kept, **defaults, **given}
