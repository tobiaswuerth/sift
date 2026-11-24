import sys
from types import MappingProxyType

from tools import (
    check_dead,
    check_idioms,
    check_imports,
    check_names,
    check_package,
    check_prose,
    check_shape,
    check_stubs,
    check_text,
    check_twins,
    fts5_table,
)

CHECKS = MappingProxyType(
    {
        "dead": check_dead.main,
        "fts5": fts5_table.main,
        "idioms": check_idioms.main,
        "imports": check_imports.main,
        "names": check_names.main,
        "package": check_package.main,
        "prose": check_prose.main,
        "shape": check_shape.main,
        "stubs": check_stubs.main,
        "text": check_text.main,
        "twins": check_twins.main,
    }
)


def main(argv: list[str]) -> int:
    if not argv or argv[0] not in CHECKS:
        print(f"usage: python -m tools ({' | '.join(CHECKS)}) [arguments]")
        return 2
    return CHECKS[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
