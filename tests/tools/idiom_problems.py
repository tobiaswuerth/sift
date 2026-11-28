from tools.check_idioms import idioms


def problems(path: str, source: str) -> list[str]:
    return [finding.problem for finding in idioms(path, source)]
