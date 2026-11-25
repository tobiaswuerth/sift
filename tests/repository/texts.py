import re


def matches(pattern: str, text: str) -> list[str]:
    found: list[str] = re.findall(pattern, text)
    return found


def jobs(workflow: str) -> str:
    return "\n" + workflow.split("\njobs:\n", 1)[1]


def job(workflow: str, name: str) -> str:
    return re.split(r"\n  [\w-]+:\n", jobs(workflow).split(f"\n  {name}:\n", 1)[1], maxsplit=1)[0]
