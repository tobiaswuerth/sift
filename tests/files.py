import json
from collections.abc import Mapping, Sequence
from pathlib import Path


def write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def jsonl(records: Sequence[Mapping[str, object]]) -> str:
    return "".join(json.dumps(each) + "\n" for each in records)
