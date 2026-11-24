import json
from pathlib import Path

from sift.interfaces.cli.commands.making import as_json
from tests.answers import SAMPLES

SNAPSHOT = Path(__file__).with_name("json_contract.json")


def test_the_json_of_every_answer_keeps_its_shape() -> None:
    found = {name: json.loads(as_json(answer)) for name, answer in SAMPLES.items()}
    assert found == json.loads(SNAPSHOT.read_text(encoding="utf-8"))
