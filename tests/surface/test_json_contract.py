import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

from sift.api.events import EVENT_TAGS
from sift.api.wire import Json, encoded, encoded_error
from sift.interfaces.cli.commands.making import as_json
from sift.interfaces.server.handled import HANDLED
from tests.answers import SAMPLES
from tests.contract_samples import ERROR_SAMPLES, EVENT_SAMPLES, REQUEST_SAMPLES, SERVED

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

SNAPSHOT = Path(__file__).with_name("json_contract.json")


def contract() -> dict[str, Json]:
    return {
        "answers": {name: json.loads(as_json(answer)) for name, answer in {**SAMPLES, "serve": SERVED}.items()},
        "requests": {name: encoded(request) for name, request in REQUEST_SAMPLES.items()},
        "errors": {name: {"error": encoded_error(error)} for name, error in ERROR_SAMPLES.items()},
        "events": {
            name: {"event": EVENT_TAGS[type(event)], "data": encoded(event)} for name, event in EVENT_SAMPLES.items()
        },
    }


def test_the_json_of_every_answer_request_error_and_event_keeps_its_shape_and_order() -> None:
    written = json.dumps(contract(), ensure_ascii=False, indent=2) + "\n"
    assert written == SNAPSHOT.read_text(encoding="utf-8")


def test_the_snapshot_holds_a_request_of_every_kind_served_each_field_off_its_default() -> None:
    assert set(REQUEST_SAMPLES) == set(HANDLED)
    left = [
        f"{name}.{field.name}"
        for name, request in REQUEST_SAMPLES.items()
        for field in dataclasses.fields(cast("type[DataclassInstance]", type(request)))
        if cast("object", field.default) is not dataclasses.MISSING
        and cast("object", getattr(request, field.name)) == cast("object", field.default)
    ]
    assert left == []
    assert {type(event) for event in EVENT_SAMPLES.values()} == set(EVENT_TAGS)
