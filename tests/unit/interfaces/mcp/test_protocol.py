from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

from sift.api.errors import InvalidRequestError, NotFoundError
from sift.api.wire import Json
from sift.interfaces.mcp.protocol import LATEST, Serves, reply
from sift.interfaces.mcp.tools import Tool, ToolRunning


def _unused(_running: ToolRunning, _arguments: Mapping[str, object]) -> Json:
    raise AssertionError


ECHO = Tool("echo", "Echo", "says back what it is given", {"type": "object"}, _unused, read_only=True)
REFUSED = InvalidRequestError("k is 0, not 1 to 10,000", "give 1 to 10,000", "k")


@dataclass(slots=True)
class Served:
    calls: list[tuple[str, Mapping[str, object]]] = field(default_factory=list[tuple[str, Mapping[str, object]]])

    def instructions(self) -> str:
        return "search here first"

    def tools(self) -> tuple[Tool, ...]:
        return (ECHO,)

    def called(self, tool: Tool, arguments: Mapping[str, object]) -> Json:
        self.calls.append((tool.name, arguments))
        if arguments.get("k") == 0:
            raise REFUSED
        if arguments.get("note") == "gone":
            raise NotFoundError("no note or document 'gone'", "take the id from a search result")
        return {"said": cast("Json", dict(arguments))}


def asked(method: str, params: object = None, *, served: Serves | None = None) -> Json | None:
    message: dict[str, object] = {"jsonrpc": "2.0", "id": 7, "method": method}
    if params is not None:
        message["params"] = params
    return reply(message, served or Served(), "2.0.0")


def test_initialize_agrees_on_a_version_both_speak_and_gives_the_instructions() -> None:
    hello = {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "an agent", "version": "1"}}
    assert asked("initialize", hello) == {
        "jsonrpc": "2.0",
        "id": 7,
        "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "sift", "title": "sift", "version": "2.0.0"},
            "instructions": "search here first",
        },
    }


def test_a_version_sift_does_not_speak_is_answered_with_its_newest() -> None:
    answer = asked("initialize", {"protocolVersion": "1999-01-01", "capabilities": {}})
    assert isinstance(answer, dict)
    result = answer["result"]
    assert isinstance(result, dict)
    assert result["protocolVersion"] == LATEST


def test_notifications_and_answers_get_no_reply() -> None:
    assert reply({"jsonrpc": "2.0", "method": "notifications/initialized"}, Served(), "2.0.0") is None
    assert reply({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {}}, Served(), "2.0.0") is None
    assert reply({"jsonrpc": "2.0", "id": 3, "result": {}}, Served(), "2.0.0") is None


def test_ping_is_answered() -> None:
    assert asked("ping") == {"jsonrpc": "2.0", "id": 7, "result": {}}


def test_tools_are_listed_with_their_schemas_and_whether_they_change_anything() -> None:
    assert asked("tools/list") == {
        "jsonrpc": "2.0",
        "id": 7,
        "result": {
            "tools": [
                {
                    "name": "echo",
                    "title": "Echo",
                    "description": "says back what it is given",
                    "inputSchema": {"type": "object"},
                    "annotations": {
                        "title": "Echo",
                        "readOnlyHint": True,
                        "destructiveHint": False,
                        "idempotentHint": True,
                        "openWorldHint": False,
                    },
                }
            ]
        },
    }


def test_a_tool_called_answers_as_text_and_as_data() -> None:
    served = Served()
    answer = asked("tools/call", {"name": "echo", "arguments": {"q": "ü"}}, served=served)
    assert answer == {
        "jsonrpc": "2.0",
        "id": 7,
        "result": {
            "content": [{"type": "text", "text": '{"said":{"q":"ü"}}'}],
            "structuredContent": {"said": {"q": "ü"}},
            "isError": False,
        },
    }
    assert served.calls == [("echo", {"q": "ü"})]


def test_a_tool_called_without_arguments_gets_none() -> None:
    served = Served()
    asked("tools/call", {"name": "echo"}, served=served)
    assert served.calls == [("echo", {})]


def test_a_refused_call_says_what_went_wrong_and_how_to_fix_it_to_the_agent() -> None:
    answer = asked("tools/call", {"name": "echo", "arguments": {"k": 0}})
    assert answer == {
        "jsonrpc": "2.0",
        "id": 7,
        "result": {
            "content": [{"type": "text", "text": "k: k is 0, not 1 to 10,000; fix: give 1 to 10,000"}],
            "structuredContent": {
                "error": {
                    "kind": "invalid_request",
                    "code": "invalid_request",
                    "message": "k is 0, not 1 to 10,000",
                    "remedy": "give 1 to 10,000",
                    "field": "k",
                }
            },
            "isError": True,
        },
    }


def test_an_error_without_a_field_is_said_plainly() -> None:
    answer = asked("tools/call", {"name": "echo", "arguments": {"note": "gone"}})
    assert isinstance(answer, dict)
    result = answer["result"]
    assert isinstance(result, dict)
    assert result["content"] == [
        {"type": "text", "text": "no note or document 'gone'; fix: take the id from a search result"}
    ]


def test_an_unknown_tool_is_a_protocol_error_naming_the_tools() -> None:
    assert asked("tools/call", {"name": "vsearch", "arguments": {}}) == {
        "jsonrpc": "2.0",
        "id": 7,
        "error": {"code": -32602, "message": "there is no tool 'vsearch': sift's tools are echo"},
    }


def test_arguments_that_are_no_object_are_refused() -> None:
    answer = asked("tools/call", {"name": "echo", "arguments": ["q"]})
    assert answer == {
        "jsonrpc": "2.0",
        "id": 7,
        "error": {"code": -32602, "message": "a tool's arguments are a JSON object, not ['q']"},
    }


def test_an_unknown_method_is_said_to_be() -> None:
    assert asked("server/discover") == {
        "jsonrpc": "2.0",
        "id": 7,
        "error": {"code": -32601, "message": "sift answers no method 'server/discover'"},
    }


def test_what_is_no_message_is_refused_without_an_id() -> None:
    for wrong in (
        [{"jsonrpc": "2.0", "id": 1, "method": "ping"}],
        "ping",
        {"id": 1, "method": "ping"},
        {"jsonrpc": "2.0"},
    ):
        assert reply(wrong, Served(), "2.0.0") == {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32600, "message": "send one JSON-RPC 2.0 message as a JSON object"},
        }


def test_params_that_are_no_object_are_refused() -> None:
    assert asked("tools/call", ["echo"]) == {
        "jsonrpc": "2.0",
        "id": 7,
        "error": {"code": -32602, "message": "params are a JSON object, not ['echo']"},
    }
