import json
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Protocol, cast

from sift.api.errors import SiftError
from sift.api.wire import Json, encoded_error
from sift.interfaces.mcp.tools import Tool

SPOKEN = ("2025-03-26", "2025-06-18", "2025-11-25")
LATEST = SPOKEN[-1]
NAME = "sift"
MCP_PATH = "/mcp"
NOT_A_MESSAGE = -32600
NO_METHOD = -32601
WRONG_PARAMS = -32602
ONE_MESSAGE = "send one JSON-RPC 2.0 message as a JSON object"


class Serves(Protocol):
    def instructions(self) -> str: ...

    def tools(self) -> tuple[Tool, ...]: ...

    def called(self, tool: Tool, arguments: Mapping[str, object]) -> Json: ...


class RefusedError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code: int = code
        self.message: str = message


def _object(value: object, what: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise RefusedError(WRONG_PARAMS, f"{what} are a JSON object, not {value!r}")
    return cast("dict[str, object]", value)


def _initialized(served: Serves, params: Mapping[str, object], version: str) -> Json:
    asked = params.get("protocolVersion")
    return {
        "protocolVersion": asked if isinstance(asked, str) and asked in SPOKEN else LATEST,
        "capabilities": {"tools": {"listChanged": False}},
        "serverInfo": {"name": NAME, "title": NAME, "version": version},
        "instructions": served.instructions(),
    }


def _listed(tool: Tool) -> Json:
    return {
        "name": tool.name,
        "title": tool.title,
        "description": tool.description,
        "inputSchema": tool.schema,
        "annotations": {
            "title": tool.title,
            "readOnlyHint": tool.read_only,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
    }


def _text(value: Json) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _said(error: SiftError) -> str:
    told = f"{error.message}; fix: {error.remedy}" if error.remedy else error.message
    return f"{error.field}: {told}" if error.field else told


def _tool(served: Serves, name: object) -> Tool:
    tools = served.tools()
    found = next((tool for tool in tools if tool.name == name), None)
    if found is None:
        named = ", ".join(tool.name for tool in tools)
        raise RefusedError(WRONG_PARAMS, f"there is no tool {name!r}: sift's tools are {named}")
    return found


def _called(served: Serves, params: Mapping[str, object], _version: str) -> Json:
    tool = _tool(served, params.get("name"))
    arguments = _object(params.get("arguments", {}), "a tool's arguments")
    try:
        answer = served.called(tool, arguments)
    except SiftError as error:
        failed: Json = {"error": encoded_error(error)}
        return {"content": [{"type": "text", "text": _said(error)}], "structuredContent": failed, "isError": True}
    return {"content": [{"type": "text", "text": _text(answer)}], "structuredContent": answer, "isError": False}


def _pinged(_served: Serves, _params: Mapping[str, object], _version: str) -> Json:
    return {}


def _tools_listed(served: Serves, _params: Mapping[str, object], _version: str) -> Json:
    return {"tools": [_listed(tool) for tool in served.tools()]}


METHODS: Mapping[str, Callable[[Serves, Mapping[str, object], str], Json]] = MappingProxyType(
    {"initialize": _initialized, "ping": _pinged, "tools/list": _tools_listed, "tools/call": _called}
)


def _refused(at: Json, code: int, message: str) -> Json:
    return {"jsonrpc": "2.0", "id": at, "error": {"code": code, "message": message}}


def _method(method: object) -> Callable[[Serves, Mapping[str, object], str], Json]:
    found = METHODS.get(method) if isinstance(method, str) else None
    if found is None:
        raise RefusedError(NO_METHOD, f"sift answers no method {method!r}")
    return found


def _answered(served: Serves, message: Mapping[str, object], version: str) -> Json:
    at = cast("Json", message["id"])
    try:
        method = _method(message.get("method"))
        params = _object(message.get("params", {}), "params")
        return {"jsonrpc": "2.0", "id": at, "result": method(served, params, version)}
    except RefusedError as refused:
        return _refused(at, refused.code, refused.message)


def reply(message: object, served: Serves, version: str) -> Json | None:
    if not isinstance(message, dict) or cast("dict[str, object]", message).get("jsonrpc") != "2.0":
        return _refused(None, NOT_A_MESSAGE, ONE_MESSAGE)
    held = cast("dict[str, object]", message)
    if "method" not in held and "id" not in held:
        return _refused(None, NOT_A_MESSAGE, ONE_MESSAGE)
    if "id" not in held or "method" not in held:
        return None
    return _answered(served, held, version)
