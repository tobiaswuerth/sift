import asyncio
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import cast
from urllib.parse import urlsplit

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from sift.api.collections import Audience, Status
from sift.api.errors import InternalError, InvalidRequestError, SiftError, system_failure
from sift.api.jobs import ServedJob
from sift.api.wire import Json
from sift.interfaces.acts.acting import answered
from sift.interfaces.mcp.instructions import instructions, unlisted
from sift.interfaces.mcp.offered import TOOLS
from sift.interfaces.mcp.protocol import SPOKEN, reply
from sift.interfaces.mcp.tools import Started, Tool
from sift.interfaces.server.app import FAILED_REMEDY, HOSTS, LOGGER, Sent, Serving, body_of, json_of, read_for, run_job
from sift.interfaces.server.handled import Handled

JOB_WAIT_SECONDS = 20.0
PARSE_ERROR = -32700
NOT_A_MESSAGE = -32600
INTERNAL = -32603
POSTED = MappingProxyType({"Allow": "POST"})


@dataclass(frozen=True, slots=True)
class AgentsRunning:
    serving: Serving
    handled: Mapping[str, Handled]
    wait: float = JOB_WAIT_SECONDS

    def _sent(self, name: str, data: Json) -> Sent:
        return Sent(name, self.handled[name], data, Audience.AGENTS)

    def read(self, name: str, data: Json) -> object:
        return read_for(self.serving, self._sent(name, data))

    def job(self, name: str, data: Json) -> Started:
        job = self.serving.jobs.queued(name)
        job.watched = False
        ran = self.serving.writer.submit(run_job, self.serving, self._sent(name, data), job)
        try:
            return Started(job.number, ran.result(timeout=self.wait))
        except TimeoutError:
            return Started(job.number, None)

    def jobs(self) -> tuple[ServedJob, ...]:
        return self.serving.jobs.listed()


@dataclass(frozen=True, slots=True)
class AgentsServed:
    running: AgentsRunning

    def instructions(self) -> str:
        try:
            return instructions(answered(self.running.read("status", {}), Status))
        except SiftError as error:
            return unlisted(error)
        except OSError as error:
            return unlisted(system_failure(error))

    def tools(self) -> tuple[Tool, ...]:
        return TOOLS

    def called(self, tool: Tool, arguments: Mapping[str, object]) -> Json:
        try:
            return tool.call(self.running, arguments)
        except SiftError:
            raise
        except OSError as error:
            raise system_failure(error) from error
        except RecursionError:
            raise
        except Exception as error:
            logging.getLogger(LOGGER).exception("the tool %s failed", tool.name)
            said = f"the home process failed: {type(error).__name__}: {error}"
            raise InternalError(said, FAILED_REMEDY) from error


def _refused(status: int, code: int, message: str) -> JSONResponse:
    return JSONResponse({"jsonrpc": "2.0", "id": None, "error": {"code": code, "message": message}}, status)


def _asked(message: object, key: str) -> Json:
    return cast("Json", cast("dict[str, object]", message).get(key)) if isinstance(message, dict) else None


async def _replied(serving: Serving, served: AgentsServed, message: object) -> Json | None:
    try:
        return await asyncio.wrap_future(serving.pool.submit(reply, message, served, serving.version))
    except RecursionError:
        raise
    except Exception as error:
        logging.getLogger(LOGGER).exception("the message %s failed", _asked(message, "method"))
        said = f"the home process failed: {type(error).__name__}: {error}; fix: {FAILED_REMEDY}"
        return {"jsonrpc": "2.0", "id": _asked(message, "id"), "error": {"code": INTERNAL, "message": said}}


def _foreign(origin: str | None) -> bool:
    return origin is not None and urlsplit(origin).hostname not in HOSTS


async def agents_answered(serving: Serving, handled: Mapping[str, Handled], request: Request) -> Response:
    if request.method != "POST":
        return Response(status_code=405, headers=dict(POSTED))
    if _foreign(request.headers.get("origin")):
        return _refused(403, NOT_A_MESSAGE, "sift answers pages of this machine alone")
    version = request.headers.get("mcp-protocol-version")
    if version is not None and version not in SPOKEN:
        return _refused(400, NOT_A_MESSAGE, f"sift speaks MCP {', '.join(SPOKEN)}, not {version}")
    body = await body_of(request)
    if body is None:
        return _refused(413, NOT_A_MESSAGE, "the message is too long")
    try:
        message = json_of(body)
    except (InvalidRequestError, ValueError) as error:
        return _refused(400, PARSE_ERROR, f"the message is no JSON: {error}")
    answer = await _replied(serving, AgentsServed(AgentsRunning(serving, handled)), message)
    return Response(status_code=202) if answer is None else JSONResponse(answer)
