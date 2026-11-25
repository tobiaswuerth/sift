from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.routing import Route
from starlette.types import ASGIApp, Receive, Scope, Send

from sift.interfaces.server.agents import agents_answered
from sift.interfaces.server.app import HOSTS, Serving, command_route, unknown_command
from sift.interfaces.server.handled import HANDLED, Handled, for_agents

if TYPE_CHECKING:
    from starlette.requests import Request
    from starlette.responses import Response

MCP_PATH = "/mcp"
LISTENED_FOR_LEAVING = MappingProxyType({"version": "3.0", "spec_version": "2.3"})


@dataclass(frozen=True, slots=True)
class ListeningForLeaving:
    app: ASGIApp

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await self.app({**scope, "asgi": dict(LISTENED_FOR_LEAVING)}, receive, send)


def served_app(serving: Serving, handled: Mapping[str, Handled] = HANDLED) -> Starlette:
    agents: Callable[[Request], Awaitable[Response]] = partial(agents_answered, serving, for_agents(handled))
    return Starlette(
        routes=[
            *(command_route(serving, name, each) for name, each in handled.items()),
            Route(MCP_PATH, agents, methods=["GET", "POST", "DELETE"]),
        ],
        middleware=[Middleware(ListeningForLeaving), Middleware(TrustedHostMiddleware, allowed_hosts=list(HOSTS))],
        exception_handlers={404: unknown_command, 405: unknown_command},
    )
