import asyncio
from typing import cast, override

from uvicorn.protocols.http.h11_impl import H11Protocol, RequestResponseCycle


class HeadsInTime(H11Protocol):
    head_due: asyncio.TimerHandle | None = None

    @override
    def connection_made(self, transport: asyncio.Transport) -> None:
        super().connection_made(transport)
        self._awaited()

    @override
    def data_received(self, data: bytes) -> None:
        before = cast("RequestResponseCycle | None", self.cycle)
        super().data_received(data)
        if self.cycle is before and (before is None or before.response_complete):
            self._awaited()
        else:
            self._came()

    @override
    def connection_lost(self, exc: Exception | None) -> None:
        self._came()
        super().connection_lost(exc)

    def _awaited(self) -> None:
        if self.head_due is None:
            self.head_due = self.loop.call_later(self.timeout_keep_alive, self.timeout_keep_alive_handler)

    def _came(self) -> None:
        if self.head_due is not None:
            self.head_due.cancel()
            self.head_due = None
