import json
from typing import cast

type Message = dict[str, object]

STDIO = ("serve", "--stdio", "--idle-exit", "120")


def stdio_search(query: str) -> str:
    search = {"name": "search", "arguments": {"query": query, "mode": "keyword"}}
    messages = (
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": search},
    )
    return "".join(json.dumps(each) + "\n" for each in messages)


def answers_of(stdout: str) -> dict[int, Message]:
    read = [cast("Message", json.loads(line)) for line in stdout.splitlines()]
    return {cast("int", each["id"]): each for each in read}


def found_ids(answer: Message) -> list[str]:
    hits = cast("list[Message]", cast("Message", cast("Message", answer["result"])["structuredContent"])["hits"])
    return sorted(cast("str", hit["id"]) for hit in hits)
