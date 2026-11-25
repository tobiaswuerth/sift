from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType
from typing import cast

from sift.api.collections import (
    ActivateCollections,
    AddCollection,
    DeactivateCollections,
    MoveCollections,
    RemoveCollection,
    ShowStatus,
)
from sift.api.embedding import EmbedCollections
from sift.api.indexing import IndexCollections
from sift.api.jobs import ListJobs
from sift.api.links import FindMentions, FindSimilar, ListUnresolved, ShowLinks
from sift.api.search import SearchHybrid, SearchKeywords, SearchVectors
from sift.api.serving import ServeHome
from sift.interfaces.cli.context import Command, Commands
from sift.interfaces.cli.reading import ReadDocuments

COMMANDS_PACKAGE = "sift.interfaces.cli.commands"
LOGGED = frozenset({"add", "rm", "move", "activate", "deactivate", "index", "embed", "serve"})


@dataclass(frozen=True, slots=True)
class CommandEntry:
    help: str
    request: type[object]
    module: str


ENTRIES = MappingProxyType(
    {
        "add": CommandEntry("add a folder as a collection", AddCollection, "collections"),
        "rm": CommandEntry(
            "remove a collection and its store; its files stay where they are", RemoveCollection, "collections"
        ),
        "move": CommandEntry(
            "move the stores of collections, or of all, to another folder or drive", MoveCollections, "collections"
        ),
        "activate": CommandEntry("search collections deactivated before again", ActivateCollections, "collections"),
        "deactivate": CommandEntry(
            "leave collections out of searches, and of index and embed unless named",
            DeactivateCollections,
            "collections",
        ),
        "index": CommandEntry("read what is new or changed in collections", IndexCollections, "indexing"),
        "query": CommandEntry(
            "find passages for a question: keyword and semantic, reranked", SearchHybrid, "searching"
        ),
        "search": CommandEntry("find passages by their words, ranked by BM25", SearchKeywords, "searching"),
        "embed": CommandEntry(
            "embed passages for semantic search, with the model SIFT_EMBED_MODEL names", EmbedCollections, "indexing"
        ),
        "vsearch": CommandEntry("find passages near a query in meaning", SearchVectors, "searching"),
        "get": CommandEntry("print documents by their ids", ReadDocuments, "collections"),
        "similar": CommandEntry("find documents near one in meaning, by its own vectors", FindSimilar, "searching"),
        "mentions": CommandEntry("find documents naming a note without linking to it", FindMentions, "searching"),
        "links": CommandEntry("the links of a note: those it holds, and those into it", ShowLinks, "links"),
        "unresolved": CommandEntry("link targets no note answers to yet, most linked first", ListUnresolved, "links"),
        "jobs": CommandEntry("what sift serve does in the background, or stop one of its jobs", ListJobs, "jobs"),
        "status": CommandEntry("show each collection: its folder and what its store holds", ShowStatus, "collections"),
        "serve": CommandEntry(
            "serve this home: its models and codes held, commands answered at once", ServeHome, "serving"
        ),
    }
)


def _as_commands(module: object) -> Commands:
    return cast("Commands", module)


def command_of(name: str) -> Command:
    return _as_commands(import_module(f"{COMMANDS_PACKAGE}.{ENTRIES[name].module}")).COMMANDS[name]
