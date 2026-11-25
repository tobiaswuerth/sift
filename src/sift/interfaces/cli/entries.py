from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

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
from sift.api.serving import OpenUi, ServeHome
from sift.interfaces.cli.context import Command, Commands
from sift.interfaces.cli.reading import ReadDocuments
from sift.late import late

COMMANDS_PACKAGE = "sift.interfaces.cli.commands"
LOGGED = frozenset({"add", "rm", "move", "activate", "deactivate", "index", "embed", "serve"})
SEARCH_EXAMPLE = 'sift search "dark matter" -k 5'
GET_EXAMPLE = "sift get notes/physics.md --lines 17-26"


@dataclass(frozen=True, slots=True)
class CommandEntry:
    help: str
    request: type[object]
    module: str
    example: str
    here: bool = False


ENTRIES = MappingProxyType(
    {
        "add": CommandEntry("add a folder as a collection", AddCollection, "collections", "sift add ~/notes"),
        "index": CommandEntry(
            "read what is new or changed in collections", IndexCollections, "indexing", "sift index notes"
        ),
        "embed": CommandEntry(
            "embed passages for semantic search, with the model SIFT_EMBED_MODEL names",
            EmbedCollections,
            "indexing",
            "sift embed notes",
        ),
        "search": CommandEntry(
            "find passages by their words, ranked by BM25", SearchKeywords, "searching", SEARCH_EXAMPLE
        ),
        "vsearch": CommandEntry(
            "find passages near a query in meaning",
            SearchVectors,
            "searching",
            'sift vsearch "what holds galaxies together"',
        ),
        "query": CommandEntry(
            "find passages for a question: keyword and semantic, reranked",
            SearchHybrid,
            "searching",
            'sift query "what holds galaxies together?"',
        ),
        "get": CommandEntry("print documents by their ids", ReadDocuments, "collections", GET_EXAMPLE),
        "links": CommandEntry(
            "show the links of a note: those it holds, and those into it",
            ShowLinks,
            "links",
            'sift links "Acme AG" --direction both',
        ),
        "mentions": CommandEntry(
            "find documents naming a note without linking to it", FindMentions, "searching", 'sift mentions "Acme AG"'
        ),
        "similar": CommandEntry(
            "find documents near one in meaning, by its own vectors",
            FindSimilar,
            "searching",
            "sift similar notes/physics.md",
        ),
        "unresolved": CommandEntry(
            "list the link targets no note answers to yet, most linked first",
            ListUnresolved,
            "links",
            "sift unresolved -c notes",
        ),
        "status": CommandEntry(
            "show each collection: its folder and what its store holds", ShowStatus, "collections", "sift status notes"
        ),
        "activate": CommandEntry(
            "search collections deactivated before again", ActivateCollections, "collections", "sift activate archive"
        ),
        "deactivate": CommandEntry(
            "leave collections out of searches, and of index and embed unless named",
            DeactivateCollections,
            "collections",
            "sift deactivate archive",
        ),
        "move": CommandEntry(
            "move the stores of collections, or of all, to another folder or drive",
            MoveCollections,
            "collections",
            "sift move notes D:/stores",
        ),
        "rm": CommandEntry(
            "remove a collection and its store; its files stay where they are",
            RemoveCollection,
            "collections",
            "sift rm archive",
        ),
        "serve": CommandEntry(
            "serve this home: its models and codes held, commands answered at once",
            ServeHome,
            "serving",
            "sift serve",
            here=True,
        ),
        "ui": CommandEntry(
            "open this home's pages in the browser: search, browse and read; served here if no sift serve runs",
            OpenUi,
            "serving",
            "sift ui",
            here=True,
        ),
        "jobs": CommandEntry(
            "list what sift serve does in the background, or stop one of its jobs",
            ListJobs,
            "jobs",
            "sift jobs --stop 3",
        ),
    }
)
EXAMPLES = ("sift add ~/notes", "sift index", SEARCH_EXAMPLE, GET_EXAMPLE)


def command_of(name: str) -> Command:
    return late(f"{COMMANDS_PACKAGE}.{ENTRIES[name].module}", Commands).COMMANDS[name]


if TYPE_CHECKING:
    from sift.interfaces.cli.commands import collections as _collections
    from sift.interfaces.cli.commands import indexing as _indexing
    from sift.interfaces.cli.commands import jobs as _jobs
    from sift.interfaces.cli.commands import links as _links
    from sift.interfaces.cli.commands import searching as _searching
    from sift.interfaces.cli.commands import serving as _serving

    _COLLECTIONS: Commands = _collections
    _INDEXING: Commands = _indexing
    _JOBS: Commands = _jobs
    _LINKS: Commands = _links
    _SEARCHING: Commands = _searching
    _SERVING: Commands = _serving
