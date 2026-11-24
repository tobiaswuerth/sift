from types import MappingProxyType

from sift.interfaces.acts.searching import MENTIONS, QUERY, SEARCH, SIMILAR, VSEARCH
from sift.interfaces.cli.commands.making import needing_collections, searched

COMMANDS = MappingProxyType(
    {
        "search": searched(SEARCH, "keyword"),
        "vsearch": searched(VSEARCH, "semantic"),
        "query": searched(QUERY, "hybrid"),
        "mentions": needing_collections(searched(MENTIONS, "keyword")),
        "similar": needing_collections(searched(SIMILAR, "semantic")),
    }
)
