from types import MappingProxyType

from sift.interfaces.acts.searching import MENTIONS, QUERY, SEARCH, SIMILAR, VSEARCH
from sift.interfaces.cli.commands.making import needing_collections, searched

COMMANDS = MappingProxyType(
    {
        "search": searched(SEARCH),
        "vsearch": searched(VSEARCH),
        "query": searched(QUERY),
        "mentions": needing_collections(searched(MENTIONS)),
        "similar": needing_collections(searched(SIMILAR)),
    }
)
