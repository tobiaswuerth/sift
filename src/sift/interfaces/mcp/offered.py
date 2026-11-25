from sift.interfaces.mcp.holdings import COLLECTIONS_TOOL, REINDEX_TOOL
from sift.interfaces.mcp.reading import GET_TOOL
from sift.interfaces.mcp.searches import RELATED_TOOL, SEARCH_TOOL
from sift.interfaces.mcp.tools import Tool

TOOLS: tuple[Tool, ...] = (SEARCH_TOOL, GET_TOOL, RELATED_TOOL, COLLECTIONS_TOOL, REINDEX_TOOL)
