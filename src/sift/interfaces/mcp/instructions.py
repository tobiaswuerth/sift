from sift.api.collections import CollectionStatus, Counted, Status
from sift.domain.text.wording import counted

HOW = """Search the user's local documents and notes, indexed by sift. When a task touches what the collections
below cover, search here before answering from memory or the web. Cite ids; say so when nothing relevant turns up.

Pick by what you have:
- A question, topic or description: `search` (hybrid: keywords and meaning, reranked). The default.
- An exact string, e.g. a version "3.12.4", an error code, identifier, name, DOI or "quoted phrase": `search` with
  mode keyword.
- An idea whose wording you can't guess: `search` with mode semantic.
- A document or note: `related` for more like it, its links, backlinks or unlinked mentions; `get` for its text.

Research in rounds; one search is rarely enough:
1. Survey: one call with several phrasings, query=["...", "...", "..."] (up to 20), with compact: true, k: 30,
   adaptive: true: hits without their text, so many fit.
2. Read: `get` the 3 or so best in one call, with their lines: documents=[{"id": "...", "first_line": 17,
   "last_line": 26}, ...].
3. Dig: `related` on the best one; search again with the terms and names the hits use.
4. Stop when a round returns only documents you have seen; then answer.

Narrow any search with `collections`, a `path` glob, `filters` on front matter and `sort`. Scores rank one call's
hits alone. After writing or deleting a file of a collection, `reindex` it."""
NONE_YET = "No collections are indexed yet."
LISTED = "Indexed collections (the tool `collections` lists them, with their fields and the jobs running):"


def _line(one: CollectionStatus) -> str:
    contents = one.contents
    if not isinstance(contents, Counted):
        return f"- {one.name} ({one.folder}): {contents.message}"
    said = f"- {one.name} ({one.folder}): {counted(contents.documents, 'document')}"
    if contents.fields:
        said += f"; front matter: {', '.join(contents.fields)}"
    if contents.links:
        said += f"; {counted(contents.links, 'link')}, {contents.unresolved:,} unresolved"
    return said if one.active else f"{said}; inactive"


def instructions(status: Status) -> str:
    if not status.collections:
        return f"{HOW}\n\n{NONE_YET}"
    return f"{HOW}\n\n{LISTED}\n" + "\n".join(_line(one) for one in status.collections)
