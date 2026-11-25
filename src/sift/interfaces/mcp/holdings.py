from collections.abc import Mapping
from typing import cast

from sift.api.collections import CollectionStatus, Counted, ShowStatus, Status
from sift.api.embedding import Embedded
from sift.api.indexing import IndexResult, Skipped
from sift.api.jobs import ServedJob
from sift.api.reindexing import Reindexed, ReindexFiles
from sift.api.wire import Json
from sift.interfaces.acts.acting import answered
from sift.interfaces.mcp.answers import ToolAnswer, said, with_notes
from sift.interfaces.mcp.schemas import schema_of
from sift.interfaces.mcp.tools import Tool, ToolRunning

MOST_FIELDS_LISTED = 200
RUNS_ON = "it runs on: collections lists it until it ends"
COLLECTIONS = (
    "What the user's indexed collections hold, those agents may search: each one's folder, documents, passages and "
    "how many are embedded (searched by meaning), its front matter fields (to filter and sort by) and links; and the "
    "jobs that run or wait, such as a reindex."
)
REINDEX = (
    "Read files again right after writing, changing or deleting them, so that searches find them as they are now; "
    "their new passages are embedded too where their collection is. Changes only sift's index, never a file. Says "
    "how each file went, or, if other work runs first, that its job runs on."
)


def _counted(contents: Counted) -> ToolAnswer:
    shown: ToolAnswer = {
        "documents": contents.documents,
        "passages": contents.chunks,
        "embedded": contents.embedded,
        "fields": list[Json](contents.fields[:MOST_FIELDS_LISTED]),
        "links": contents.links,
        "unresolved": contents.unresolved,
    }
    if len(contents.fields) > MOST_FIELDS_LISTED:
        shown["more_fields"] = len(contents.fields) - MOST_FIELDS_LISTED
    if contents.unreadable:
        shown["unreadable"] = contents.unreadable
    return shown


def _collection(one: CollectionStatus) -> ToolAnswer:
    shown: ToolAnswer = {"name": one.name, "folder": one.folder}
    if not one.active:
        shown["active"] = False
    if isinstance(one.contents, Counted):
        return {**shown, **_counted(one.contents)}
    return {**shown, "problem": said(one.contents)}


def _job(job: ServedJob, seen: frozenset[str]) -> ToolAnswer:
    shown: ToolAnswer = {
        "job": job.id,
        "command": job.command,
        "running": job.running,
        "seconds": round(job.seconds, 1),
    }
    if job.collection in seen:
        shown["collection"] = job.collection
    if job.done is not None:
        shown["done"] = job.done
    return shown


def _collections(running: ToolRunning, arguments: Mapping[str, object]) -> Json:
    status = answered(running.read("status", cast("Json", dict(arguments))), Status)
    seen = frozenset(one.name for one in status.collections)
    return {
        "collections": [_collection(one) for one in status.collections],
        "jobs": [_job(job, seen) for job in running.jobs()],
    }


def _files(indexed: IndexResult) -> ToolAnswer:
    files = list[Json]()
    for one in indexed.collections:
        for each in one.named:
            files.append({"id": each.id, "state": each.state, **({"note": each.note} if each.note else {})})
    skipped = tuple(one.outcome for one in indexed.collections if isinstance(one.outcome, Skipped))
    return with_notes({"files": files}, skipped)


def _reindex(running: ToolRunning, arguments: Mapping[str, object]) -> Json:
    started = running.job("reindex", cast("Json", dict(arguments)))
    if started.answer is None:
        return {"job": started.number, "running": RUNS_ON}
    done = answered(started.answer, Reindexed)
    shown: ToolAnswer = {"job": started.number, **_files(done.indexed)}
    if done.embedded is not None:
        stored = (each.outcome for each in done.embedded.collections)
        shown["embedded"] = sum(each.stored for each in stored if isinstance(each, Embedded))
    return shown


COLLECTIONS_TOOL = Tool("collections", "Collections", COLLECTIONS, schema_of(ShowStatus), _collections, read_only=True)
REINDEX_TOOL = Tool("reindex", "Reindex", REINDEX, schema_of(ReindexFiles), _reindex, read_only=False)
