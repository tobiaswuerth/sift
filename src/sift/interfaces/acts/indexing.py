from sift.api.embedding import EmbedCollections, EmbedResult
from sift.api.indexing import IndexCollections, IndexResult
from sift.api.reindexing import Reindexed, ReindexFiles
from sift.app import embedding, indexing
from sift.app.filing import Readers
from sift.app.scans import Scanning
from sift.interfaces.acts.acting import Act, Acting


def _readers(acting: Acting) -> Readers:
    return Readers(scanning=Scanning(acting.settings.ocr, gpus=acting.settings.gpus))


def _index(acting: Acting, request: IndexCollections) -> IndexResult:
    return indexing.index(acting.settings.home, request, acting.report, readers=_readers(acting))


def _reindex(acting: Acting, request: ReindexFiles) -> Reindexed:
    home, models = acting.settings.home, acting.models()
    grouped = indexing.files_of(home, request.files, acting.audience)
    names = embedding.embedded_by(home, tuple(one.name for one in grouped), models.name)
    indexed = indexing.index_files(home, grouped, acting.report, readers=_readers(acting))
    if not names:
        return Reindexed(indexed)
    return Reindexed(indexed, embedding.embed(home, EmbedCollections(names), acting.report, models))


def _embed(acting: Acting, request: EmbedCollections) -> EmbedResult:
    return embedding.embed(acting.settings.home, request, acting.report, acting.models())


INDEX = Act("index", IndexCollections, IndexResult, _index, writes=True)
EMBED = Act("embed", EmbedCollections, EmbedResult, _embed, writes=True)
REINDEX = Act("reindex", ReindexFiles, Reindexed, _reindex, writes=True)
