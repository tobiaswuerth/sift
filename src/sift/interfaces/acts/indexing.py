from sift.api.embedding import EmbedCollections, EmbedResult
from sift.api.indexing import IndexCollections, IndexResult
from sift.app import embedding, indexing
from sift.app.filing import Readers
from sift.app.scans import Scanning
from sift.interfaces.acts.acting import Act, Acting


def _index(acting: Acting, request: IndexCollections) -> IndexResult:
    readers = Readers(scanning=Scanning(acting.settings.ocr, gpus=acting.settings.gpus))
    return indexing.index(acting.settings.home, request, acting.report, readers=readers)


def _embed(acting: Acting, request: EmbedCollections) -> EmbedResult:
    return embedding.embed(acting.settings.home, request, acting.report, acting.models())


INDEX = Act("index", IndexCollections, IndexResult, _index, writes=True)
EMBED = Act("embed", EmbedCollections, EmbedResult, _embed, writes=True)
