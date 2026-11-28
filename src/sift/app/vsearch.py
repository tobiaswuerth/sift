from contextlib import ExitStack
from pathlib import Path

from sift.api.search import SearchResult, SearchVectors
from sift.app import hits, searching
from sift.app.loading import prepared
from sift.app.searching import WATCHING, Wants, Watching
from sift.app.vector_candidates import RESCORING, Rescoring, alike, nearest_hits, query_vector
from sift.inference.models import Models

EMPTY_REMEDY = "say what to find, e.g. how dark matter halos form"


def vsearch(
    home: Path,
    request: SearchVectors,
    models: Models,
    asked: Rescoring = RESCORING,
    watching: Watching = WATCHING,
) -> SearchResult:
    searching.checked(request)
    searching.checked_query(request.query, EMPTY_REMEDY)
    expired = searching.expired_after(request.budget, watching.clock)
    narrowing = searching.narrowed(home, request, watching.audience)
    with ExitStack() as cleanup:
        stores = searching.opened(
            cleanup, narrowing, Wants(model=models.name, vectors_required=True, codes=watching.codes), expired
        )
        if not any(len(vectors.coded.ids) for _, vectors in stores.embedded()):
            return hits.result(stores, hits.Answered(()), "vectors")
        prepared(watching.report, (models.prepare,))
        query = query_vector(models, request.query)
        alike(stores, models.name, query)
        return hits.result(stores, nearest_hits(stores, request, query, asked), "vectors")
