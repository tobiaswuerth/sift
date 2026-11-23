from contextlib import ExitStack
from pathlib import Path

from sift.api.events import Reporter
from sift.api.search import SearchResult, SearchVectors
from sift.app import hits, searching
from sift.app.loading import prepared, unreported
from sift.app.searching import Wants
from sift.app.vector_candidates import RESCORING, Rescoring, alike, nearest_hits, query_vector
from sift.inference.models import Models

EMPTY_REMEDY = "say what to find, e.g. sift vsearch 'how dark matter halos form'"


def vsearch(
    home: Path, request: SearchVectors, models: Models, asked: Rescoring = RESCORING, report: Reporter = unreported
) -> SearchResult:
    searching.checked(request)
    searching.checked_query(request.query, EMPTY_REMEDY)
    narrowing = searching.narrowed(home, request)
    with ExitStack() as cleanup:
        stores = searching.opened(cleanup, narrowing, Wants(model=models.name, vectors_required=True))
        if not any(len(vectors.coded.ids) for _, vectors in stores.embedded()):
            return hits.result(stores, (), partial=False)
        prepared(report, (models.prepare,))
        query = query_vector(models, request.query)
        alike(stores, models.name, query)
        return hits.result(stores, nearest_hits(stores, request, query, asked), partial=False)
