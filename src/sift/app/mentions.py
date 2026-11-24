import time
from pathlib import Path

from sift.api.errors import InvalidRequestError
from sift.api.links import FindMentions
from sift.api.search import SearchResult
from sift.app import hits, searching
from sift.app.keyword_candidates import Wanted, keyword_result
from sift.app.linked import LeftOut
from sift.app.notes import Note, note_of, read_in
from sift.app.searching import Clock
from sift.domain.query.grammar import MOST_WORDS, any_phrase, words_in
from sift.domain.text.mentions import names, phrases
from sift.retrieval.keyword.search import TUNING, Tuning
from sift.storage.database import Reading
from sift.storage.documents import fields_by_document
from sift.storage.links import titled

ALSO_REMEDY = "name each field, e.g. emails and phone"
SHORT_REMEDY = "give the note a longer alias, or name more of its fields to look for"


def _names(note: Note, also: tuple[str, ...]) -> list[str]:
    def read(transaction: Reading) -> list[str]:
        document = note.find(transaction)
        row = titled(transaction, [document])[document]
        return names(row.path, row.key, row.title, fields_by_document(transaction, [document])[document], also)

    return read_in(note.collection, read)


def mentions(
    home: Path, request: FindMentions, tuning: Tuning = TUNING, clock: Clock = time.perf_counter
) -> SearchResult:
    searching.checked(request)
    for field in request.also:
        if not field.strip():
            said = f"names front matter fields, e.g. emails, not {field!r}"
            raise InvalidRequestError(said, ALSO_REMEDY, "also")
    named = note_of(home, request.id)
    found = _names(named.note, request.also)
    looked_for = phrases(found)
    if not looked_for:
        said = ", ".join(map(repr, found))
        raise InvalidRequestError(f"the note's names are too short to search for: {said}", SHORT_REMEDY)
    expression = any_phrase(looked_for)
    if words_in(expression) > MOST_WORDS:
        remedy = "name fewer fields to look for too" if request.also else "give it fewer or shorter names"
        raise InvalidRequestError(f"the note's names hold more than {MOST_WORDS:,} words", remedy)
    expired = searching.expired_after(request.budget, clock)
    wanted = Wanted(request.wanted, not request.all_passages, tuning, expired)
    narrowing = searching.narrowed(home, request, LeftOut(named.note, linking=True))
    return hits.also_unsearched(keyword_result(narrowing, request, wanted, expression), named.unsearched)
