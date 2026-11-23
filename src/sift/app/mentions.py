from dataclasses import replace
from pathlib import Path

from sift.api.errors import InvalidRequestError
from sift.api.links import FindMentions
from sift.api.search import SearchResult
from sift.app import searching
from sift.app.collections import chosen_or_active
from sift.app.notes import Note, note_of, read_in
from sift.app.search import Asked, Wanted, searched_as
from sift.domain.query.grammar import any_phrase
from sift.domain.text.mentions import names, phrases
from sift.retrieval.keyword.search import TUNING, Tuning, never
from sift.storage.database import Reading
from sift.storage.documents import fields_by_document
from sift.storage.links import titled

ALSO_REMEDY = "name each field, e.g. --also emails,phone"


def _names(note: Note, also: tuple[str, ...]) -> list[str]:
    def read(transaction: Reading) -> list[str]:
        document = note.find(transaction)
        return names(
            titled(transaction, [document])[document].path, fields_by_document(transaction, [document])[document], also
        )

    return read_in(note.collection, read)


def mentions(home: Path, request: FindMentions, tuning: Tuning = TUNING) -> SearchResult:
    searching.checked(request)
    for field in request.also:
        if not field.strip():
            raise InvalidRequestError(f"--also names front matter fields, e.g. emails, not {field!r}", ALSO_REMEDY)
    named = note_of(home, chosen_or_active(home, ()), request.id)
    expression = any_phrase(phrases(_names(named.note, request.also)))
    wanted = Wanted(request.k, not request.all_passages, tuning, never)
    found = searched_as(Asked(request, expression, None, wanted, searching.narrowed(home, request, named.note)))
    said = {each.collection for each in found.unsearched}
    passed = tuple(each for each in named.unsearched if each.collection not in said)
    return replace(found, unsearched=(*found.unsearched, *passed))
