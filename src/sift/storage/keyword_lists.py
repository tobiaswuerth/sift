import json
from collections.abc import Iterator, Sequence
from typing import NamedTuple

import numpy as np

from sift.domain.arrays import Ids
from sift.domain.codec.segments import PackedList, counted, unpacked, within
from sift.domain.errors import DamagedError
from sift.storage.columns import columns
from sift.storage.database import Count, Reading, batches, first, rows

MOST_POINT = 0x10FFFF
SURROGATES_START, SURROGATES_END = 0xD800, 0xE000
WITHIN = """SELECT first, lists FROM keyword_packs WHERE segment = :segment AND first >= coalesce(
                (SELECT max(first) FROM keyword_packs WHERE segment = :segment AND first <= :low), :low)"""
PACKS_WITHIN = (f"{WITHIN} ORDER BY first", f"{WITHIN} AND first < :high ORDER BY first")
PACKS_AT_ONCE = 64
GONE = "SELECT term, sum(chunks) FROM keyword_gone WHERE term IN (SELECT value FROM json_each(?)) GROUP BY term"
GONE_WITHIN = "SELECT term, sum(chunks) FROM keyword_gone WHERE term >= :low AND term <= :through GROUP BY term"
GONE_PAST = "more passages of a word are counted gone than its lists hold in the keyword index"
PACK_OF = """SELECT s.id, t.value,
                    (SELECT p.rowid FROM keyword_packs p WHERE p.segment = s.id AND p.first <= t.value
                     ORDER BY p.first DESC LIMIT 1)
             FROM keyword_served s, json_each(?) t"""
RANGES = "SELECT ranges FROM keyword_ranges WHERE segment = ? AND term = ?"
PACKS = "SELECT rowid, segment, first, lists FROM keyword_packs WHERE rowid IN (SELECT value FROM json_each(?))"
PACK_PLACED = """SELECT first, lists, places FROM keyword_packs WHERE segment = ? AND first <= ?
                 ORDER BY first DESC LIMIT 1"""


class SegmentRow(NamedTuple):
    id: int
    first: int
    last: int
    chunks: int
    average: float


class Part(NamedTuple):
    segment: int
    term: str
    chunks: int
    ranged: int
    data: bytes


class PackOf(NamedTuple):
    segment: int
    term: str
    pack: int | None


class PackLists(NamedTuple):
    id: int
    segment: int
    first: str
    lists: bytes


class PackPlaced(NamedTuple):
    first: str
    lists: bytes
    places: bytes


class RangesRow(NamedTuple):
    ranges: bytes


class BlockRow(NamedTuple):
    number: int
    postings: bytes


class PositionsRow(NamedTuple):
    number: int
    positions: bytes


class Frequency(NamedTuple):
    term: str
    chunks: int


class PackHeads(NamedTuple):
    first: str
    lists: bytes


class LengthsRow(NamedTuple):
    block: int
    lengths: bytes


def segments(transaction: Reading) -> list[SegmentRow]:
    sql = "SELECT id, first, last, chunks, average FROM keyword_served ORDER BY first"
    return rows(transaction, SegmentRow, sql)


class IdsRow(NamedTuple):
    ids: str


def dead_within(transaction: Reading, low: int, high: int) -> Ids:
    sql = """SELECT json_group_array(place) FROM
             (SELECT place FROM keyword_dead WHERE place BETWEEN ? AND ? ORDER BY place)"""
    (found,) = rows(transaction, IdsRow, sql, (low, high))
    return np.array(json.loads(found.ids), np.int64)


def chunks_placed(transaction: Reading, places: Sequence[int]) -> tuple[Ids, Ids, Ids]:
    sql = "SELECT place, id, document FROM chunks WHERE place IN (SELECT value FROM json_each(?)) ORDER BY place"
    found, chunks, documents = columns(transaction, sql, (json.dumps(list(places)),), 3)
    return found, chunks, documents


def gone(transaction: Reading, terms: Sequence[str]) -> dict[str, int]:
    return {found.term: found.chunks for found in rows(transaction, Frequency, GONE, (json.dumps(list(terms)),))}


def _after(prefix: str) -> str | None:
    kept = prefix.rstrip(chr(MOST_POINT))
    if not kept:
        return None
    following = ord(kept[-1]) + 1
    return kept[:-1] + chr(SURROGATES_END if SURROGATES_START <= following < SURROGATES_END else following)


def living(held: int, deleted: int) -> int:
    if deleted > held:
        raise DamagedError(GONE_PAST)
    return held - deleted


def _packs_within(transaction: Reading, low: str, high: str | None) -> Iterator[tuple[int, list[PackHeads]]]:
    for segment in segments(transaction):
        bounds = {"segment": segment.id, "low": low, "high": high}
        for packs in batches(transaction, PackHeads, PACKS_WITHIN[high is not None], bounds, PACKS_AT_ONCE):
            yield segment.id, packs


def prefixed(transaction: Reading, prefix: str, most: int) -> dict[str, int]:
    high = _after(prefix)
    held: dict[str, int] = {}
    deleted: dict[str, int] = {}
    certain, known = 0, prefix
    for _, packs in _packs_within(transaction, prefix, high):
        found = counted([(pack.first, pack.lists) for pack in packs], prefix, high)
        through = found[-1][0] if found else known
        if through > known:
            since = rows(transaction, Frequency, GONE_WITHIN, {"low": known, "through": through})
            deleted |= {each.term: each.chunks for each in since}
            known = through
        for term, chunks in found:
            before, since_gone = held.get(term, 0), deleted.get(term, 0)
            held[term] = before + chunks
            certain += before <= since_gone < before + chunks
        if certain > most:
            break
    else:
        live = ((term, living(chunks, deleted.get(term, 0))) for term, chunks in sorted(held.items()))
        return {term: chunks for term, chunks in live if chunks}
    surely = ((term, chunks - deleted.get(term, 0)) for term, chunks in sorted(held.items()))
    return {term: chunks for term, chunks in surely if chunks > 0}


def _packed(transaction: Reading, terms: Sequence[str]) -> dict[int, list[str]]:
    held: dict[int, list[str]] = {}
    for found in rows(transaction, PackOf, PACK_OF, (json.dumps(sorted(set(terms))),)):
        if found.pack is not None:
            held.setdefault(found.pack, []).append(found.term)
    return held


def _ranged(transaction: Reading, segment: int, term: str) -> bytes:
    found = first(transaction, RangesRow, RANGES, (segment, term))
    if found is None:
        raise DamagedError("the ranges of a long list are missing from the keyword index")
    return found.ranges


def _part(transaction: Reading, segment: int, held: PackedList) -> Part:
    if held.postings is None:
        return Part(segment, held.term, held.chunks, 1, _ranged(transaction, segment, held.term))
    return Part(segment, held.term, held.chunks, 0, held.postings)


def parts(transaction: Reading, terms: Sequence[str]) -> list[Part]:
    asked = _packed(transaction, terms)
    found = [
        _part(transaction, pack.segment, held)
        for pack in rows(transaction, PackLists, PACKS, (json.dumps(list(asked)),))
        for held in unpacked(pack.first, pack.lists, None, asked[pack.id])
    ]
    return sorted(found, key=lambda part: (part.segment, part.term))


def parts_within(transaction: Reading, prefix: str) -> list[Part]:
    high = _after(prefix)
    found = [
        _part(transaction, segment, held)
        for segment, packs in _packs_within(transaction, prefix, high)
        for pack in packs
        for held in within(pack.first, pack.lists, prefix, high)
    ]
    return sorted(found, key=lambda part: (part.segment, part.term))


def blocks(transaction: Reading, segment: int, term: str, numbers: Sequence[int]) -> list[BlockRow]:
    sql = """SELECT number, postings FROM keyword_blocks
             WHERE segment = ? AND term = ? AND number IN (SELECT value FROM json_each(?)) ORDER BY number"""
    return rows(transaction, BlockRow, sql, (segment, term, json.dumps(list(numbers))))


def positions(transaction: Reading, segment: int, term: str, numbers: Sequence[int] | None) -> list[PositionsRow]:
    if numbers is None:
        pack = first(transaction, PackPlaced, PACK_PLACED, (segment, term))
        held = [] if pack is None else unpacked(pack.first, pack.lists, pack.places, [term])
        return [PositionsRow(0, each.positions) for each in held if each.positions is not None]
    sql = """SELECT number, positions FROM keyword_blocks
             WHERE segment = ? AND term = ? AND number IN (SELECT value FROM json_each(?)) ORDER BY number"""
    return rows(transaction, PositionsRow, sql, (segment, term, json.dumps(list(numbers))))


def lengths(transaction: Reading, segment: int, numbers: Sequence[int]) -> list[LengthsRow]:
    sql = """SELECT block, lengths FROM keyword_lengths
             WHERE segment = ? AND block IN (SELECT value FROM json_each(?)) ORDER BY block"""
    return rows(transaction, LengthsRow, sql, (segment, json.dumps(list(numbers))))


def keyword_generation(transaction: Reading) -> int:
    (held,) = rows(transaction, Count, "SELECT keyword_changes FROM counts")
    return held.n
