import json
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from sift.domain.arrays import Ids
from sift.domain.codec.segments import unpacked
from sift.domain.errors import DamagedError
from sift.storage.columns import columns
from sift.storage.database import Count, Reading, first, rows

MOST_POINT = 0x10FFFF
SURROGATES_START, SURROGATES_END = 0xD800, 0xE000
EXPANSIONS = (
    "SELECT term, chunks FROM keyword_terms WHERE term >= ?",
    "SELECT term, chunks FROM keyword_terms WHERE term >= ? AND term < ?",
)
PACK_OF = """SELECT s.id, t.value,
                    (SELECT p.rowid FROM keyword_packs p WHERE p.segment = s.id AND p.first <= t.value
                     ORDER BY p.first DESC LIMIT 1)
             FROM keyword_served s, json_each(?) t"""
RANGES = "SELECT ranges FROM keyword_ranges WHERE segment = ? AND term = ?"
PACKS = "SELECT rowid, segment, lists FROM keyword_packs WHERE rowid IN (SELECT value FROM json_each(?))"
PACK_PLACED = """SELECT lists, places FROM keyword_packs WHERE segment = ? AND first <= ?
                 ORDER BY first DESC LIMIT 1"""
VOLUMES = (
    "SELECT count(*), coalesce(sum(chunks), 0) FROM keyword_terms WHERE term >= ?",
    "SELECT count(*), coalesce(sum(chunks), 0) FROM keyword_terms WHERE term >= ? AND term < ?",
)


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
    lists: bytes


class PackPlaced(NamedTuple):
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


class Volume(NamedTuple):
    words: int
    passages: int


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


def frequencies(transaction: Reading, terms: Sequence[str]) -> dict[str, int]:
    sql = "SELECT term, chunks FROM keyword_terms WHERE term IN (SELECT value FROM json_each(?))"
    return {found.term: found.chunks for found in rows(transaction, Frequency, sql, (json.dumps(list(terms)),))}


def _after(prefix: str) -> str | None:
    kept = prefix.rstrip(chr(MOST_POINT))
    if not kept:
        return None
    following = ord(kept[-1]) + 1
    return kept[:-1] + chr(SURROGATES_END if SURROGATES_START <= following < SURROGATES_END else following)


def _bounded(prefix: str) -> tuple[int, tuple[str, ...]]:
    after = _after(prefix)
    return (0, (prefix,)) if after is None else (1, (prefix, after))


def expansions(transaction: Reading, prefix: str) -> list[str]:
    which, parameters = _bounded(prefix)
    return [found.term for found in rows(transaction, Frequency, EXPANSIONS[which], parameters)]


def volume(transaction: Reading, prefix: str) -> Volume:
    which, parameters = _bounded(prefix)
    (found,) = rows(transaction, Volume, VOLUMES[which], parameters)
    return found


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


def parts(transaction: Reading, terms: Sequence[str]) -> list[Part]:
    asked = _packed(transaction, terms)
    found: list[Part] = []
    for pack in rows(transaction, PackLists, PACKS, (json.dumps(list(asked)),)):
        for held in unpacked(pack.lists, None, asked[pack.id]):
            if held.postings is None:
                ranges = _ranged(transaction, pack.segment, held.term)
                found.append(Part(pack.segment, held.term, held.chunks, 1, ranges))
            else:
                found.append(Part(pack.segment, held.term, held.chunks, 0, held.postings))
    return sorted(found, key=lambda part: (part.segment, part.term))


def blocks(transaction: Reading, segment: int, term: str, numbers: Sequence[int]) -> list[BlockRow]:
    sql = """SELECT number, postings FROM keyword_blocks
             WHERE segment = ? AND term = ? AND number IN (SELECT value FROM json_each(?)) ORDER BY number"""
    return rows(transaction, BlockRow, sql, (segment, term, json.dumps(list(numbers))))


def positions(transaction: Reading, segment: int, term: str, numbers: Sequence[int] | None) -> list[PositionsRow]:
    if numbers is None:
        pack = first(transaction, PackPlaced, PACK_PLACED, (segment, term))
        held = [] if pack is None else unpacked(pack.lists, pack.places, [term])
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
