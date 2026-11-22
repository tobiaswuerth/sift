import json
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

from sift.domain.arrays import Ids
from sift.storage.database import Reading, rows

MOST_POINT = 0x10FFFF
SURROGATES_START, SURROGATES_END = 0xD800, 0xE000


class SegmentRow(NamedTuple):
    id: int
    first: int
    last: int
    chunks: int
    tokens: int
    average: float


class Part(NamedTuple):
    segment: int
    term: str
    chunks: int
    ranged: int
    data: bytes


class BlockRow(NamedTuple):
    number: int
    postings: bytes


class PositionsRow(NamedTuple):
    number: int
    positions: bytes


class Frequency(NamedTuple):
    term: str
    chunks: int


class Chunk(NamedTuple):
    id: int


class Owner(NamedTuple):
    id: int
    document: int


class LengthsRow(NamedTuple):
    block: int
    lengths: bytes


def segments(transaction: Reading) -> list[SegmentRow]:
    sql = "SELECT id, first, last, chunks, tokens, average FROM keyword_segments ORDER BY id"
    return rows(transaction, SegmentRow, sql)


def dead(transaction: Reading) -> Ids:
    found = rows(transaction, Chunk, "SELECT chunk FROM keyword_dead ORDER BY chunk")
    return np.array([chunk.id for chunk in found], np.int64)


def frequencies(transaction: Reading, terms: Sequence[str]) -> dict[str, int]:
    sql = "SELECT term, chunks FROM keyword_terms WHERE term IN (SELECT value FROM json_each(?))"
    return {found.term: found.chunks for found in rows(transaction, Frequency, sql, (json.dumps(list(terms)),))}


def _after(prefix: str) -> str | None:
    kept = prefix.rstrip(chr(MOST_POINT))
    if not kept:
        return None
    following = ord(kept[-1]) + 1
    return kept[:-1] + chr(SURROGATES_END if SURROGATES_START <= following < SURROGATES_END else following)


def expansions(transaction: Reading, prefix: str) -> list[str]:
    after = _after(prefix)
    sql, parameters = (
        ("SELECT term, chunks FROM keyword_terms WHERE term >= ? AND term < ?", (prefix, after))
        if after is not None
        else ("SELECT term, chunks FROM keyword_terms WHERE term >= ?", (prefix,))
    )
    return [found.term for found in rows(transaction, Frequency, sql, parameters)]


def parts(transaction: Reading, terms: Sequence[str]) -> list[Part]:
    sql = """SELECT segment, term, chunks, ranges IS NOT NULL, coalesce(ranges, postings) FROM keyword_postings
             WHERE segment IN (SELECT id FROM keyword_segments) AND term IN (SELECT value FROM json_each(?))
             ORDER BY segment, term"""
    return rows(transaction, Part, sql, (json.dumps(list(terms)),))


def blocks(transaction: Reading, segment: int, term: str, numbers: Sequence[int]) -> list[BlockRow]:
    sql = """SELECT number, postings FROM keyword_blocks
             WHERE segment = ? AND term = ? AND number IN (SELECT value FROM json_each(?)) ORDER BY number"""
    return rows(transaction, BlockRow, sql, (segment, term, json.dumps(list(numbers))))


def positions(transaction: Reading, segment: int, term: str, numbers: Sequence[int] | None) -> list[PositionsRow]:
    if numbers is None:
        sql = "SELECT 0, positions FROM keyword_postings WHERE segment = ? AND term = ?"
        return rows(transaction, PositionsRow, sql, (segment, term))
    sql = """SELECT number, positions FROM keyword_blocks
             WHERE segment = ? AND term = ? AND number IN (SELECT value FROM json_each(?)) ORDER BY number"""
    return rows(transaction, PositionsRow, sql, (segment, term, json.dumps(list(numbers))))


def lengths(transaction: Reading, segment: int, numbers: Sequence[int]) -> list[LengthsRow]:
    sql = """SELECT block, lengths FROM keyword_lengths
             WHERE segment = ? AND block IN (SELECT value FROM json_each(?)) ORDER BY block"""
    return rows(transaction, LengthsRow, sql, (segment, json.dumps(list(numbers))))


def owners(transaction: Reading, chunks: Ids) -> dict[int, int]:
    sql = "SELECT id, document FROM chunks WHERE id IN (SELECT value FROM json_each(?))"
    return {found.id: found.document for found in rows(transaction, Owner, sql, (json.dumps(chunks.tolist()),))}
