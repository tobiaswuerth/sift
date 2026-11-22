import re
from collections.abc import Sequence
from dataclasses import dataclass

from sift.api.errors import InvalidRequestError
from sift.domain.query.expressions import NOTHING, And, Expression, Not, Nothing, Or, Phrase, Prefix, Term, leaves
from sift.domain.text.tokens import tokens

SPACES = frozenset(" \t\n\r\f\v\x1c\x1d\x1e\x1f\N{NULL}")
QUOTE, OPEN, CLOSE, STAR = '"', "(", ")", "*"
OUTSIDE_WORDS = SPACES | {QUOTE, OPEN, CLOSE}
AND, OR, NOT = "AND", "OR", "NOT"
OPERATORS = frozenset({AND, OR, NOT})
NEAR = "NEAR"
MOST_DEPTH = 100
MOST_LEAVES = 1000
FEWEST_RELAXED = 2
SURROGATE = re.compile(f"[{chr(0xD800)}-{chr(0xDFFF)}]")
WORDLIKE = re.compile(r"\w")
REMEDY = 'write words, "phrases", prefix*, AND, OR, NOT and (parentheses), e.g. (bert OR roberta) AND "fine tuning"'
PREFIX_REMEDY = "end one word of letters and digits with *, e.g. superconduct*"


@dataclass(frozen=True, slots=True)
class Word:
    text: str
    quoted: bool


type Lexeme = Word | str
type Parsed = tuple[Expression, int, int]


def _refused(problem: str, remedy: str = REMEDY) -> InvalidRequestError:
    return InvalidRequestError(f"the query is not understood: {problem}", remedy)


def _quoted(query: str, at: int) -> tuple[Word, int]:
    parts: list[str] = []
    start = at + 1
    while (end := query.find(QUOTE, start)) >= 0:
        parts.append(query[start:end])
        if query[end + 1 : end + 2] != QUOTE:
            return Word("".join(parts), quoted=True), end + 1
        parts.append(QUOTE)
        start = end + 2
    raise _refused(f"a quote is not closed: {query[at:]}", 'close it, or write "" for a quote inside quotes')


def _bare(query: str, at: int) -> tuple[Lexeme, int]:
    end = at
    while end < len(query) and query[end] not in OUTSIDE_WORDS:
        end += 1
    text = query[at:end]
    if text == NEAR and query[end:].lstrip("".join(SPACES)).startswith(OPEN):
        raise _refused("NEAR is not supported", 'write the words side by side, or as a "phrase"')
    return (text if text in OPERATORS else Word(text, quoted=False)), end


def lexemes(query: str) -> tuple[Lexeme, ...]:
    found: list[Lexeme] = []
    at = 0
    while at < len(query):
        character = query[at]
        if character in SPACES:
            at += 1
            continue
        if character in (OPEN, CLOSE):
            found.append(character)
            at += 1
            continue
        lexeme, at = _quoted(query, at) if character == QUOTE else _bare(query, at)
        found.append(lexeme)
    return tuple(found)


def _prefix(text: str) -> Prefix | Nothing:
    body = text.removesuffix(STAR)
    if not body or STAR in body:
        raise _refused(f"{text!r}: * ends a prefix", PREFIX_REMEDY)
    found = tokens(body)
    if len(found) > 1:
        raise _refused(f"{text!r}: a prefix is one word", PREFIX_REMEDY)
    return Prefix(found[0]) if found else NOTHING


def leaf_of(word: Word) -> Expression:
    if not word.quoted and STAR in word.text:
        return _prefix(word.text)
    found = tokens(word.text)
    if not found:
        return NOTHING
    return Term(found[0]) if len(found) == 1 else Phrase(tuple(found))


def _joined(kind: type[And] | type[Or], parts: Sequence[Parsed]) -> tuple[Expression, int]:
    kept = [(expression, depth) for expression, depth, _ in parts if expression != NOTHING]
    if not kept:
        return NOTHING, 1
    if len(kept) == 1:
        return kept[0]
    flat = tuple(
        inner
        for expression, _ in kept
        for inner in (expression.parts if isinstance(expression, kind) else (expression,))
    )
    return kind(flat), 1 + max(depth for _, depth in kept)


def _missing(items: Sequence[Lexeme], at: int) -> InvalidRequestError:
    if at >= len(items):
        return _refused("it ends where words are missing")
    item = items[at]
    if item == CLOSE:
        return _refused("parentheses hold nothing" if at and items[at - 1] == OPEN else '")" where words are missing')
    return _refused(f"{item} needs words before and after it")


def _group(items: Sequence[Lexeme], at: int, nesting: int) -> Parsed:
    if nesting >= MOST_DEPTH:
        raise _refused(f"parentheses nested more than {MOST_DEPTH} deep")
    expression, depth, at = _any(items, at + 1, nesting + 1)
    if at >= len(items) or items[at] != CLOSE:
        raise _refused('a "(" is not closed')
    return expression, depth, at + 1


def _adjacent(items: Sequence[Lexeme], at: int, nesting: int) -> Parsed:
    parts: list[Parsed] = []
    while at < len(items) and (isinstance(items[at], Word) or items[at] == OPEN):
        item = items[at]
        if isinstance(item, Word):
            parts.append((leaf_of(item), 1, at + 1))
            at += 1
        else:
            parts.append(_group(items, at, nesting))
            at = parts[-1][2]
    if not parts:
        raise _missing(items, at)
    return (*_joined(And, parts), at)


def _deep(depth: int) -> int:
    if depth > MOST_DEPTH:
        raise _refused(f"it nests more than {MOST_DEPTH} deep")
    return depth


def _but(items: Sequence[Lexeme], at: int, nesting: int) -> Parsed:
    expression, depth, at = _adjacent(items, at, nesting)
    while at < len(items) and items[at] == NOT:
        dropped, other, at = _adjacent(items, at + 1, nesting)
        if NOTHING not in (expression, dropped):
            expression, depth = Not(expression, dropped), _deep(1 + max(depth, other))
    return expression, depth, at


def _parts(items: Sequence[Lexeme], at: int, operator: str, nesting: int) -> list[Parsed]:
    part = _but if operator == AND else _all
    found = [part(items, at, nesting)]
    while found[-1][2] < len(items) and items[found[-1][2]] == operator:
        found.append(part(items, found[-1][2] + 1, nesting))
    return found


def _all(items: Sequence[Lexeme], at: int, nesting: int) -> Parsed:
    parts = _parts(items, at, AND, nesting)
    end = parts[-1][2]
    if any(expression == NOTHING for expression, _, _ in parts):
        return NOTHING, 1, end
    expression, depth = _joined(And, parts)
    return expression, _deep(depth), end


def _any(items: Sequence[Lexeme], at: int, nesting: int) -> Parsed:
    parts = _parts(items, at, OR, nesting)
    expression, depth = _joined(Or, parts)
    return expression, _deep(depth), parts[-1][2]


def parsed(query: str) -> Expression:
    if SURROGATE.search(query):
        raise _refused("it holds half a surrogate pair", "send the query as text in UTF-8")
    items = lexemes(query)
    if not items:
        return NOTHING
    expression, _, at = _any(items, 0, 0)
    if at < len(items):
        raise _refused('a ")" closes nothing')
    if len(leaves(expression)) > MOST_LEAVES:
        raise _refused(f"it holds more than {MOST_LEAVES:,} words", "ask with fewer words")
    return expression


def relaxed(query: str) -> Expression | None:
    items = lexemes(query)
    if any(item in OPERATORS for item in items):
        return None
    words = [item for item in items if isinstance(item, Word) and WORDLIKE.search(item.text)]
    if len(words) < FEWEST_RELAXED:
        return None
    expression, _ = _joined(Or, [(leaf_of(word), 1, 0) for word in words])
    return None if expression == NOTHING else expression
