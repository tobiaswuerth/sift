"""Keyword queries as FTS5 reads them, for the postings to answer (see `topk`): terms joined by AND, OR, NOT and
parentheses, and words side by side, which FTS5 joins before anything else ("a NOT b c" is "a NOT (b c)"; then NOT,
AND, OR). Each word is tokenized by FTS5 itself, as the index was: into one term, or into a phrase of several
tokens, one after the other ("first-stage" quoted, `a_b`), which FTS5 checks for the chunks the postings find; a word
of no token (an en dash) matches nothing, as in FTS5: it drops out of words side by side and of OR, and empties an AND.

What else FTS5 reads (prefixes, NEAR, columns, initial tokens) is left to FTS5, and so is a query FTS5 would reject,
so that its error is FTS5's: parse() returns None for those, and rejected() tells the second kind."""

import sqlite3
import threading
from functools import lru_cache

from . import store

OPERATORS = ("AND", "OR", "NOT")
SPACE = " \t\n\r"  # what FTS5 skips between tokens


def _bareword(c):
    """Whether FTS5 takes the character for part of a bareword: non-ASCII, a letter, a digit, an underscore, or 0x1A."""
    return c >= "\x80" or c.isalnum() or c in "_\x1a"


def _lexed(text):
    """The query's tokens: ("word", text), "(" or ")", or an operator; None if it holds what FTS5 has to read, or
    what it would reject."""
    out, i = [], 0
    while i < len(text):
        c = text[i]
        if c in SPACE:
            i += 1
        elif c in "()":
            out.append(c)
            i += 1
        elif c == '"':
            end, word = i + 1, []
            while True:
                quote = text.find('"', end)
                if quote < 0:
                    return None  # unterminated
                word.append(text[end:quote])
                if text[quote + 1 : quote + 2] != '"':
                    break
                word.append('"')
                end = quote + 2
            out.append(("word", "".join(word)))
            i = quote + 1
        elif _bareword(c):
            end = i
            while end < len(text) and _bareword(text[end]):
                end += 1
            word = text[i:end]
            if word == "NEAR":
                return None
            out.append(word if word in OPERATORS else ("word", word))
            i = end
        else:  # * + ^ : { } - , and what FTS5 rejects
            return None
    return out


class _Parser:
    """FTS5's grammar, for the tokens _lexed() gives: or := and (OR and)*; and := not (AND not)*; not := primary
    (NOT primary)*; primary := ( or ) | word+."""

    def __init__(self, tokens):
        self.tokens, self.at = tokens, 0

    def peek(self):
        return self.tokens[self.at] if self.at < len(self.tokens) else None

    def take(self):
        self.at += 1
        return self.tokens[self.at - 1]

    def parts(self, operator, part):
        found = [part()]
        while self.peek() == operator:
            self.take()
            found.append(part())
        return found

    def any(self):
        return _joined("or", [part for part in self.parts("OR", self.all) if part is not NOTHING])

    def all(self):
        parts = self.parts("AND", self.but)
        return NOTHING if NOTHING in parts else _joined("and", parts)

    def but(self):
        node = self.primary()
        while self.peek() == "NOT":
            self.take()
            right = self.primary()
            node = node if NOTHING in (node, right) else ("not", node, right)
        return node

    def primary(self):
        if self.peek() == "(":
            self.take()
            node = self.any()
            if self.take() != ")":
                raise SyntaxError
            return node
        words = []
        while isinstance(self.peek(), tuple):
            words.append(_term(self.take()[1]))
        if not words:
            raise SyntaxError
        return _joined("and", [word for word in words if word is not NOTHING])


NOTHING = ("nothing",)  # what matches no chunk: a word of no token (an en dash), as FTS5 reads it, and an AND with one


def _joined(kind, parts):
    """Parts joined by AND or OR; NOTHING for none."""
    return NOTHING if not parts else parts[0] if len(parts) == 1 else (kind, parts)


def _term(word):
    """A word as FTS5 reads it: a term, a phrase of its tokens, or NOTHING for a word of no token."""
    tokens = tokenize(word)
    if not tokens:
        return NOTHING
    return ("term", tokens[0]) if len(tokens) == 1 else ("phrase", tokens)


def parse(query):
    """The query as an expression of terms (see `topk`), or None if FTS5 is to read it: also a query that matches
    nothing, which FTS5 knows at once."""
    tokens = _lexed(query)
    if not tokens:
        return None
    parser = _Parser(tokens)
    try:
        node = parser.any()
    except (SyntaxError, IndexError, RecursionError):  # RecursionError: parentheses nested hundreds deep
        return None
    return node if parser.peek() is None and node is not NOTHING and _depth(node) <= MOST_DEPTH else None


MOST_DEPTH = 256  # how deeply FTS5 lets a query nest: deeper, it says so itself


def _depth(node):
    """How deeply an expression nests: 1 for a term or phrase alone. A chain of NOTs nests one level each."""
    deepest, todo = 0, [(node, 1)]
    while todo:
        item, depth = todo.pop()
        deepest = max(deepest, depth)
        if item[0] == "not":
            todo += [(item[1], depth + 1), (item[2], depth + 1)]
        elif item[0] in ("and", "or"):
            todo += [(child, depth + 1) for child in item[1]]
    return deepest


def rejected(query):
    """Whether FTS5 rejects the query, its syntax or a column it names: it never reads such a query as written."""
    tokens("")  # the tokenizer, made on first use
    return _tokenizer.rejects(query)


class _Tokenizer:
    """FTS5's tokenizer, as the index uses it, in a table of its own in memory."""

    def __init__(self):
        self.db = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        self.db.execute(f"CREATE VIRTUAL TABLE words USING fts5(word, tokenize='{store.TOKENIZER}')")
        self.db.execute("CREATE VIRTUAL TABLE tokens USING fts5vocab(words, instance)")
        self.db.execute(f"CREATE VIRTUAL TABLE queries USING fts5(title, text, meta, tokenize='{store.TOKENIZER}')")
        self.lock = threading.Lock()

    def __call__(self, text):
        with self.lock:
            self.db.execute("INSERT INTO words(rowid, word) VALUES (1, ?)", (text,))
            try:
                return [t for (t,) in self.db.execute("SELECT term FROM tokens ORDER BY offset")]
            finally:
                self.db.execute("DELETE FROM words")

    def rejects(self, query):
        """Whether FTS5 rejects the query, as asked of an empty table of the keyword index's columns."""
        with self.lock:
            try:
                self.db.execute("SELECT 1 FROM queries WHERE queries MATCH ?", (query,)).fetchall()
            except sqlite3.OperationalError:
                return True
            return False


_tokenizer = None


def tokens(text):
    """The tokens FTS5 makes of a text, in order."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = _Tokenizer()
    return tuple(_tokenizer(text))


@lru_cache(maxsize=4096)
def tokenize(text):
    """The tokens FTS5 makes of a text, in order, remembered: a query's words come again and again."""
    return tokens(text)
