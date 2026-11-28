import re

MOST_HEAD_BYTES = 256 << 10
MOST_COMMENT_NESTING = 100
TOO_LARGE = f"its headers hold more than {MOST_HEAD_BYTES >> 10} KiB"
NESTED_TOO_DEEPLY = f"its headers nest comments more than {MOST_COMMENT_NESTING} deep"
FOLDED = frozenset(b" \t")
OPEN, CLOSE, QUOTE, ESCAPE = b'()"\\'
LINE_END = re.compile(rb"\r\n|\r|\n")


class UnreadableHeadError(ValueError):
    pass


def _nests_deeper_than(field: bytes, most: int) -> bool:
    level, quoted, escaped = 0, False, False
    for byte in field:
        if escaped:
            escaped = False
        elif byte == ESCAPE:
            escaped = True
        elif quoted:
            quoted = byte != QUOTE
        elif byte == QUOTE:
            quoted = level == 0
        elif byte == OPEN:
            level += 1
            if level > most:
                return True
        elif byte == CLOSE:
            level = max(level - 1, 0)
    return False


def comments_deeper_than(head: bytes, most: int) -> bool:
    return head.count(b"(") > most and any(_nests_deeper_than(field, most) for field in _fields(head))


def _fields(head: bytes) -> list[bytes]:
    fields: list[bytearray] = []
    for line in LINE_END.split(head):
        if fields and line[:1] and line[0] in FOLDED:
            fields[-1] += line
        else:
            fields.append(bytearray(line))
    return [bytes(field) for field in fields]


def refuse_unreadable(head: bytes) -> None:
    if len(head) > MOST_HEAD_BYTES:
        raise UnreadableHeadError(TOO_LARGE)
    if comments_deeper_than(head, MOST_COMMENT_NESTING):
        raise UnreadableHeadError(NESTED_TOO_DEEPLY)
