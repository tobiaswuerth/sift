from email import policy
from email.message import EmailMessage, MIMEPart
from email.parser import BytesParser

import pytest
from hypothesis import given
from hypothesis import strategies as st

from sift.domain.text.documents import FileRead, Unread
from sift.ingest.mail import MOST_NESTED, NESTED_TOO_DEEPLY, UNREADABLE, mail
from sift.ingest.nesting import nesting
from tests.mails import atoms, failing, mail_bytes, message, nested_parts, read, read_file


@pytest.mark.parametrize("depth", [MOST_NESTED + 1, 5_000])
def test_a_mail_nested_deeper_than_sift_reads_is_unread_before_it_is_parsed(depth: int) -> None:
    assert mail(nested_parts(depth), "a.eml", failing) == Unread(NESTED_TOO_DEEPLY)


def test_a_mail_nested_as_deep_as_sift_reads_is_read() -> None:
    assert isinstance(read(nested_parts(MOST_NESTED), "a.eml", failing), FileRead)


def _attached_in_chain(levels: int) -> bytes:
    inner = b"Subject: 0\r\n\r\nbottom\r\n"
    for level in range(1, levels):
        head = b'Subject: %d\r\nMIME-Version: 1.0\r\nContent-Type: multipart/mixed; boundary="c%d"\r\n\r\n' % (
            level,
            level,
        )
        inner = head + b"--c%d\r\nContent-Type: message/rfc822\r\n\r\n" % level + inner + b"\r\n--c%d--\r\n" % level
    return inner


def test_mail_attached_in_mail_deeper_than_sift_reads_is_unread() -> None:
    assert nesting(_attached_in_chain(10), 1_000) == 18
    assert mail(_attached_in_chain(200), "a.eml", failing) == Unread(NESTED_TOO_DEEPLY)


def test_many_mails_attached_side_by_side_nest_no_deeper_than_one() -> None:
    raw = mail_bytes(message(attachments=[(None, message(body=f"Mail {n}.")) for n in range(3 * MOST_NESTED)]))
    assert nesting(raw, MOST_NESTED) == 2
    assert len(read_file(raw, "a.eml").documents) == 1 + 3 * MOST_NESTED


def _depth(found: MIMEPart) -> int:
    return 1 + max(map(_depth, found.iter_parts()), default=0) if found.is_multipart() else 0


REPORT = b"MIME-Version: 1.0\r\nContent-Type: multipart/report; boundary=r\r\n\r\n--r\r\n"
STATUS = b"Content-Type: message/delivery-status\r\n\r\nReporting-MTA: dns; example.com\r\n\r\n"
STATUS_MAIL = b"Content-Type: message/rfc822\r\n\r\nSubject: bounced\r\n--r--\r\n"
DIGEST = b"Content-Type: multipart/digest; boundary=d\r\n\r\n--d\r\n\r\nSubject: one\r\n\r\nwords\r\n--d--\r\n"
DOUBLED = b"Content-Type: multipart/mixed; boundary=b\r\n\r\n--b\r\n--b--\r\nContent-Type: message/rfc822\r\n\r\nx\r\n"


@pytest.mark.parametrize(
    "raw, measured, made",
    [
        (DIGEST, 2, 2),
        (REPORT + STATUS + STATUS_MAIL, 3, 3),
        (DOUBLED, 2, 2),
        (DOUBLED + b"--b\r\n--b", 2, 2),
    ],
)
def test_digests_reports_and_boundaries_in_a_row_are_measured_as_python_s_parser_reads_them(
    raw: bytes, measured: int, made: int
) -> None:
    parsed = _parsed(raw)
    assert parsed is not None
    assert (nesting(raw, 1_000), _depth(parsed)) == (measured, made)


def _parsed(raw: bytes) -> EmailMessage | None:
    try:
        return BytesParser(EmailMessage, policy=policy.default).parsebytes(raw)
    except UNREADABLE:
        return None


def _attaching(parts: list[EmailMessage]) -> EmailMessage:
    return message(attachments=[(None, each) for each in parts])


def _nested(inner: st.SearchStrategy[EmailMessage]) -> st.SearchStrategy[EmailMessage]:
    return st.lists(inner, min_size=1, max_size=3).map(_attaching)


def _leaf(words: str) -> EmailMessage:
    return message(body=words)


leaves = st.text("ab ", min_size=1, max_size=5).map(_leaf)
trees = st.recursive(leaves, _nested, max_leaves=12)


@given(trees)
def test_the_nesting_measured_of_a_mail_is_the_nesting_python_s_parser_makes_of_it(tree: EmailMessage) -> None:
    raw = mail_bytes(tree)
    parsed = _parsed(raw)
    assert parsed is not None
    assert nesting(raw, 1_000) == _depth(parsed)


@given(st.lists(atoms, max_size=40).map(b"".join))
def test_the_nesting_measured_of_any_mail_is_at_least_the_nesting_python_s_parser_makes_of_it(raw: bytes) -> None:
    try:
        measured = nesting(raw, 1_000)
    except UNREADABLE:
        return
    parsed = _parsed(raw)
    assert parsed is None or measured >= _depth(parsed)
