from sift.domain.text.documents import Conversion, Unread
from sift.ingest.mail import mail
from tests.mails import only


def never_converted(_raw: bytes, _name: str) -> Conversion:
    raise AssertionError("nothing is converted of a mail refused before it is parsed")


def commented(depth: int) -> bytes:
    nested = b"(" * depth + b")" * depth
    return b"From: a@example.com " + nested + b"\r\nSubject: Nested\r\n\r\nThe body.\r\n"


def test_a_header_whose_comments_nest_deeper_than_sift_reads_is_unreadable() -> None:
    assert mail(commented(5000), "m.eml", never_converted) == Unread("its headers nest comments more than 100 deep")
    assert mail(commented(101), "m.eml", never_converted) == Unread("its headers nest comments more than 100 deep")
    assert only(commented(100), "m.eml").title == "Nested"


def test_comments_are_counted_per_header_and_not_inside_quotes() -> None:
    quoted = b'To: "' + b"(" * 300 + b'" <b@example.com>\r\n'
    each = b"".join(b"X-%d: (\r\n" % n for n in range(300))
    raw = b"From: a@example.com\r\n" + quoted + each + b"Subject: Many\r\n\r\nThe body.\r\n"
    assert only(raw, "m.eml").title == "Many"


def test_comments_closed_or_escaped_nest_no_deeper_and_folded_lines_continue_theirs() -> None:
    closed = b"From: a@example.com " + b"(a)" * 300 + b"\r\n"
    escaped = b"Cc: c@example.com (" + b"\\(" * 300 + b")\r\n"
    raw = closed + escaped + b"Subject: Flat\r\n\r\nThe body.\r\n"
    assert only(raw, "m.eml").title == "Flat"
    folded = b"From: a@example.com (\r\n" + b" (\r\n" * 150 + b"Subject: Folded\r\n\r\nThe body.\r\n"
    assert mail(folded, "m.eml", never_converted) == Unread("its headers nest comments more than 100 deep")


def test_a_mail_whose_headers_hold_more_than_sift_reads_is_unreadable_before_it_is_parsed() -> None:
    def headed(size: int) -> bytes:
        start = b"Subject: Long\r\nX-Long: "
        return start + b"x" * (size - len(start) - 2) + b"\r\n\r\nThe body.\r\n"

    assert mail(headed((256 << 10) + 1), "m.eml", never_converted) == Unread("its headers hold more than 256 KiB")
    assert only(headed(256 << 10), "m.eml").title == "Long"
