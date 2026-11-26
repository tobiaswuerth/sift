import ssl
import sys
from pathlib import Path

import pytest

from sift.api.errors import StoreUnavailableError
from sift.inference.remote import fingerprint
from sift.interfaces.worker.keys import worker_key


def test_the_worker_makes_its_key_once_and_keeps_it(tmp_path: Path) -> None:
    first = worker_key(tmp_path / "home")
    assert first == worker_key(tmp_path / "home")
    assert first.key == f"{first.token}.{first.pin}"
    assert len(first.token) >= 43
    assert sorted(path.name for path in (tmp_path / "home" / "worker").iterdir()) == ["cert.pem", "key.pem", "token"]


def test_the_key_names_the_certificate_the_worker_shows(tmp_path: Path) -> None:
    found = worker_key(tmp_path)
    shown = ssl.PEM_cert_to_DER_cert(found.certificate.read_text(encoding="ascii"))
    assert fingerprint(shown) == found.pin


def test_another_home_makes_another_key(tmp_path: Path) -> None:
    one, other = worker_key(tmp_path / "one"), worker_key(tmp_path / "other")
    assert (one.token == other.token, one.pin == other.pin) == (False, False)


def test_a_key_with_a_file_missing_is_made_anew(tmp_path: Path) -> None:
    first = worker_key(tmp_path)
    (tmp_path / "worker" / "token").unlink()
    again = worker_key(tmp_path)
    assert (again.token == first.token, again.pin == first.pin) == (False, False)


@pytest.mark.parametrize(
    "name, kept",
    [
        ("cert.pem", b"no certificate"),
        ("cert.pem", b""),
        ("cert.pem", bytes([255, 254])),
        ("cert.pem", b"-----BEGIN CERTIFICATE-----" + bytes([10]) + b"!" + bytes([10]) + b"-----END CERTIFICATE-----"),
        ("token", b""),
        ("token", b"  "),
        ("token", bytes([255])),
    ],
)
def test_a_key_that_cannot_be_read_is_refused_saying_to_make_a_new_one(tmp_path: Path, name: str, kept: bytes) -> None:
    worker_key(tmp_path)
    (tmp_path / "worker" / name).write_bytes(kept)
    with pytest.raises(StoreUnavailableError, match=r"the worker's key in .*worker cannot be read: ") as refused:
        worker_key(tmp_path)
    assert refused.value.remedy == "delete that folder: the worker makes a new key, to copy to the machines that use it"


@pytest.mark.skipif(sys.platform == "win32", reason="Windows guards files with ACLs; a user's folder is private")
def test_the_key_files_are_the_user_s_alone(tmp_path: Path) -> None:
    worker_key(tmp_path)
    for name in ("token", "cert.pem", "key.pem"):
        assert (tmp_path / "worker" / name).stat().st_mode & 0o077 == 0
