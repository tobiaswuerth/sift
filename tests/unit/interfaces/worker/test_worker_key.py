import ssl
import sys
from pathlib import Path

import pytest

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


@pytest.mark.skipif(sys.platform == "win32", reason="Windows guards files with ACLs; a user's folder is private")
def test_the_key_files_are_the_user_s_alone(tmp_path: Path) -> None:
    worker_key(tmp_path)
    for name in ("token", "cert.pem", "key.pem"):
        assert (tmp_path / "worker" / name).stat().st_mode & 0o077 == 0
