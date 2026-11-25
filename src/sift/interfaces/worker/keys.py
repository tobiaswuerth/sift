import datetime as dt
import secrets
import ssl
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from sift.inference.remote import fingerprint
from sift.storage.database import private_bytes

FOLDER = "worker"
ADMISSION_FILE = "token"
CERTIFICATE = "cert.pem"
PRIVATE_KEY = "key.pem"
TOKEN_BYTES = 32
COMMON_NAME = "sift worker"
VALID_DAYS = 100 * 365


@dataclass(frozen=True, slots=True)
class WorkerKey:
    token: str
    pin: str
    certificate: Path
    private_key: Path

    @property
    def key(self) -> str:
        return f"{self.token}.{self.pin}"


def _made(certificate: Path, private_key: Path, token: Path) -> None:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, COMMON_NAME)])
    now = dt.datetime.now(dt.UTC)
    made = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(days=1))
        .not_valid_after(now + dt.timedelta(days=VALID_DAYS))
        .sign(key, hashes.SHA256())
    )
    pem = serialization.Encoding.PEM
    private_bytes(private_key, key.private_bytes(pem, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    private_bytes(certificate, made.public_bytes(pem))
    private_bytes(token, secrets.token_urlsafe(TOKEN_BYTES).encode())


def worker_key(home: Path) -> WorkerKey:
    folder = home / FOLDER
    token, certificate, private_key = folder / ADMISSION_FILE, folder / CERTIFICATE, folder / PRIVATE_KEY
    if not (token.is_file() and certificate.is_file() and private_key.is_file()):
        _made(certificate, private_key, token)
    pin = fingerprint(ssl.PEM_cert_to_DER_cert(certificate.read_text(encoding="ascii")))
    return WorkerKey(token.read_text(encoding="ascii").strip(), pin, certificate, private_key)
