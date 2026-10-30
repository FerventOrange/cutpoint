"""Private TLS identity for the local CLIProxyAPI listener."""

from __future__ import annotations

import contextlib
import ipaddress
import os
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

from cutpoint.assistant.private_files import private_directory, private_file


def ensure_identity(root: Path) -> tuple[Path, Path]:
    """Create a fresh local leaf certificate and key for one daemon start."""

    private_directory(root)
    cert_path, key_path = root / "proxy-cert.pem", root / "proxy-key.pem"
    for path in (cert_path, key_path):
        if path.exists() or path.is_symlink():
            private_file(path)

    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Cutpoint local proxy")])
    now = datetime.now(UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(
            x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
            critical=False,
        )
        .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=True)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    material = (
        (cert_path, cert.public_bytes(serialization.Encoding.PEM)),
        (
            key_path,
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            ),
        ),
    )
    for path, content in material:
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=root)
        temp_path = Path(temp_name)
        try:
            os.close(fd)
            fd = -1
            private_file(temp_path)
            temp_path.write_bytes(content)
            temp_path.replace(path)
        finally:
            if fd >= 0:
                with contextlib.suppress(OSError):
                    os.close(fd)
            with contextlib.suppress(OSError):
                temp_path.unlink()
    return cert_path, key_path
