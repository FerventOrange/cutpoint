"""Durable CLIProxyAPI credentials backed by the OS keyring."""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import os
import re
import secrets
import stat
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from cutpoint.assistant import keys
from cutpoint.assistant.private_files import private_directory, private_file
from cutpoint.ffmpeg_utils import FolderBusyError, lock_path, swap_lock

_PREFIX = "proxy-credentials-v1"
_CHUNK = 1000
_MAX_CHUNKS = 100_000
_GENERATION = re.compile(r"[0-9a-f]{32}\Z")
_WINDOWS_RESERVED = {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "CLOCK$"}
_WINDOWS_DEVICE = re.compile(r"(?:COM|LPT)[1-9¹²³]\Z")
_STAGE_FILE = ".cutpoint-credential-stage"
_RECOVERY_FILE = "credential-recovery.enc"
_RECOVERY_MAGIC = b"CPCR1"
_RECOVERY_KEY = "credential-recovery-key"
logger = logging.getLogger(__name__)


def _safe_name(name: object) -> bool:
    if not isinstance(name, str) or name in (".", "..") or not name.lower().endswith(".json"):
        return False
    if name.endswith((".", " ")) or any(
        ord(char) < 32 or ord(char) == 127 or char in '/\\<>:"|?*' for char in name
    ):
        return False
    base = name[:-5]
    if base in ("", ".", ".."):
        return False
    stem = base.split(".", 1)[0].rstrip(" .").upper()
    return stem not in _WINDOWS_RESERVED and not _WINDOWS_DEVICE.fullmatch(stem)


def _fail(action: str, err: Exception) -> keys.KeyringError:
    return keys.KeyringError(
        f"the keyring could not {action} proxy credentials ({type(err).__name__})"
    )


def _private_write(path: Path, content: str | bytes) -> None:
    private_directory(path.parent)
    if path.exists() or path.is_symlink():
        private_file(path)
    fd, name = tempfile.mkstemp(prefix=".credentials-", dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            private_file(temp)
            stream.write(content.encode("utf-8") if isinstance(content, str) else content)
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(path)
        private_file(path)
        if os.name != "nt":
            parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


def _read_file(path: Path) -> str:
    private_file(path)
    return path.read_text(encoding="utf-8")


def _read_bytes(path: Path) -> bytes:
    private_file(path)
    return path.read_bytes()


class CredentialStore:
    """Persist CLIProxyAPI auth JSON through the OS keyring; runtime copies are daemon-owned."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.auth_dir = self.root / "auths"
        self._cached_recovery_key: bytes | None = None

    @property
    def user(self) -> str:
        namespace = hashlib.sha256(str(self.root.expanduser().resolve()).encode()).hexdigest()
        return f"{_PREFIX}-{namespace}"

    @property
    def _marker(self) -> Path:
        return self.root / "credentials-runtime.json"

    @contextlib.contextmanager
    def _locked(self, *, wait: bool) -> Iterator[None]:
        private_directory(self.root)
        deadline = time.monotonic() + 5 if wait else 0
        while True:
            try:
                lock = lock_path(self.root / "credentials")
                with swap_lock(self.root / "credentials"):
                    if lock is not None and lock.exists():
                        private_file(lock)
                    yield
                return
            except FolderBusyError:
                if time.monotonic() >= deadline:
                    if wait:
                        raise keys.KeyringError("the credential store is busy") from None
                    raise
                time.sleep(0.05)

    @property
    def _recovery_journal(self) -> Path:
        return self.root / _RECOVERY_FILE

    def _recovery_key(self, ring: Any, *, create: bool) -> bytes:
        value = self._get(ring, _RECOVERY_KEY)
        if value is None and create:
            key = secrets.token_bytes(32)
            self._set(ring, _RECOVERY_KEY, base64.b64encode(key).decode("ascii"))
            value = self._get(ring, _RECOVERY_KEY)
        try:
            key = base64.b64decode(value, validate=True) if value is not None else b""
            if len(key) != 32:
                raise ValueError
        except ValueError:
            raise keys.KeyringError("the keyring has no valid proxy recovery key") from None
        self._cached_recovery_key = key
        return key

    def _seal_recovery(self, snapshot: dict[str, bytes], *, stored_authority: bool = False) -> None:
        key = self._cached_recovery_key
        if key is None:
            raise keys.KeyringError("the proxy recovery key is unavailable")
        files = {name: value for name, value in snapshot.items() if name != _STAGE_FILE}
        stage = snapshot.get(_STAGE_FILE)
        payload = json.dumps(
            {
                "version": 1,
                "authority": "stored" if stored_authority else "runtime",
                "files": {
                    name: base64.b64encode(data).decode("ascii") for name, data in files.items()
                },
                "stage": (base64.b64encode(stage).decode("ascii") if stage is not None else None),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        nonce = secrets.token_bytes(12)
        aad = _RECOVERY_MAGIC + self.user.encode("ascii")
        encrypted = AESGCM(key).encrypt(nonce, payload, aad)
        _private_write(self._recovery_journal, _RECOVERY_MAGIC + nonce + encrypted)

    def _open_recovery(self) -> tuple[dict[str, bytes], bool]:
        key = self._cached_recovery_key
        if key is None:
            raise keys.KeyringError("the proxy recovery key is unavailable")
        raw = _read_bytes(self._recovery_journal)
        try:
            if not raw.startswith(_RECOVERY_MAGIC) or len(raw) < len(_RECOVERY_MAGIC) + 28:
                raise ValueError
            offset = len(_RECOVERY_MAGIC)
            aad = _RECOVERY_MAGIC + self.user.encode("ascii")
            payload = AESGCM(key).decrypt(raw[offset : offset + 12], raw[offset + 12 :], aad)
            decoded = json.loads(payload)
            if (
                not isinstance(decoded, dict)
                or decoded.get("version") != 1
                or decoded.get("authority") not in ("runtime", "stored")
                or not isinstance(decoded.get("files"), dict)
            ):
                raise ValueError
            stored_authority = decoded["authority"] == "stored"
            snapshot: dict[str, bytes] = {}
            for name, encoded in decoded["files"].items():
                if not _safe_name(name) or not isinstance(encoded, str):
                    raise ValueError
                data = base64.b64decode(encoded, validate=True)
                if not stored_authority and not isinstance(json.loads(data.decode("utf-8")), dict):
                    raise ValueError
                snapshot[name] = data
            stage = decoded.get("stage")
            if stage is not None:
                if not isinstance(stage, str):
                    raise ValueError
                base64.b64decode(stage, validate=True)
            return snapshot, stored_authority
        except (InvalidTag, ValueError):
            raise keys.KeyringError("the encrypted proxy recovery journal is invalid") from None

    def _raw_owned_files(self, managed: set[str], marker: dict[str, Any]) -> dict[str, bytes]:
        if not self.auth_dir.exists() and not self.auth_dir.is_symlink():
            return {}
        private_directory(self.auth_dir)
        result: dict[str, bytes] = {}
        names = set(managed)
        if marker.get("stage"):
            names.add(_STAGE_FILE)
        for path in self.auth_dir.iterdir():
            if path.is_symlink() or path.is_dir():
                raise OSError("unsafe credential path")
            if path.name == _STAGE_FILE and marker.get("stage"):
                names.add(path.name)
            elif path.suffix.lower() == ".json":
                if not _safe_name(path.name) or not path.is_file():
                    raise OSError("unsafe credential path")
                names.add(path.name)
        for name in names:
            path = self.auth_dir / name
            if path.is_symlink():
                raise OSError("unsafe credential path")
            if path.exists():
                result[name] = _read_bytes(path)
        return result

    @staticmethod
    def _select_os_backend(module: Any) -> Any:
        backend = module.get_keyring()

        def supported(candidate: Any) -> bool:
            return (
                type(candidate).__module__,
                type(candidate).__name__,
            ) in {
                ("keyring.backends.SecretService", "Keyring"),
                ("keyring.backends.kwallet", "DBusKeyring"),
                ("keyring.backends.macOS", "Keyring"),
                ("keyring.backends.Windows", "WinVaultKeyring"),
            }

        if supported(backend):
            return backend
        if (
            type(backend).__module__ == "keyring.backends.chainer"
            and type(backend).__name__ == "ChainerBackend"
        ):
            selected = next((item for item in backend.backends if supported(item)), None)
            if selected is not None:
                return selected
        raise keys.KeyringError("no supported operating-system keyring is available")

    def _ring(self) -> Any:
        try:
            return self._select_os_backend(keys._keyring())
        except keys.KeyringError:
            raise
        except Exception as err:  # noqa: BLE001
            raise _fail("access", err) from None

    def _get(self, ring: Any, name: str) -> str | None:
        try:
            value = ring.get_password(keys.SERVICE, f"{self.user}-{name}")
            return value
        except Exception as err:  # noqa: BLE001
            raise _fail("read", err) from None

    def _set(self, ring: Any, name: str, value: str) -> None:
        try:
            ring.set_password(keys.SERVICE, f"{self.user}-{name}", value)
            if ring.get_password(keys.SERVICE, f"{self.user}-{name}") != value:
                raise RuntimeError("keyring readback mismatch")
        except Exception as err:  # noqa: BLE001
            raise _fail("write", err) from None

    def _delete(self, ring: Any, name: str) -> None:
        try:
            value = ring.get_password(keys.SERVICE, f"{self.user}-{name}")
            if value is not None:
                ring.delete_password(keys.SERVICE, f"{self.user}-{name}")
        except Exception as err:  # noqa: BLE001 - old immutable generation only
            logger.warning("could not remove old proxy credential data (%s)", type(err).__name__)

    def _load(self, ring: Any) -> dict[str, str] | None:
        raw = self._get(ring, "manifest")
        if raw is None:
            return None
        try:
            manifest = json.loads(raw)
            generation, count = manifest["generation"], manifest["count"]
            if (
                manifest["version"] != 1
                or not isinstance(generation, str)
                or not _GENERATION.fullmatch(generation)
                or type(count) is not int
                or not 1 <= count <= _MAX_CHUNKS
            ):
                raise ValueError
            parts = [self._get(ring, f"generation-{generation}-{index}") for index in range(count)]
            if any(value is None or len(value) > _CHUNK or not value.isascii() for value in parts):
                raise ValueError
            encoded = "".join(parts)
            payload = base64.b64decode(encoded, validate=True)
            digest = hashlib.sha256(payload).hexdigest()
            if digest != manifest["sha256"]:
                raise ValueError
            snapshot = json.loads(payload.decode("utf-8"))
            if not isinstance(snapshot, dict) or any(
                not _safe_name(name) or not isinstance(value, str)
                for name, value in snapshot.items()
            ):
                raise ValueError
            return snapshot
        except keys.KeyringError:
            raise
        except Exception as err:  # noqa: BLE001 - keyring contents are untrusted
            raise keys.KeyringError(
                f"the keyring contains invalid proxy credentials ({type(err).__name__})"
            ) from None

    def _commit(self, ring: Any, snapshot: dict[str, str]) -> None:
        payload = json.dumps(
            snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        encoded = base64.b64encode(payload).decode("ascii")
        digest = hashlib.sha256(payload).hexdigest()
        generation = secrets.token_hex(16)
        count = (len(encoded) + _CHUNK - 1) // _CHUNK
        if count > _MAX_CHUNKS:
            raise ValueError("proxy credential snapshot is too large")
        old_raw = self._get(ring, "manifest")
        for index in range(count):
            name = f"generation-{generation}-{index}"
            self._set(ring, name, encoded[index * _CHUNK : (index + 1) * _CHUNK])
        manifest = json.dumps(
            {"version": 1, "generation": generation, "count": count, "sha256": digest},
            separators=(",", ":"),
        )
        self._set(ring, "manifest", manifest)
        if old_raw:
            try:
                old = json.loads(old_raw)
                old_generation, old_count = old["generation"], old["count"]
                if (
                    isinstance(old_generation, str)
                    and _GENERATION.fullmatch(old_generation)
                    and type(old_count) is int
                    and 1 <= old_count <= _MAX_CHUNKS
                ):
                    for index in range(old_count):
                        if old_generation != generation:
                            self._delete(ring, f"generation-{old_generation}-{index}")
                else:
                    raise ValueError
            except Exception as err:  # noqa: BLE001 - manifest is already committed
                logger.warning("could not clean old proxy credential data (%s)", type(err).__name__)

    def _snapshot(self, *, final: bool) -> dict[str, str] | None:
        try:
            self.auth_dir.lstat()
        except FileNotFoundError:
            if final:
                raise OSError("credential directory is missing") from None
            return None
        private_directory(self.auth_dir)
        paths = list(self.auth_dir.iterdir())
        snapshot: dict[str, str] = {}
        for path in paths:
            try:
                mode = path.lstat().st_mode
            except FileNotFoundError:
                if final:
                    raise
                return None
            if stat.S_ISLNK(mode) or stat.S_ISDIR(mode):
                raise OSError("unsafe credential path")
            if path.suffix.lower() != ".json":
                continue
            if not _safe_name(path.name) or not stat.S_ISREG(mode):
                raise OSError("unsafe credential path")
            try:
                raw = _read_file(path)
                value = json.loads(raw)
                if not isinstance(value, dict):
                    raise ValueError("credential must be a JSON object")
            except FileNotFoundError:
                if final:
                    raise
                try:
                    path.lstat()
                except FileNotFoundError:
                    return None
                raise
            except (UnicodeError, ValueError):
                if final:
                    raise
                return None
            snapshot[path.name] = raw
        return snapshot

    def _marker_data(self) -> dict[str, Any] | None:
        if not self._marker.exists() and not self._marker.is_symlink():
            return None
        if self._marker.is_symlink():
            raise OSError("unsafe credentials marker")
        marker = json.loads(_read_file(self._marker))
        managed = marker.get("managed", [])
        stage = marker.get("stage", False)
        if (
            marker.get("version") != 1
            or marker.get("status") not in ("active", "stored", "sealed")
            or not isinstance(managed, list)
            or any(not _safe_name(name) for name in managed)
            or not isinstance(stage, bool)
            or (stage and marker.get("status") not in ("stored", "sealed"))
        ):
            raise ValueError("invalid credentials marker")
        return marker

    def _write_marker(self, status: str, managed: set[str], *, stage: bool = False) -> None:
        _private_write(
            self._marker,
            json.dumps(
                {"version": 1, "status": status, "managed": sorted(managed), "stage": stage}
            ),
        )

    def _recover_stage(self, marker: dict[str, Any] | None) -> None:
        if not self.auth_dir.exists() and not self.auth_dir.is_symlink():
            return
        private_directory(self.auth_dir)
        stage = self.auth_dir / _STAGE_FILE
        if stage.is_symlink():
            raise OSError("unsafe credential staging path")
        if stage.exists():
            if not marker or not marker.get("stage") or not stage.is_file():
                raise OSError("unowned credential staging file")
            private_file(stage)
            stage.unlink()

    def _cleanup(self, names: set[str], keep: set[str]) -> None:
        private_directory(self.auth_dir)
        for name in names - keep:
            path = self.auth_dir / name
            if path.is_symlink():
                raise OSError("unsafe credential path")
            if path.exists():
                private_file(path)
                path.unlink()

    def _materialize(self, snapshot: dict[str, str], managed: set[str]) -> None:
        private_directory(self.auth_dir)
        allowed = managed | set(snapshot)
        for path in self.auth_dir.iterdir():
            if path.is_symlink() or path.is_dir():
                raise OSError("unsafe credential path")
            if path.suffix.lower() != ".json":
                continue
            if not _safe_name(path.name) or path.name not in allowed:
                raise OSError("unowned credential file")
            private_file(path)
        stage = self.auth_dir / _STAGE_FILE
        for name, content in snapshot.items():
            if not _safe_name(name):
                raise OSError("unsafe credential name")
            path = self.auth_dir / name
            if path.exists():
                private_file(path)
            if stage.exists() or stage.is_symlink():
                raise OSError("unowned credential staging file")
            self._write_marker("stored", managed, stage=True)
            try:
                fd = os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            except FileExistsError:
                self._write_marker("stored", managed)
                raise
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                private_file(stage)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            stage.replace(path)
            private_file(path)
            self._write_marker("stored", managed)

    def prepare(self) -> None:
        with self._locked(wait=True):
            ring = self._ring()
            has_journal = self._recovery_journal.exists() or self._recovery_journal.is_symlink()
            self._recovery_key(ring, create=not has_journal)
            if has_journal:
                recovered, stored_authority = self._open_recovery()
                current_marker = self._marker_data()
                owned = self._raw_owned_files(
                    set(current_marker.get("managed", [])) if current_marker else set(),
                    current_marker or {},
                )
                if stored_authority:
                    committed = self._load(ring)
                    if committed is None:
                        raise keys.KeyringError("the keyring has no committed proxy credentials")
                else:
                    committed = {name: data.decode("utf-8") for name, data in recovered.items()}
                    self._commit(ring, committed)
                committed_names = set(committed)
                old_managed = set(current_marker.get("managed", [])) if current_marker else set()
                self._write_marker(
                    "stored",
                    old_managed | committed_names | {name for name in owned if _safe_name(name)},
                    stage=bool(current_marker and current_marker.get("stage")),
                )
                self._cleanup(set(owned) | old_managed, committed_names)
                if current_marker and current_marker.get("stage"):
                    self._recover_stage(current_marker)
                self._write_marker("stored", committed_names)
                self._recovery_journal.unlink()
            stored = self._load(ring)
            marker = self._marker_data()
            state = marker.get("status") if marker else None
            managed = set(marker.get("managed", [])) if marker else set()
            if state == "sealed":
                raise keys.KeyringError("sealed proxy credentials have no recovery journal")
            self._recover_stage(marker)
            if state is not None and stored is None:
                raise keys.KeyringError("the keyring has no committed proxy credentials")
            if state == "active":
                runtime = self._snapshot(final=True)
                assert runtime is not None
                if runtime != stored:
                    self._commit(ring, runtime)
                stored = runtime
                self._write_marker("stored", managed | set(runtime))
            elif state is None:
                private_directory(self.auth_dir)
                runtime = self._snapshot(final=True)
                merged = dict(stored or {})
                merged.update(runtime)
                if stored is None or merged != stored:
                    self._commit(ring, merged)
                    stored = merged
                managed.update(runtime)
                self._write_marker("stored", managed)
            managed.update(stored or {})
            self._cleanup(managed, set(stored or {}))
            self._materialize(stored or {}, managed)
            self._write_marker("active", set(stored or {}))

    def sync(self, *, final: bool = False) -> None:
        try:
            with self._locked(wait=final):
                marker = self._marker_data()
                if not marker or marker.get("status") != "active" or marker.get("stage"):
                    raise keys.KeyringError(
                        "proxy credentials are not active; refusing to overwrite durable data"
                    )
                ring = self._ring()
                current = self._snapshot(final=final)
                if current is None:
                    return
                stored = self._load(ring)
                if stored is None or current != stored:
                    self._commit(ring, current)
                    self._write_marker("active", set(marker.get("managed", [])) | set(current))
        except FolderBusyError:
            if final:
                raise keys.KeyringError(
                    "the credential store is busy; credentials were not saved"
                ) from None
            return

    def finish(self) -> None:
        with self._locked(wait=True):
            marker = self._marker_data()
            if not marker:
                if not self.auth_dir.exists():
                    return
                if self.auth_dir.is_symlink() or any(self.auth_dir.iterdir()):
                    raise keys.KeyringError("unowned proxy credential files were left untouched")
                return
            managed = set(marker.get("managed", []))
            if marker.get("status") == "sealed":
                self._open_recovery()
                self._recover_stage(marker)
                self._cleanup(managed, set())
                return
            if marker.get("status") == "stored":
                if marker.get("stage"):
                    raw = self._raw_owned_files(managed, marker)
                    self._seal_recovery(raw, stored_authority=True)
                    self._write_marker(
                        "sealed",
                        managed | {name for name in raw if _safe_name(name)},
                        stage=True,
                    )
                    self._cleanup(set(raw), set())
                else:
                    self._recover_stage(marker)
                    self._cleanup(managed, set())
                return
            if not self.auth_dir.exists():
                raise OSError("credential directory is missing")
            self._recover_stage(marker)
            raw = self._raw_owned_files(managed, marker)
            try:
                snapshot = self._snapshot(final=True)
                assert snapshot is not None
                ring = self._ring()
                stored = self._load(ring)
                if stored is None:
                    raise keys.KeyringError("the keyring has no committed proxy credentials")
                if snapshot != stored:
                    self._commit(ring, snapshot)
            except Exception as persist_error:
                try:
                    self._seal_recovery(raw)
                except Exception as recovery_error:  # noqa: BLE001 - retain plaintext
                    raise keys.KeyringError(
                        f"could not encrypt proxy credentials; private files remain in "
                        f"{self.auth_dir} ({type(recovery_error).__name__})"
                    ) from None
                self._write_marker(
                    "sealed",
                    managed | {name for name in raw if _safe_name(name)},
                    stage=_STAGE_FILE in raw,
                )
                self._cleanup(set(raw), set())
                raise persist_error
            managed.update(snapshot)
            self._write_marker("stored", managed)
            self._cleanup(managed, set())


def checkpoint(root: Path) -> None:
    """Persist a completed CLIProxyAPI login/logout without restoring or cleaning runtime files."""
    CredentialStore(root).sync(final=True)
