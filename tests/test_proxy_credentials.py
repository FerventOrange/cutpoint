"""Durable CLIProxyAPI credentials never rely on plaintext persistence."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cutpoint.assistant import keys, proxy_credentials
from cutpoint.assistant.proxy_credentials import CredentialStore, checkpoint


class FakeRing:
    def __init__(self) -> None:
        self.saved: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, user: str) -> str | None:
        return self.saved.get((service, user))

    def set_password(self, service: str, user: str, value: str) -> None:
        self.saved[(service, user)] = value

    def delete_password(self, service: str, user: str) -> None:
        del self.saved[(service, user)]


@pytest.fixture
def ring(monkeypatch: pytest.MonkeyPatch) -> FakeRing:
    fake = FakeRing()
    monkeypatch.setattr(keys, "_keyring", lambda: fake)
    monkeypatch.setattr(CredentialStore, "_select_os_backend", staticmethod(lambda module: module))
    return fake


def _auth(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _manifest(store: CredentialStore, ring: FakeRing) -> dict[str, Any]:
    return json.loads(ring.saved[(keys.SERVICE, f"{store.user}-manifest")])


def _stored_snapshot(store: CredentialStore, ring: FakeRing) -> dict[str, str]:
    manifest = _manifest(store, ring)
    encoded = "".join(
        ring.saved[(keys.SERVICE, f"{store.user}-generation-{manifest['generation']}-{index}")]
        for index in range(manifest["count"])
    )
    return json.loads(base64.b64decode(encoded))


def test_only_operating_system_backends_are_accepted_and_chainer_does_not_fallback(
    tmp_path: Path,
) -> None:
    writes: list[str] = []

    def locked_set(self, service: str, user: str, value: str) -> None:
        raise RuntimeError("locked")

    secure_type = type(
        "Keyring",
        (),
        {
            "__module__": "keyring.backends.SecretService",
            "get_password": lambda self, service, user: None,
            "set_password": locked_set,
        },
    )
    plaintext_type = type(
        "PlaintextKeyring",
        (),
        {
            "__module__": "keyring.backends.file",
            "get_password": lambda self, service, user: None,
            "set_password": lambda self, service, user, value: writes.append(value),
        },
    )
    secure, plaintext = secure_type(), plaintext_type()
    chainer_type = type(
        "ChainerBackend",
        (),
        {"__module__": "keyring.backends.chainer", "backends": [plaintext, secure]},
    )
    module = SimpleNamespace(get_keyring=lambda: chainer_type())
    store = CredentialStore(tmp_path / "proxy")
    assert store._select_os_backend(module) is secure
    with pytest.raises(keys.KeyringError):
        store._set(secure, "manifest", "secret")
    assert writes == []
    with pytest.raises(keys.KeyringError, match="operating-system keyring"):
        store._select_os_backend(SimpleNamespace(get_keyring=lambda: plaintext))


def test_prepare_fails_closed_without_a_working_backend(tmp_path: Path, monkeypatch) -> None:
    class NoRing:
        def get_password(self, *_: str) -> None:
            return None

        def set_password(self, *_: str) -> None:
            return None

    monkeypatch.setattr(keys, "_keyring", lambda: NoRing())
    root = tmp_path / "proxy"
    account = root / "auths" / "existing.json"
    _auth(account, {"token": "unchanged"})
    store = CredentialStore(root)
    with pytest.raises(keys.KeyringError):
        store.prepare()
    assert not (store.auth_dir / ".cutpoint-credential-stage").exists()
    assert json.loads(account.read_text(encoding="utf-8")) == {"token": "unchanged"}


def test_unicode_snapshot_is_chunked_and_recovered(tmp_path: Path, ring: FakeRing) -> None:
    root = tmp_path / "proxy"
    account = "codex-user+tag@example.com-用户.JSON"
    secret = "🔒" * 1100
    _auth(root / "auths" / account, {"refresh_token": secret})
    store = CredentialStore(root)
    store.prepare()
    store.sync(final=True)
    manifest = _manifest(store, ring)
    assert manifest["count"] > 1
    assert set(manifest) == {"version", "generation", "count", "sha256"}
    assert len(ring.saved[(keys.SERVICE, f"{store.user}-manifest")]) < 256
    assert all(value.isascii() and len(value) <= 1000 for value in ring.saved.values())
    assert _stored_snapshot(store, ring) == {
        account: (root / "auths" / account).read_text(encoding="utf-8")
    }
    store.finish()
    CredentialStore(root).prepare()
    assert (
        json.loads((root / "auths" / account).read_text(encoding="utf-8"))["refresh_token"]
        == secret
    )


def test_restart_materializes_stored_accounts_before_sync(tmp_path: Path, ring: FakeRing) -> None:
    root = tmp_path / "proxy"
    account = "codex-user@example.com.json"
    store = CredentialStore(root)
    _auth(store.auth_dir / account, {"refresh_token": "durable"})
    store.prepare()
    store.finish()
    assert not (store.auth_dir / account).exists()

    restarted = CredentialStore(root)
    restarted.prepare()
    assert json.loads((restarted.auth_dir / account).read_text())["refresh_token"] == "durable"
    restarted.sync(final=True)
    assert json.loads(_stored_snapshot(restarted, ring)[account])["refresh_token"] == "durable"


def test_active_recovery_preserves_rotation_and_deletion(tmp_path: Path, ring: FakeRing) -> None:
    root = tmp_path / "proxy"
    store = CredentialStore(root)
    rotated = "codex-user@example.com-rotation.json"
    deleted = "codex-user@example.com-deleted.json"
    _auth(root / "auths" / rotated, {"token": "old"})
    _auth(root / "auths" / deleted, {"token": "gone"})
    store.prepare()
    _auth(root / "auths" / rotated, {"token": "new"})
    (root / "auths" / deleted).unlink()

    # Simulate termination before the periodic sync; active runtime state, including deletion, wins.
    CredentialStore(root).prepare()
    assert json.loads((root / "auths" / rotated).read_text())["token"] == "new"
    store.finish()
    durable = _stored_snapshot(store, ring)
    assert json.loads(durable[rotated])["token"] == "new"
    assert deleted not in durable


def test_checkpoint_commits_runtime_without_cleanup(tmp_path: Path, ring: FakeRing) -> None:
    root = tmp_path / "proxy"
    CredentialStore(root).prepare()
    auth = root / "auths" / "login.json"
    _auth(auth, {"refresh_token": "value"})
    checkpoint(root)
    assert auth.exists()
    assert CredentialStore(root)._load(CredentialStore(root)._ring()) == {
        "login.json": auth.read_text()
    }


@pytest.mark.parametrize("partial", [b"{", b'{"token":"\xe2'])
def test_partial_runtime_write_defers_whole_snapshot_and_final_rejects(
    tmp_path: Path, ring: FakeRing, partial: bytes
) -> None:
    """Incomplete JSON or UTF-8 must defer the whole periodic snapshot, not stop supervision."""
    root = tmp_path / "proxy"
    store = CredentialStore(root)
    _auth(root / "auths" / "account.json", {"token": "complete"})
    store.prepare()
    _auth(root / "auths" / "account.json", {"token": "incomplete-new"})
    partial_path = root / "auths" / "partial.json"
    partial_path.write_bytes(partial)
    store.sync()
    assert store._load(store._ring()) == {"account.json": json.dumps({"token": "complete"})}
    with pytest.raises(ValueError):
        store.sync(final=True)
    assert store._load(store._ring()) == {"account.json": json.dumps({"token": "complete"})}
    completed = json.dumps({"token": "complete-€"}, ensure_ascii=False)
    partial_path.write_text(completed, encoding="utf-8")
    store.sync()
    assert store._load(store._ring()) == {
        "account.json": json.dumps({"token": "incomplete-new"}),
        "partial.json": completed,
    }


def test_disappearing_credential_defers_nonfinal_snapshot_until_coherent_scan(
    tmp_path: Path, ring: FakeRing, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = CredentialStore(tmp_path / "proxy")
    first = store.auth_dir / "a.json"
    disappearing = store.auth_dir / "b.json"
    _auth(first, {"token": "old"})
    _auth(disappearing, {"token": "present"})
    store.prepare()
    committed = _stored_snapshot(store, ring)
    _auth(first, {"token": "rotated"})

    private_file = proxy_credentials.private_file

    def remove_during_private_check(path: Path) -> None:
        private_file(path)
        if path == disappearing:
            path.unlink()

    monkeypatch.setattr(proxy_credentials, "private_file", remove_during_private_check)
    store.sync()
    assert _stored_snapshot(store, ring) == committed

    monkeypatch.setattr(proxy_credentials, "private_file", private_file)
    store.sync()
    assert _stored_snapshot(store, ring) == {"a.json": json.dumps({"token": "rotated"})}


def test_corrupt_manifest_chunk_fails_closed(tmp_path: Path, ring: FakeRing) -> None:
    store = CredentialStore(tmp_path / "proxy")
    _auth(store.auth_dir / "account.json", {"token": "secret-value"})
    store.prepare()
    manifest = _manifest(store, ring)
    chunk_key = (
        keys.SERVICE,
        f"{store.user}-generation-{manifest['generation']}-0",
    )
    ring.saved[chunk_key] = "corrupt"
    with pytest.raises(keys.KeyringError):
        store.sync(final=True)


def test_manifest_rejects_unbounded_chunk_counts(tmp_path: Path, ring: FakeRing) -> None:
    store = CredentialStore(tmp_path / "proxy")
    _auth(store.auth_dir / "account.json", {"token": "secret-value"})
    store.prepare()
    key = (keys.SERVICE, f"{store.user}-manifest")
    original = ring.saved[key]
    for count in (0, 100_001, True):
        manifest = json.loads(original)
        manifest["count"] = count
        ring.saved[key] = json.dumps(manifest)
        with pytest.raises(keys.KeyringError):
            store.sync(final=True)


def test_symlink_auth_file_is_never_followed_or_deleted(tmp_path: Path, ring: FakeRing) -> None:
    root = tmp_path / "proxy"
    outside = tmp_path / "outside.json"
    outside.write_text('{"token":"outside"}', encoding="utf-8")
    store = CredentialStore(root)
    store.prepare()
    link = store.auth_dir / "credential.json"
    try:
        link.symlink_to(outside)
    except OSError as err:
        pytest.skip(f"symlinks unavailable ({type(err).__name__})")
    with pytest.raises(OSError):
        store.sync(final=True)
    assert outside.read_text(encoding="utf-8") == '{"token":"outside"}'
    assert link.is_symlink()


def test_nested_auth_directory_fails_closed(tmp_path: Path, ring: FakeRing) -> None:
    store = CredentialStore(tmp_path / "proxy")
    store.prepare()
    nested = store.auth_dir / "nested"
    nested.mkdir()

    with pytest.raises(OSError):
        store.sync(final=True)
    assert nested.exists()


def test_nonfinal_sync_propagates_private_file_failure(
    tmp_path: Path, ring: FakeRing, monkeypatch
) -> None:
    store = CredentialStore(tmp_path / "proxy")
    account = store.auth_dir / "account.json"
    _auth(account, {"token": "value"})
    store.prepare()
    private_file = proxy_credentials.private_file

    def reject_account(path: Path) -> None:
        if path == account:
            raise PermissionError("private file access denied")
        private_file(path)

    monkeypatch.setattr(proxy_credentials, "private_file", reject_account)
    with pytest.raises(PermissionError, match="private file access denied"):
        store.sync()


def test_missing_marker_imports_runtime_over_the_stored_account(
    tmp_path: Path, ring: FakeRing
) -> None:
    root = tmp_path / "proxy"
    store = CredentialStore(root)
    store.prepare()
    auth = root / "auths" / "account.json"
    _auth(auth, {"token": "old"})
    checkpoint(root)
    (root / "credentials-runtime.json").unlink()
    _auth(auth, {"token": "new"})

    CredentialStore(root).prepare()
    durable = CredentialStore(root)._load(ring)
    assert durable is not None
    assert json.loads(durable["account.json"])["token"] == "new"


def test_stored_cleanup_removes_only_marked_owned_files(tmp_path: Path, ring: FakeRing) -> None:
    root = tmp_path / "proxy"
    store = CredentialStore(root)
    _auth(store.auth_dir / "account.json", {"token": "owned"})
    store.prepare()
    (root / "credentials-runtime.json").write_text(
        json.dumps({"version": 1, "status": "stored", "managed": ["account.json"]}),
        encoding="utf-8",
    )
    (store.auth_dir / "account.json").write_text('{"token":"stale"}', encoding="utf-8")
    foreign = store.auth_dir / "notes.txt"
    foreign.write_text("keep", encoding="utf-8")

    store.prepare()
    assert json.loads((store.auth_dir / "account.json").read_text())["token"] == "owned"
    store.finish()
    assert not (store.auth_dir / "account.json").exists()
    assert foreign.read_text(encoding="utf-8") == "keep"


def test_failed_staging_keeps_previous_commit_and_sanitizes_backend_error(
    tmp_path: Path, ring: FakeRing, monkeypatch
) -> None:
    root = tmp_path / "proxy"
    store = CredentialStore(root)
    _auth(store.auth_dir / "account.json", {"token": "safe-old"})
    store.prepare()
    manifest_key = (keys.SERVICE, f"{store.user}-manifest")
    old_manifest = ring.saved[manifest_key]
    _auth(store.auth_dir / "account.json", {"token": "private-new-value"})

    class FailingRing:
        def get_password(self, service: str, user: str) -> str | None:
            return ring.get_password(service, user)

        def set_password(self, service: str, user: str, value: str) -> None:
            raise RuntimeError("private-new-value")

    monkeypatch.setattr(keys, "_keyring", lambda: FailingRing())
    with pytest.raises(keys.KeyringError) as error:
        store.sync(final=True)
    assert "private-new-value" not in str(error.value)
    assert ring.saved[manifest_key] == old_manifest
    assert json.loads((store.auth_dir / "account.json").read_text())["token"] == "private-new-value"


def test_nonfinal_sync_rejects_symlink_without_following_it(tmp_path: Path, ring: FakeRing) -> None:
    root = tmp_path / "proxy"
    store = CredentialStore(root)
    store.prepare()
    outside = tmp_path / "outside.json"
    outside.write_text('{"token":"outside"}', encoding="utf-8")
    link = store.auth_dir / "account.json"
    try:
        link.symlink_to(outside)
    except OSError as err:
        pytest.skip(f"symlinks unavailable ({type(err).__name__})")
    with pytest.raises(OSError):
        store.sync()
    assert outside.read_text(encoding="utf-8") == '{"token":"outside"}'


def test_symlink_auth_directory_is_never_cleaned(tmp_path: Path, ring: FakeRing) -> None:
    root = tmp_path / "proxy"
    store = CredentialStore(root)
    store.prepare()
    external = tmp_path / "outside"
    external.mkdir()
    auth = external / "owned.json"
    auth.write_text('{"token":"outside"}', encoding="utf-8")
    store.auth_dir.rmdir()
    try:
        store.auth_dir.symlink_to(external, target_is_directory=True)
    except OSError as err:
        pytest.skip(f"symlinks unavailable ({type(err).__name__})")
    (root / "credentials-runtime.json").write_text(
        json.dumps({"version": 1, "status": "stored", "managed": ["owned.json"]}),
        encoding="utf-8",
    )

    with pytest.raises(OSError):
        store.prepare()
    assert auth.read_text(encoding="utf-8") == '{"token":"outside"}'


def test_old_chunk_cleanup_failure_warns_without_secret(
    tmp_path: Path, ring: FakeRing, monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    store = CredentialStore(tmp_path / "proxy")
    auth = store.auth_dir / "account.json"
    _auth(auth, {"token": "old"})
    store.prepare()
    _auth(auth, {"token": "new"})

    class FailingDeleteRing:
        def get_password(self, service: str, user: str) -> str | None:
            return ring.get_password(service, user)

        def set_password(self, service: str, user: str, value: str) -> None:
            ring.set_password(service, user, value)

        def delete_password(self, service: str, user: str) -> None:
            raise RuntimeError("private-cleanup-secret")

    monkeypatch.setattr(keys, "_keyring", lambda: FailingDeleteRing())
    store.sync(final=True)
    assert "RuntimeError" in caplog.text
    assert "private-cleanup-secret" not in caplog.text
    assert json.loads(_stored_snapshot(store, ring)["account.json"])["token"] == "new"


def test_unsafe_windows_names_and_ads_are_rejected(tmp_path: Path, ring: FakeRing) -> None:
    store = CredentialStore(tmp_path / "proxy")
    store.prepare()
    invalid = ("CON.json", "credential:token.json", r"nested\credential.json")
    created: list[Path] = []
    for name in invalid:
        path = store.auth_dir / name
        try:
            path.write_text('{"token":"unsafe"}', encoding="utf-8")
        except (OSError, ValueError):
            continue
        created.append(path)
    if not created:
        pytest.skip("host filesystem prevents creating Windows-special filenames")
    with pytest.raises(OSError):
        store.sync(final=True)
    assert all(path.read_text(encoding="utf-8") == '{"token":"unsafe"}' for path in created)


def test_checkpoint_after_finish_cannot_delete_durable_accounts(
    tmp_path: Path, ring: FakeRing
) -> None:
    root = tmp_path / "proxy"
    store = CredentialStore(root)
    _auth(store.auth_dir / "codex-user@example.com.json", {"refresh_token": "durable"})
    store.prepare()
    store.finish()

    store.finish()

    with pytest.raises(keys.KeyringError):
        checkpoint(root)
    assert _stored_snapshot(store, ring) == {
        "codex-user@example.com.json": json.dumps({"refresh_token": "durable"})
    }


@pytest.mark.parametrize("interruption", ["created", "written"])
def test_interrupted_materialization_recovers_owned_stage_only(
    tmp_path: Path, ring: FakeRing, monkeypatch, interruption: str
) -> None:
    root = tmp_path / "proxy"
    store = CredentialStore(root)
    account = "codex-user@example.com.json"
    _auth(store.auth_dir / account, {"refresh_token": "staged-secret"})
    foreign = store.auth_dir / "notes.txt"
    foreign.write_text("leave alone", encoding="utf-8")
    store.prepare()
    store.finish()

    stage = store.auth_dir / ".cutpoint-credential-stage"
    open_file, replace = os.open, os.replace

    def interrupt_after_create(path, *args, **kwargs):
        fd = open_file(path, *args, **kwargs)
        if Path(path) == stage:
            os.close(fd)
            raise RuntimeError("simulated interruption")
        return fd

    def interrupt_before_replace(source, target):
        if Path(source) == stage:
            raise RuntimeError("simulated interruption")
        replace(source, target)

    with monkeypatch.context() as patcher:
        if interruption == "created":
            patcher.setattr(os, "open", interrupt_after_create)
        else:
            patcher.setattr(os, "replace", interrupt_before_replace)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            CredentialStore(root).prepare()
        expected = "" if interruption == "created" else _stored_snapshot(store, ring)[account]
        assert stage.read_text(encoding="utf-8") == expected

    CredentialStore(root).prepare()
    assert not stage.exists()
    assert (
        json.loads((root / "auths" / account).read_text(encoding="utf-8"))["refresh_token"]
        == "staged-secret"
    )
    CredentialStore(root).finish()
    assert foreign.read_text(encoding="utf-8") == "leave alone"


@pytest.mark.parametrize("arrival", ["existing", "concurrent"])
def test_unowned_staging_file_is_never_removed(
    tmp_path: Path, ring: FakeRing, monkeypatch, arrival: str
) -> None:
    root = tmp_path / "proxy"
    store = CredentialStore(root)
    _auth(store.auth_dir / "codex-user@example.com.json", {"refresh_token": "durable"})
    store.prepare()
    store.finish()

    stage = store.auth_dir / ".cutpoint-credential-stage"
    open_file = os.open

    def competing_create(path, *args, **kwargs):
        if Path(path) == stage:
            stage.write_text("unowned", encoding="utf-8")
        return open_file(path, *args, **kwargs)

    with monkeypatch.context() as patcher:
        if arrival == "existing":
            stage.write_text("unowned", encoding="utf-8")
        else:
            patcher.setattr(os, "open", competing_create)
        with pytest.raises(OSError):
            CredentialStore(root).prepare()
    assert stage.read_text(encoding="utf-8") == "unowned"
    with pytest.raises(OSError):
        CredentialStore(root).prepare()
    assert stage.read_text(encoding="utf-8") == "unowned"


def test_finish_preserves_refresh_and_deletion_in_encrypted_journal(
    tmp_path: Path, ring: FakeRing, monkeypatch
) -> None:
    root = tmp_path / "proxy"
    account = "codex-user@example.com.json"
    deleted = "claude-user@example.com.json"
    store = CredentialStore(root)
    _auth(store.auth_dir / account, {"refresh_token": "old"})
    _auth(store.auth_dir / deleted, {"refresh_token": "remove"})
    store.prepare()
    old_manifest = _manifest(store, ring)
    _auth(store.auth_dir / account, {"refresh_token": "rotated-value"})
    (store.auth_dir / deleted).unlink()

    class LockedRing:
        def get_password(self, service: str, user: str) -> str | None:
            return ring.get_password(service, user)

        def set_password(self, service: str, user: str, value: str) -> None:
            raise RuntimeError("rotated-value")

        def delete_password(self, service: str, user: str) -> None:
            ring.delete_password(service, user)

    monkeypatch.setattr(keys, "_keyring", lambda: LockedRing())

    with pytest.raises(keys.KeyringError) as error:
        store.finish()
    assert "rotated-value" not in str(error.value)
    journal = store._recovery_journal.read_bytes()
    assert b"rotated-value" not in journal
    assert not (store.auth_dir / account).exists()
    assert not (store.auth_dir / deleted).exists()
    assert _manifest(store, ring) == old_manifest

    monkeypatch.setattr(keys, "_keyring", lambda: ring)
    restored = CredentialStore(root)
    restored.prepare()
    assert json.loads((store.auth_dir / account).read_text())["refresh_token"] == "rotated-value"
    assert not (store.auth_dir / deleted).exists()
    durable = _stored_snapshot(restored, ring)
    assert json.loads(durable[account])["refresh_token"] == "rotated-value"
    assert deleted not in durable
    assert not restored._recovery_journal.exists()


def test_corrupt_recovery_journal_never_overwrites_committed_snapshot(
    tmp_path: Path, ring: FakeRing, monkeypatch
) -> None:
    root = tmp_path / "proxy"
    store = CredentialStore(root)
    account = "account.json"
    _auth(store.auth_dir / account, {"token": "old"})
    store.prepare()
    old_manifest = _manifest(store, ring)
    _auth(store.auth_dir / account, {"token": "new"})

    class LockedRing:
        def get_password(self, service: str, user: str) -> str | None:
            return ring.get_password(service, user)

        def set_password(self, service: str, user: str, value: str) -> None:
            raise RuntimeError("locked")

    monkeypatch.setattr(keys, "_keyring", lambda: LockedRing())
    with pytest.raises(keys.KeyringError):
        store.finish()
    ciphertext = bytearray(store._recovery_journal.read_bytes())
    ciphertext[-1] ^= 1
    store._recovery_journal.write_bytes(ciphertext)

    monkeypatch.setattr(keys, "_keyring", lambda: ring)
    with pytest.raises(keys.KeyringError, match="recovery journal"):
        CredentialStore(root).prepare()
    assert _manifest(store, ring) == old_manifest
    assert store._recovery_journal.exists()


def test_journal_write_failure_keeps_last_plaintext_copy(
    tmp_path: Path, ring: FakeRing, monkeypatch
) -> None:
    root = tmp_path / "proxy"
    store = CredentialStore(root)
    account = store.auth_dir / "account.json"
    _auth(account, {"token": "old"})
    store.prepare()
    _auth(account, {"token": "latest-refresh"})

    class LockedRing:
        def get_password(self, service: str, user: str) -> str | None:
            return ring.get_password(service, user)

        def set_password(self, service: str, user: str, value: str) -> None:
            raise RuntimeError("locked")

    monkeypatch.setattr(keys, "_keyring", lambda: LockedRing())
    private_write = proxy_credentials._private_write

    def fail_journal(path: Path, content: str | bytes) -> None:
        if path.name == "credential-recovery.enc":
            raise OSError("disk full")
        private_write(path, content)

    monkeypatch.setattr(proxy_credentials, "_private_write", fail_journal)
    with pytest.raises(keys.KeyringError, match="private files remain"):
        store.finish()
    assert json.loads(account.read_text(encoding="utf-8"))["token"] == "latest-refresh"
    assert not store._recovery_journal.exists()


def test_invalid_partial_runtime_is_encrypted_but_never_restored_as_old_snapshot(
    tmp_path: Path, ring: FakeRing
) -> None:
    root = tmp_path / "proxy"
    store = CredentialStore(root)
    account = store.auth_dir / "account.json"
    _auth(account, {"token": "old"})
    store.prepare()
    old_manifest = _manifest(store, ring)
    account.write_text('{"token":"partial-secret', encoding="utf-8")

    with pytest.raises(ValueError):
        store.finish()
    ciphertext = store._recovery_journal.read_bytes()
    assert b"partial-secret" not in ciphertext
    assert not account.exists()
    with pytest.raises(keys.KeyringError, match="recovery journal"):
        CredentialStore(root).prepare()
    assert _manifest(store, ring) == old_manifest


def test_locked_recovery_key_keeps_journal_and_old_snapshot(
    tmp_path: Path, ring: FakeRing, monkeypatch
) -> None:
    root = tmp_path / "proxy"
    store = CredentialStore(root)
    account = store.auth_dir / "account.json"
    _auth(account, {"token": "old"})
    store.prepare()
    old_manifest = _manifest(store, ring)
    _auth(account, {"token": "latest"})

    class LockedRing:
        def get_password(self, service: str, user: str) -> str | None:
            raise RuntimeError("keyring locked")

        def set_password(self, service: str, user: str, value: str) -> None:
            raise RuntimeError("keyring locked")

    monkeypatch.setattr(keys, "_keyring", lambda: LockedRing())
    with pytest.raises(keys.KeyringError):
        store.finish()
    assert not account.exists()
    assert store._recovery_journal.exists()
    monkeypatch.setattr(keys, "_keyring", lambda: LockedRing())
    with pytest.raises(keys.KeyringError):
        CredentialStore(root).prepare()
    assert _manifest(store, ring) == old_manifest
    assert store._recovery_journal.exists()


def test_finish_seals_owned_stage_after_prepare_interruption(
    tmp_path: Path, ring: FakeRing, monkeypatch
) -> None:
    root = tmp_path / "proxy"
    account = "account.json"
    store = CredentialStore(root)
    _auth(store.auth_dir / account, {"token": "durable"})
    store.prepare()
    store.finish()
    stage = store.auth_dir / ".cutpoint-credential-stage"
    open_file = os.open

    def interrupt_after_create(path, *args, **kwargs):
        fd = open_file(path, *args, **kwargs)
        if Path(path) == stage:
            os.close(fd)
            raise RuntimeError("simulated interruption")
        return fd

    with monkeypatch.context() as patcher:
        patcher.setattr(os, "open", interrupt_after_create)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            store.prepare()
    store.finish()
    assert not stage.exists()
    assert store._recovery_journal.exists()
    assert b"durable" not in store._recovery_journal.read_bytes()

    CredentialStore(root).prepare()
    assert json.loads((store.auth_dir / account).read_text())["token"] == "durable"


def test_missing_active_directory_cannot_be_snapshotted_as_empty(
    tmp_path: Path, ring: FakeRing
) -> None:
    root = tmp_path / "proxy"
    store = CredentialStore(root)
    account = store.auth_dir / "account.json"
    _auth(account, {"token": "durable"})
    store.prepare()
    old_manifest = _manifest(store, ring)
    account.unlink()
    store.auth_dir.rmdir()

    with pytest.raises(OSError, match="directory is missing"):
        CredentialStore(root).prepare()
    assert _manifest(store, ring) == old_manifest
