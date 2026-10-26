"""refmatch ``mask_friends`` search: soft and hard mask only (friends no longer lift it)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cutpoint import refmatch, ski
from cutpoint.styles import get_style


def test_mask_modes_collapse_old_threshold_lists() -> None:
    assert refmatch.mask_modes((0, 5, 6, 7, 8, 11), 7.0) == (0.0, 7.0)
    assert refmatch.mask_modes((0, 7, 11), 7.0) == (0.0, 7.0)
    assert refmatch.mask_modes((11,), 7.0) == (7.0,)  # cross-validation refits one mode
    assert refmatch.mask_modes((5, 0), 0.0) == (float(ski.FRIENDS_STRONG), 0.0)


def test_fit_weights_fits_each_mask_once(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def stop(_ctx, _keys, thresholds, *_a, **_k):
        seen.append(thresholds)
        raise RuntimeError("stop")

    monkeypatch.setattr(refmatch, "_train", stop)
    with pytest.raises(RuntimeError, match="stop"):
        refmatch.fit_weights([], thresholds=(0, 5, 6, 7, 8, 11), save=False)
    assert seen == [(0.0, get_style("ski").mask_friends)]


def test_old_preset_mask_friends_is_still_read_as_a_hard_mask(home: Path) -> None:
    styles = home / "styles"
    styles.mkdir(parents=True)
    preset = {"name": "old", "base": "ski", "weights": {"speed": 1.0}, "mask_friends": 5}
    (styles / "old.json").write_text(json.dumps(preset), encoding="utf-8")
    style = get_style("old")
    assert style.mask_friends == 5.0
    mask, friends = np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 1.0])
    assert ski.effective_mask(mask, friends, style.mask_friends).tolist() == [1.0, 1.0, 0.0]
    assert ski.effective_mask(mask, friends, 0).tolist() == [0.0, 0.0, 0.0]  # soft


def test_log_fit_reads_a_preset_saved_with_a_bom(tmp_path: Path, caplog) -> None:
    preset = tmp_path / "learned.json"
    preset.write_bytes(b"\xef\xbb\xbf" + json.dumps({"weights": {"speed": 2.0}}).encode())
    rep = {"preset": str(preset), "features_fitted": ["speed"], "features": {}}
    with caplog.at_level("INFO", logger="cutpoint.refmatch"), pytest.raises(KeyError):
        refmatch._log_fit(rep, {"speed": 1.0})  # stops at the first missing report key
    assert "speed      1.00 -> 2.00" in caplog.text
