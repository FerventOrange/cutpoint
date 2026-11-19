"""``events.sheets.contact_sheet``: the JPEG grid of captioned frames behind ``events --sheets``
and the 360 QA sheets. The video grab is replaced by flat synthetic frames."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from cutpoint.errors import FFmpegError
from cutpoint.events import sheets

W, H = 96, 64


@pytest.fixture
def grabs(monkeypatch):
    """``grab`` returns a flat grey frame of the value in the file name (``v50.mp4`` -> 50), or
    None for ``bad.mp4``. Records ``(name, t, width)`` per call."""
    seen: list[tuple[str, float, int]] = []

    def grab(path, t, width=480):
        seen.append((Path(path).name, t, width))
        if Path(path).stem == "bad":
            return None
        return np.full((H, W, 3), int(Path(path).stem[1:]), np.uint8)

    monkeypatch.setattr(sheets, "grab", grab)
    return seen


def item(value: int | str, t: float = 0.0, **extra) -> dict:
    name = value if isinstance(value, str) else f"v{value}"
    return {"path": f"{name}.mp4", "t": t, "caption": "cap", **extra}


def decode(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    assert img is not None
    return img


def centre(sheet: np.ndarray, row: int, col: int) -> int:
    """Mean of the middle of tile (row, col), clear of the caption band."""
    return int(sheet[H * row + 20 : H * row + 30, W * col + 40 : W * col + 50].mean())


def test_tiles_are_laid_out_in_reading_order_and_the_last_row_is_padded_black(
    tmp_path: Path, grabs
) -> None:
    out = tmp_path / "sub" / "sheet.jpg"
    got = sheets.contact_sheet([item(v) for v in (20, 70, 120, 170, 220)], out, cols=3, width=W)
    assert got == out and out.is_file()  # the folder is created
    sheet = decode(out)
    assert sheet.shape == (2 * H, 3 * W, 3)
    cells = [centre(sheet, i // 3, i % 3) for i in range(6)]
    assert cells == pytest.approx([20, 70, 120, 170, 220, 0], abs=6)  # sixth: blank tile
    assert [c[2] for c in grabs] == [W] * 5  # the tile width is passed to the grab


def test_each_frame_is_grabbed_at_its_own_time(tmp_path: Path, grabs) -> None:
    sheets.contact_sheet([item(10, 1.5), item(20, 7.25)], tmp_path / "s.jpg", cols=2, width=W)
    assert [(n, t) for n, t, _ in grabs] == [("v10.mp4", 1.5), ("v20.mp4", 7.25)]


def test_a_frame_that_cannot_be_read_is_left_out_not_left_blank(tmp_path: Path, grabs) -> None:
    out = tmp_path / "s.jpg"
    sheets.contact_sheet([item(40), item("bad"), item(200)], out, cols=2, width=W)
    sheet = decode(out)
    assert sheet.shape == (H, 2 * W, 3)  # two tiles in one row, the bad one is not a gap
    assert [centre(sheet, 0, c) for c in (0, 1)] == pytest.approx([40, 200], abs=6)


def test_no_readable_frame_writes_nothing(tmp_path: Path, grabs) -> None:
    out = tmp_path / "out" / "s.jpg"
    assert sheets.contact_sheet([item("bad")], out) is None
    assert sheets.contact_sheet([], out) is None
    assert not out.exists() and not out.parent.exists()


def test_the_caption_sits_on_a_black_band_in_white(tmp_path: Path, grabs) -> None:
    out = tmp_path / "s.jpg"
    sheets.contact_sheet([item(150, caption="HELLO")], out, cols=1, width=W)
    sheet = decode(out)
    band = sheet[H - 22 :, :]
    assert (band.max(axis=2) > 200).any()  # white text
    assert np.median(band) < 40  # on black
    assert sheet[: H - 22].mean() == pytest.approx(150, abs=6)  # the picture is untouched above


@pytest.mark.parametrize(("kept", "channel", "others"), [(True, 1, (0, 2)), (False, 2, (0, 1))])
def test_a_box_is_green_when_kept_and_red_when_dropped(
    tmp_path: Path, grabs, kept, channel, others
) -> None:
    """Normalised box corners (0.25, 0.25)-(0.75, 0.6) of the tile; BGR channel order."""
    out = tmp_path / "s.jpg"
    box = (0.25, 0.25, 0.75, 0.6, kept)
    sheets.contact_sheet([item(100, boxes=[box])], out, cols=1, width=W)
    sheet = decode(out).astype(int)
    x, y = int(0.25 * W), int(0.4 * H)  # on the left edge of the box
    px = sheet[y - 1 : y + 2, x - 1 : x + 2].reshape(-1, 3).max(axis=0)
    assert px[channel] > 180
    assert all(px[o] < 80 for o in others)
    inside = sheet[int(0.4 * H), int(0.5 * W)]  # the box is an outline, not a fill
    assert abs(int(inside.mean()) - 100) < 12


def test_tiles_of_different_heights_are_padded_to_the_tallest(tmp_path: Path, monkeypatch) -> None:
    def grab(path, t, width=480):
        h = 40 if Path(path).stem == "short" else H
        return np.full((h, W, 3), 100, np.uint8)

    monkeypatch.setattr(sheets, "grab", grab)
    out = tmp_path / "s.jpg"
    sheets.contact_sheet([item("short"), item("tall")], out, cols=2, width=W)
    assert decode(out).shape == (H, 2 * W, 3)


# --------------------------------------------------------------------------- grab
def test_grab_decodes_the_png_frame_and_seeks_to_a_clamped_rounded_time(monkeypatch) -> None:
    seen = []
    _ok, png = cv2.imencode(".png", np.full((8, 12, 3), 90, np.uint8))

    def grab_images(path, t, width, fmt):
        seen.append((t, width, fmt))
        return [png.tobytes()]

    monkeypatch.setattr(sheets, "grab_images", grab_images)
    img = sheets.grab(Path("a.mp4"), 1.23456, 320)
    assert img.shape == (8, 12, 3) and int(img.mean()) == 90
    sheets.grab(Path("a.mp4"), -3.0)
    assert seen == [(1.23, 320, "png"), (0.0, 480, "png")]


def test_grab_gives_none_when_ffmpeg_fails(monkeypatch) -> None:
    def boom(*a, **k):
        raise FFmpegError("no frame")

    monkeypatch.setattr(sheets, "grab_images", boom)
    assert sheets.grab(Path("a.mp4"), 1.0) is None


# --------------------------------------------------------------------------- frames and title
def test_a_decoded_frame_replaces_the_grab_and_a_highlight_colours_the_band(
    tmp_path: Path, grabs
) -> None:
    out = tmp_path / "s.jpg"
    frame = np.full((H, W, 3), 120, np.uint8)
    hot = {"frame": frame, "caption": "hot", "highlight": (0, 0, 160)}
    cold = {"frame": frame.copy(), "caption": "cold"}
    sheets.contact_sheet([hot, cold], out, cols=2, width=999)
    sheet = decode(out)
    assert grabs == []  # nothing was grabbed
    assert sheet.shape == (H, 2 * W, 3)
    band = sheet[H - 6 : H - 1, :]
    left, right = band[:, W - 12 : W - 2].mean(axis=(0, 1)), band[:, -12:-2].mean(axis=(0, 1))
    assert left[2] > 120 and left[0] < 40  # red (BGR) band
    assert right.max() < 40  # black band


def test_a_title_adds_a_strip_above_the_grid_and_no_title_adds_none(tmp_path: Path, grabs) -> None:
    plain, titled, empty = (tmp_path / n for n in ("p.jpg", "t.jpg", "e.jpg"))
    sheets.contact_sheet([item(100)], plain, cols=1, width=W)
    sheets.contact_sheet([item(100)], titled, cols=1, width=W, title="HELLO " * 40)
    sheets.contact_sheet([item(100)], empty, cols=1, width=W, title="")
    assert decode(plain).shape == (H, W, 3)
    assert decode(titled).shape == (H + 26, W, 3)
    assert decode(empty).shape == (H + 26, W, 3)  # an empty title still has its strip
    strip = decode(titled)[:26]
    assert (strip.max(axis=2) > 200).any()  # white text on black
    assert np.median(strip) < 40
    assert decode(empty)[:26].max() < 40
