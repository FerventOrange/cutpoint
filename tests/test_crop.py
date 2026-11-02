"""Black-border detection (``crop.detect``) and the per-source crop table of project.json.

``cropdetect_at`` (ffmpeg) is replaced by a list of boxes, one per sampled time."""

from __future__ import annotations

from pathlib import Path

import pytest

from conftest import media_info
from cutpoint import crop
from cutpoint.crop import Crop

LETTERBOX = (1920, 800, 0, 140)  # w, h, x, y


@pytest.fixture
def boxes(monkeypatch):
    """``set(list)``: what cropdetect finds at each sample (None = nothing); returns the
    sampled times."""
    sampled: list[float] = []
    found: list = []

    def at(src, t):
        sampled.append(t)
        return found[len(sampled) - 1]

    monkeypatch.setattr(crop, "cropdetect_at", at)

    def set_(values):
        found[:] = values
        return sampled

    return set_


@pytest.fixture
def src(home, tmp_path) -> Path:
    path = tmp_path / "movie.mp4"
    path.write_bytes(b"x")
    return path


def test_the_box_most_samples_agree_on_is_the_crop(boxes, src):
    boxes([LETTERBOX, LETTERBOX, LETTERBOX, (1920, 600, 0, 240), None, LETTERBOX])
    assert crop.detect(src, media_info(src)) == Crop(*LETTERBOX)


def test_without_a_majority_the_union_is_taken_so_a_dark_scene_cannot_shrink_it(boxes, src):
    boxes([(1920, 800, 0, 140), (1920, 600, 0, 240), (1600, 800, 160, 140), None, None, None])
    assert crop.detect(src, media_info(src)) == Crop(1920, 800, 0, 140)


@pytest.mark.parametrize(
    "found",
    [[(1910, 1076, 4, 2)] * 6, [None] * 6],
    ids=["almost-full-frame", "nothing-found"],
)
def test_no_border_means_no_crop(boxes, src, found):
    boxes(found)
    assert crop.detect(src, media_info(src)) is None


def test_detection_is_cached_per_source(boxes, src):
    sampled = boxes([LETTERBOX] * 6)
    first = crop.detect(src, media_info(src))
    sampled.clear()
    assert crop.detect(src, media_info(src)) == first
    assert sampled == []


def test_a_damaged_source_is_sampled_only_before_its_read_error(boxes, src):
    sampled = boxes([LETTERBOX] * 6)
    crop.detect(src, media_info(src, duration=100.0), readable_until=50.0)
    assert sampled == pytest.approx([f * 50.0 for f in crop.SAMPLE_FRACTIONS])


@pytest.mark.parametrize(
    ("path", "spec"),
    [
        (Path("D:/rec/Stream-01.MP4"), "1920:800:0:140"),  # name pattern, any case
        (Path("D:/cams/side/a.mp4"), "auto"),  # full-path pattern
        (Path("D:/rec/other.mp4"), None),
    ],
)
def test_project_crop_matches_the_file_name_or_path(path, spec):
    cfg = {"crop": {"stream*.mp4": "1920:800:0:140", "*/cams/side/*": "auto"}}
    assert crop.project_crop(cfg, path) == spec
    assert crop.project_crop(None, path) is None
