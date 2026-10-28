"""Color handling of exports (#289, #290): probed tags, HDR/BT.2020 to SDR Rec.709, BT.601 to
BT.709, log warning, the automatic grade (``color.py``, #290 stage 2)."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from conftest import media_info, needs_ffmpeg, run_json, write
from cutpoint import assemble, cache, color, encode, extend, ffmpeg_utils
from cutpoint.boundaries import SpeechSettings
from cutpoint.color import AutoGrade, ColorStats, Metrics
from cutpoint.crop import Crop
from cutpoint.ffmpeg_utils import FFmpegError, VideoEncoder
from cutpoint.planning import Padding

BOTH = frozenset({"zscale", "tonemap", "scale"})
HLG = {"color_transfer": "arib-std-b67", "color_primaries": "bt2020", "color_space": "bt2020nc"}
PQ = {"color_transfer": "smpte2084", "color_primaries": "bt2020", "color_space": "bt2020nc"}
BT709 = {"color_transfer": "bt709", "color_primaries": "bt709", "color_space": "bt709"}
FULL_BT709 = BT709 | {"color_range": "pc", "pix_fmt": "yuvj420p"}  # GoPro
BT601 = {"color_transfer": "smpte170m", "color_primaries": "smpte170m", "color_space": "smpte170m"}
SD = {"width": 720, "height": 480}
TODAY = (
    "scale=1920:1080:force_original_aspect_ratio=decrease:out_range=tv,"
    "pad=1920:1080:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30,format=yuv420p"
)


# -- probe ---------------------------------------------------------------------------------------


def test_probe_reads_the_color_tags_and_treats_unknown_or_missing_as_none(monkeypatch):
    video = {
        "codec_type": "video", "width": 3840, "height": 2160, "avg_frame_rate": "30/1",
        "pix_fmt": "yuv420p10le", "codec_name": "hevc", "color_transfer": "arib-std-b67",
        "color_primaries": "bt2020", "color_space": "unknown",
    }  # fmt: skip
    data = {"streams": [video], "format": {"duration": "5.0"}}
    monkeypatch.setattr(ffmpeg_utils, "ffprobe_json", lambda *a, **k: data)

    info = ffmpeg_utils.probe(Path("hlg.mp4"))

    assert (info.color_transfer, info.color_primaries) == ("arib-std-b67", "bt2020")
    assert info.color_space is None and info.color_range is None


def test_filter_names_reads_ffmpeg_filters_output():
    listing = (
        "Filters:\n  T.. = Timeline support\n  ------\n"
        " TS aap               AA->A      Apply Affine Projection.\n"
        " .S zscale            V->V       Apply resizing, colorspace and bit depth conversion.\n"
        " .. buffer            |->V       Buffer video frames.\n"
    )
    assert ffmpeg_utils.filter_names(listing) == {"aap", "zscale", "buffer"}


# -- the conversion chain ------------------------------------------------------------------------


def test_pq_source_is_tone_mapped_to_bt709_with_hable():
    vf = encode.sdr_filter(media_info(pix_fmt="yuv420p10le", **PQ), BOTH)
    assert vf == (  # unchanged by #449, so PQ proxies and stills keep their names
        "zscale=tin=smpte2084:pin=bt2020:min=bt2020nc:t=linear:npl=100:p=bt709,"
        "format=gbrpf32le,tonemap=tonemap=hable:desat=0,"
        "zscale=tin=linear:pin=bt709:t=bt709:p=bt709:m=bt709:r=tv"
    )


def test_hlg_source_is_tone_mapped_with_mobius_at_diffuse_white():
    """#449: hable at npl=100 was 16 % darker than the bt.2390 reference on real HLG."""
    vf = encode.sdr_filter(media_info(pix_fmt="yuv420p10le", **HLG), BOTH)
    assert vf == (
        "zscale=tin=arib-std-b67:pin=bt2020:min=bt2020nc:t=linear:npl=203:p=bt709,"
        "format=gbrpf32le,tonemap=tonemap=mobius:param=0.5:peak=4.926:desat=0,"
        "zscale=tin=linear:pin=bt709:t=bt709:p=bt709:m=bt709:r=tv"
    )


@pytest.mark.parametrize("tags", [HLG, PQ])
def test_hdr_preview_and_still_use_the_export_tone_map(tags):
    info = media_info(pix_fmt="yuv420p10le", **tags)
    sdr = encode.sdr_filter(info, BOTH)
    assert encode.preview_filter(info, BOTH) == sdr
    assert encode.still_filter(info, BOTH).startswith(f"{sdr},colorspace=all=bt470bg:")


def test_hdr_source_without_primaries_or_matrix_tags_is_read_as_bt2020():
    vf = encode.sdr_filter(media_info(color_transfer="smpte2084"), BOTH)
    assert vf.startswith("zscale=tin=smpte2084:pin=bt2020:min=bt2020nc:")


def test_bt2020_sdr_source_only_changes_primaries_and_matrix():
    info = media_info(color_transfer="bt2020-10", color_primaries="bt2020", color_space="bt2020nc")
    assert encode.sdr_filter(info, BOTH) == (
        "zscale=tin=bt2020-10:pin=bt2020:min=bt2020nc:t=bt709:p=bt709:m=bt709:r=tv"
    )


@pytest.mark.parametrize(
    "tags",
    [
        {},
        BT709,
        {"color_transfer": "smpte170m", "color_primaries": "smpte170m"},
        {"color_range": "pc"},
    ],
)
def test_sdr_and_untagged_sources_get_no_conversion(tags):
    assert encode.color_kind(media_info(**tags)) is None
    assert encode.sdr_filter(media_info(**tags), BOTH) == ""


# -- UI stills (#412) ----------------------------------------------------------------------------


TO_JPEG_709 = "colorspace=all=bt470bg:iall=bt709:range=pc:fast=1"


@pytest.mark.parametrize("tags", [HLG, PQ])
def test_hdr_still_is_tone_mapped_then_made_bt601_after_the_scale(tags):
    info = media_info(pix_fmt="yuv420p10le", **tags)
    color = encode.still_filter(info, BOTH)
    assert color == f"{encode.sdr_filter(info, BOTH)},{TO_JPEG_709}"
    args = ffmpeg_utils.grab_images_args("v.mp4", 5.0, width=160, quality=5, color=color)
    assert args[args.index("-vf") + 1] == f"scale=160:-2,{color}"


@pytest.mark.parametrize(
    "tags",
    [
        {},  # HD without a matrix tag: BT.709, as exports read it
        BT709,
        FULL_BT709,  # the colorspace filter reads the frames' range
        {**SD, **BT709},
        {"color_space": "bt709", "color_primaries": "smpte170m"},  # the matrix decides
    ],
)
def test_bt709_still_is_converted_to_the_bt601_full_range_of_a_jpeg(tags):
    """#441: the mjpeg encoder only re-tags, and every decoder reads a JPEG as BT.601."""
    assert encode.still_filter(media_info(**tags), BOTH) == TO_JPEG_709


def test_smpte240m_still_is_converted_to_bt601():
    info = media_info(color_space="smpte240m", **SD)
    assert encode.still_filter(info, BOTH) == (
        "colorspace=all=bt470bg:iall=smpte240m:range=pc:fast=1"
    )


@pytest.mark.parametrize("tags", [{"color_space": "gbr"}, {"color_space": "fcc"}])
def test_other_matrices_keep_todays_still(tags):
    assert encode.still_filter(media_info(**tags), BOTH) == ""


@pytest.mark.parametrize(
    "tags",
    [
        SD,
        {**SD, **BT601},
        BT601,  # tagged HD
        BT601 | {"color_range": "pc", "pix_fmt": "yuvj420p"},
        {"color_space": "bt470bg", "color_primaries": "bt470bg"},
    ],
)
def test_bt601_still_command_is_unchanged(tags):
    color = encode.still_filter(media_info(**tags), BOTH)
    args = ffmpeg_utils.grab_images_args("v.mp4", 5.0, width=160, quality=5, color=color)
    assert args == [
        "-hide_banner", "-v", "error", "-ss", "5.000", "-i", "v.mp4", "-an", "-sn",
        "-vf", "scale=160:-2", "-frames:v", "1", "-c:v", "mjpeg", "-q:v", "5",
        "-f", "image2pipe", "-",
    ]  # fmt: skip


def test_hdr_still_without_zscale_keeps_todays_command_and_warns(caplog):
    info = media_info(path=Path("hdr-still.mp4"), pix_fmt="yuv420p10le", **PQ)
    color = encode.still_filter(info, frozenset({"scale"}))
    assert color == ""
    assert "no zscale or tonemap filter" in caplog.text
    args = ffmpeg_utils.grab_images_args("v.mp4", 5.0, width=160, color=color)
    assert args[args.index("-vf") + 1] == "scale=160:-2"


# -- UI proxies (#441) --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tags",
    [
        BT601 | {"color_range": "pc", "pix_fmt": "yuvj420p"},
        {**SD, "color_range": "pc"},  # untagged SD is BT.601
        {"color_space": "bt470bg", "color_range": "pc"},
    ],
)
def test_full_range_bt601_proxy_is_only_narrowed_to_tv_range(tags):
    assert encode.TV_RANGE == "scale=out_range=tv"
    assert encode.preview_filter(media_info(**tags), BOTH) == encode.TV_RANGE


@pytest.mark.parametrize(
    "tags", [BT601, SD, BT709, FULL_BT709, {}, {"color_range": "pc"}, HLG, PQ, {**SD, **HLG}]
)
def test_other_proxies_keep_the_sdr_filter(tags):
    """Same filter string as before #441, so their proxies keep their names."""
    info = media_info(**tags)
    assert encode.preview_filter(info, BOTH) == encode.sdr_filter(info, BOTH)


# -- BT.601 to BT.709 (#290) -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tags", "matrix"),
    [
        (BT601, "smpte170m"),
        (BT601 | {"color_range": "pc", "pix_fmt": "yuvj420p"}, "smpte170m"),
        ({"color_space": "bt470bg", "color_primaries": "bt470bg"}, "bt470bg"),
        ({"color_space": "smpte240m"}, "smpte240m"),
        (SD, "smpte170m"),  # untagged SD
        ({"width": 960, "height": 540, "color_range": "tv"}, "smpte170m"),
        ({**SD, **BT601}, "smpte170m"),
        # No matrix tag: SD only, the primaries tag naming the matrix; the transfer never.
        ({**SD, "color_primaries": "bt470bg", "color_range": "tv"}, "bt470bg"),
        ({**SD, "color_primaries": "bt709"}, "smpte170m"),
        ({**SD, "color_transfer": "bt709"}, "smpte170m"),
    ],
)
def test_bt601_source_is_converted_to_bt709_by_matrix(tags, matrix):
    vf = encode.bt709_filter(media_info(**tags), BOTH)
    assert vf == f"colorspace=all=bt709:iall={matrix}:irange=tv:range=tv:fast=1"


@pytest.mark.parametrize(
    "tags",
    [
        BT709,
        FULL_BT709,
        {**SD, **BT709},  # tagged SD keeps its tags
        {"color_space": "bt709", "color_primaries": "smpte170m"},  # the matrix decides
        {"color_space": "gbr"},  # RGB stays as before
    ],
)
def test_bt709_sources_keep_todays_chain(tags):
    assert encode.bt601_matrix(media_info(**tags)) is None
    assert encode.bt709_filter(media_info(**tags), BOTH) == ""


@pytest.mark.parametrize(
    "tags",
    [
        {},
        {"color_range": "tv"},
        {"color_range": "pc"},
        {"width": 1280, "height": 720},
        {"width": 1280, "height": 536},  # cropped widescreen HD: HD by its width
        {"width": 1920, "height": 800},
        {"width": 720, "height": 1280},  # vertical HD
        # No matrix tag on HD: BT.601 primaries or transfer do not make it BT.601.
        {"color_transfer": "smpte170m"},
        {"color_primaries": "smpte170m", "color_transfer": "smpte170m"},
        {"color_primaries": "bt470bg", "color_range": "tv"},
        {"color_primaries": "bt709"},
    ],
)
def test_hd_without_a_matrix_tag_is_only_tagged_bt709(tags):
    assert encode.bt601_matrix(media_info(**tags)) is None
    assert encode.bt709_filter(media_info(**tags), BOTH) == encode.TAG_BT709
    assert encode.TAG_BT709 == "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709"


@pytest.mark.parametrize(
    "tags", [HLG, {"color_transfer": "bt2020-10", "color_primaries": "bt2020"}]
)
def test_hdr_and_bt2020_sources_keep_the_sdr_conversion(tags):
    info = media_info(**tags, **SD)  # SD as well: never read as BT.601
    assert encode.bt601_matrix(info) is None
    assert encode.bt709_filter(info, BOTH) == encode.sdr_filter(info, BOTH)
    assert encode.bt709_filter(info, frozenset()) == ""  # no zscale: as before, with its WARNING


def test_bt601_conversion_logs_one_info_line_per_source(caplog):
    caplog.set_level("INFO")
    for _ in range(3):
        encode.bt709_filter(media_info(Path("dvr.mp4"), **BT601), BOTH)
    encode.bt709_filter(media_info(Path("sd.mp4"), **SD), BOTH)
    pal = {"width": 720, "height": 576, "color_primaries": "bt470bg"}
    encode.bt709_filter(media_info(Path("pal.mp4"), **pal), BOTH)
    encode.bt709_filter(media_info(Path("hd.mp4"), color_transfer="smpte170m"), BOTH)
    lines = [r.getMessage() for r in caplog.records if r.levelname == "INFO"]
    assert lines == [
        "dvr.mp4 is BT.601 video (tagged smpte170m): converting to BT.709",
        "sd.mp4 is BT.601 video (no matrix tag, 720x480): converting to BT.709",
        "pal.mp4 is BT.601 video (no matrix tag, 720x576): converting to BT.709",
    ]


def test_without_zscale_hdr_keeps_todays_chain_and_warns_once_per_source(caplog):
    caplog.set_level("WARNING")
    info = media_info(Path("hdr.mp4"), **HLG)
    for _ in range(3):
        assert encode.sdr_filter(info, frozenset({"tonemap"})) == ""
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1
    assert "hdr.mp4 is HLG video" in warnings[0] and "no zscale filter" in warnings[0]


def test_video_filter_puts_the_conversion_just_before_the_8_bit_format():
    fmt = encode.OutputFormat(1920, 1080, 30)
    assert fmt.video_filter(False) == TODAY
    assert fmt.video_filter(False, color="X") == TODAY.replace("format=yuv420p", "X,format=yuv420p")
    vertical = fmt.video_filter(True, Crop(0, 0, 1000, 1000), "X")
    assert vertical.endswith("setsar=1,fps=30,X,format=yuv420p")


# -- the export command line ---------------------------------------------------------------------


def _export_args(monkeypatch, info, kind: str = "cpu") -> list[str]:
    calls: list[list[str]] = []
    monkeypatch.setattr(encode, "run", lambda args, **_kw: calls.append(args) or "")
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: BOTH)
    fmt = encode.OutputFormat(1920, 1080, 30)
    encode._encode(VideoEncoder(kind, "/dev/dri/renderD128" if kind == "vaapi" else None),
                   info.path, info, 1.0, 5.0, Path("o.mp4"), fmt, False, None, 0.0)  # fmt: skip
    return calls[0]


@pytest.mark.parametrize(("tags", "curve"), [(HLG, "mobius:param=0.5:peak=4.926"), (PQ, "hable")])
def test_hdr_export_tone_maps_and_tags_the_output_bt709(monkeypatch, tags, curve):
    args = _export_args(monkeypatch, media_info(pix_fmt="yuv420p10le", **tags))
    graph = args[args.index("-filter_complex") + 1]
    assert f",tonemap=tonemap={curve}:desat=0," in graph
    assert graph.index("tonemap") < graph.index("format=yuv420p")
    i = args.index("-color_range")
    assert args[i : i + 8] == ["-color_range", "tv", *encode.BT709_TAGS]


def test_hdr_export_with_vaapi_converts_before_the_upload(monkeypatch):
    args = _export_args(monkeypatch, media_info(**HLG), "vaapi")
    graph = args[args.index("-filter_complex") + 1]
    assert "r=tv,format=yuv420p,format=nv12,hwupload[v]" in graph


@pytest.mark.parametrize("tags", [BT709, FULL_BT709, {"color_space": "bt709"}, {**SD, **BT709}])
def test_bt709_export_command_is_unchanged(monkeypatch, tags):
    args = _export_args(monkeypatch, media_info(**tags))
    assert args[args.index("-filter_complex") + 1].startswith(f"[0:v:0]{TODAY}[v];")
    assert not {"-color_primaries", "-color_trc", "-colorspace"} & set(args)
    assert args[args.index("-color_range") : args.index("-color_range") + 3] == [
        "-color_range", "tv", "-fps_mode",
    ]  # fmt: skip


@pytest.mark.parametrize(("tags", "matrix"), [(BT601, "smpte170m"), (SD, "smpte170m")])
def test_bt601_export_converts_and_tags_the_output_bt709(monkeypatch, tags, matrix):
    args = _export_args(monkeypatch, media_info(**tags))
    graph = args[args.index("-filter_complex") + 1]
    conv = f"colorspace=all=bt709:iall={matrix}:irange=tv:range=tv:fast=1"
    assert graph.startswith(f"[0:v:0]{TODAY.replace('format=', conv + ',format=')}[v];")
    i = args.index("-color_range")
    assert args[i : i + 8] == ["-color_range", "tv", *encode.BT709_TAGS]


def test_untagged_hd_export_is_only_tagged_bt709(monkeypatch):
    args = _export_args(monkeypatch, media_info())
    graph = args[args.index("-filter_complex") + 1]
    assert graph.startswith(f"[0:v:0]{TODAY.replace('format=', encode.TAG_BT709 + ',format=')}[v];")
    i = args.index("-color_range")
    assert args[i : i + 8] == ["-color_range", "tv", *encode.BT709_TAGS]


# -- log/flat footage ----------------------------------------------------------------------------


def _frame(lo: float, hi: float, sat: float) -> np.ndarray:
    """36x64 RGB: a luma ramp from ``lo`` to ``hi`` (0-1), tinted red by ``sat``."""
    v = np.linspace(lo, hi, 64 * 36, dtype=np.float32).reshape(36, 64)
    rgb = np.stack([v, v * (1 - sat), v * (1 - sat)], axis=-1)
    return (rgb * 255).round().astype(np.uint8)


def test_log_stats_measure_saturation_and_the_luma_range():
    sat, black, white = encode.log_stats(_frame(0.2, 0.7, 0.1))
    assert sat == pytest.approx(0.1, abs=0.01)
    assert black == pytest.approx(0.2, abs=0.02) and white == pytest.approx(0.65, abs=0.05)


@pytest.mark.parametrize(
    ("stats", "flat"),
    [
        ((0.08, 0.15, 0.65), True),  # lifted blacks, no white, grey: log
        ((0.40, 0.15, 0.65), False),  # saturated
        ((0.08, 0.00, 0.65), False),  # real blacks
        ((0.08, 0.15, 0.95), False),  # real whites
        ((0.00, 0.30, 0.32), False),  # uniform frame (fade, grey card)
        ((0.00, 0.00, 0.00), False),  # black frame
    ],
)
def test_looks_log_needs_low_saturation_and_compressed_luma(stats, flat):
    assert encode.looks_log(*stats) is flat


@pytest.fixture
def grabs(monkeypatch):
    """``encode.decode_raw`` yields one flat frame; returns the paths it was asked for."""
    asked: list[Path] = []

    def fake(path, **_kw):
        asked.append(path)
        yield 0.0, _frame(0.15, 0.7, 0.05)

    monkeypatch.setattr(encode, "decode_raw", fake)
    return asked


def test_flat_10_bit_source_warns_once_and_suggests_a_lut(grabs, caplog):
    caplog.set_level("WARNING")
    src = Path("GX010001.MP4")
    for at in (1.0, 20.0):
        encode.warn_if_log(src, media_info(src, pix_fmt="yuv420p10le", **BT709), at)
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert grabs == [src] and len(warnings) == 1
    assert "GX010001.MP4 looks like log/flat footage" in warnings[0] and "--lut FILE" in warnings[0]


def test_export_clip_checks_the_source_for_log_footage(grabs, caplog, monkeypatch, tmp_path):
    def fake_run(args, **_kw):
        Path(args[-1]).write_bytes(b"x")  # the clip ffmpeg would write
        return ""

    monkeypatch.setattr(encode, "run", fake_run)
    monkeypatch.setattr(encode, "ffmpeg_filters", lambda: BOTH)
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    caplog.set_level("WARNING")
    src = Path("GX010002.MP4")
    info = media_info(src, pix_fmt="yuv420p10le", **BT709)

    encode.export_clip(
        src, info, 1.0, 2.0, tmp_path / "o.mp4", encode.OutputFormat(1920, 1080, 30), False
    )

    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert grabs == [src]
    assert any("GX010002.MP4 looks like log/flat footage" in w for w in warnings)


@pytest.mark.parametrize(
    "info",
    [media_info(pix_fmt="yuv420p"), media_info(pix_fmt="yuvj420p"),
     media_info(pix_fmt="yuv420p10le", **HLG)],
)  # fmt: skip
def test_8_bit_and_hdr_sources_are_not_checked_for_log(grabs, info):
    encode.warn_if_log(info.path, info, 0.0)
    assert grabs == []


def test_failed_log_grab_skips_the_check(monkeypatch, caplog):
    def broken(path, **_kw):
        raise FFmpegError("no frames")
        yield  # a generator, like decode_raw

    monkeypatch.setattr(encode, "decode_raw", broken)
    caplog.set_level("WARNING")
    encode.warn_if_log(Path("a.mp4"), media_info(pix_fmt="p010le"), 0.0)
    assert not caplog.records


# -- real ffmpeg ---------------------------------------------------------------------------------


def _hlg_source(path: Path) -> None:
    """1 s, 320x180 colour bars as tagged 10-bit HLG / BT.2020 (FFV1, so any build has it)."""
    to_hlg = (
        "format=yuv444p,zscale=tin=bt709:pin=bt709:min=bt709:rin=pc:t=linear:npl=203:p=bt709,"
        "format=gbrpf32le,zscale=tin=linear:pin=bt709:t=linear:p=bt2020,"
        "zscale=tin=linear:pin=bt2020:p=bt2020:t=arib-std-b67:m=bt2020nc:r=tv:npl=203,"
        "format=yuv420p10le"
    )
    ffmpeg_utils.run(
        [
            *("ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "smptebars=s=320x180:d=1:r=30"),
            *("-vf", to_hlg, "-c:v", "ffv1"),
            *("-color_primaries", "bt2020", "-color_trc", "arib-std-b67"),
            *("-colorspace", "bt2020nc", str(path)),
        ]
    )


def _saturation(path: Path) -> float:
    frames = ffmpeg_utils.decode_raw(
        path, size=(64, 36), vf="scale=64:36,format=rgb24", start=0.5, hwaccel=None
    )
    return encode.log_stats(next(iter(frames))[1])[0]


@needs_ffmpeg
def test_real_hlg_export_is_tone_mapped_and_tagged_bt709(tmp_path, monkeypatch):
    if not {"zscale", "tonemap"} <= ffmpeg_utils.ffmpeg_filters():
        pytest.skip("this ffmpeg has no zscale/tonemap")
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    src = tmp_path / "hlg.mkv"
    _hlg_source(src)
    info = ffmpeg_utils.probe(src)
    assert encode.color_kind(info) == "HLG"
    out = tmp_path / "001.mp4"

    encode.export_clip(src, info, 0.0, 1.0, out, encode.OutputFormat(320, 180, 30), False)

    got = ffmpeg_utils.probe(out)
    assert (got.pix_fmt, got.color_transfer, got.color_primaries, got.color_space) == (
        "yuv420p", "bt709", "bt709", "bt709",
    )  # fmt: skip
    # Read as SDR, the HLG source looks washed out; the converted clip keeps the bars' colour.
    assert _saturation(out) > _saturation(src) + 0.1


def _bars(path: Path, size: str, tag: str | None, data: str = "bt601", dur: float = 1.0) -> None:
    """``dur`` s of still colour bars at ``size`` (FFV1, tv range), ``data`` the matrix of
    the pixels, all three tags ``tag``.

    With ``tag`` None the clip is untagged (ffmpeg 9 would tag the scale's output bt470bg).
    """
    tags = ["-colorspace", tag, "-color_primaries", tag, "-color_trc", tag] if tag else []
    ffmpeg_utils.run(
        [
            *("ffmpeg", "-y", "-v", "error", "-f", "lavfi"),
            *("-i", f"smptebars=s={size}:d={dur:g}:r=30,format=rgb24"),
            *("-vf", f"scale=out_color_matrix={data}:out_range=tv,format=yuv420p,"
                     f"setparams=colorspace={tag or 'unknown'}"),
            *("-c:v", "ffv1", "-color_range", "tv", *tags, str(path)),
        ]
    )  # fmt: skip


def _rgb(path: Path, matrix: str, at: float = 0.5) -> np.ndarray:
    """The frame of ``path`` at ``at`` s as float RGB, decoded with ``matrix`` (tv range)."""
    vf = (
        "scale=320:180:flags=accurate_rnd+full_chroma_int:"
        f"in_color_matrix={matrix}:in_range=tv:out_range=pc,format=rgb24"
    )
    frames = ffmpeg_utils.decode_raw(path, size=(320, 180), vf=vf, start=at, hwaccel=None)
    return next(iter(frames))[1].astype(np.float32)


def _video_md5(path: Path) -> list[str]:
    """framemd5 of the video frames of ``path``."""
    text = ffmpeg_utils.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
                             "-f", "framemd5", "-"])  # fmt: skip
    return [line.rsplit(",", 1)[-1].strip() for line in text.splitlines() if line[:1].isdigit()]


@needs_ffmpeg
def test_real_bt601_export_is_bt709_data_and_tags(tmp_path, monkeypatch):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    src = tmp_path / "dvr.mkv"
    _bars(src, "320x180", "smpte170m")
    info = ffmpeg_utils.probe(src)
    assert encode.bt601_matrix(info) == "smpte170m"
    out = tmp_path / "001.mp4"

    encode.export_clip(src, info, 0.0, 1.0, out, encode.OutputFormat(320, 180, 30), False)

    got = ffmpeg_utils.probe(out)
    assert (got.color_space, got.color_primaries, got.color_transfer) == ("bt709",) * 3
    # Read as BT.709 (what a player does with the montage's tags), the clip shows the
    # source's colours; the source read that way is several codes off.
    want = _rgb(src, "bt601")
    assert np.abs(_rgb(out, "bt709") - want).mean() < 1.5
    assert np.abs(_rgb(src, "bt709") - want).mean() > 4.0


def _jpeg_rgb(image: bytes, tmp_path: Path) -> np.ndarray:
    """A JPEG as float RGB, read as every decoder does: BT.601, full range."""
    path = tmp_path / "still.jpg"
    path.write_bytes(image)
    vf = (
        "scale=320:180:flags=accurate_rnd+full_chroma_int:"
        "in_color_matrix=bt601:in_range=pc:out_range=pc,format=rgb24"
    )
    frames = ffmpeg_utils.decode_raw(path, size=(320, 180), vf=vf, hwaccel=None)
    return next(iter(frames))[1].astype(np.float32)


def _bar_means(rgb: np.ndarray) -> np.ndarray:
    """Mean RGB inside each of the seven top colour bars of a 320x180 smptebars frame."""
    w = rgb.shape[1] / 7
    return np.array([rgb[20:100, int(i * w + w / 4) : int(i * w + 3 * w / 4)].mean((0, 1))
                     for i in range(7)])  # fmt: skip


@needs_ffmpeg
@pytest.mark.parametrize(("data", "tag"), [("bt709", "bt709"), ("bt709", None), ("bt601", None)])
def test_real_still_shows_the_colours_of_the_export(tmp_path, monkeypatch, data, tag):
    """#441: before it, a BT.709 still was about 7 codes off (32 max) on bars."""
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    src = tmp_path / "src.mkv"
    _bars(src, "1280x720" if data == "bt709" else "320x240", tag, data)
    info = ffmpeg_utils.probe(src)
    out = tmp_path / "001.mp4"
    fmt = encode.OutputFormat(info.width, info.height, 30)
    encode.export_clip(src, info, 0.0, 1.0, out, fmt, False)
    color = encode.still_filter(info)
    assert bool(color) == (data == "bt709")
    [image] = ffmpeg_utils.grab_images(src, 0.5, width=320, hwaccel=None, color=color)
    got, want = _bar_means(_jpeg_rgb(image, tmp_path)), _bar_means(_rgb(out, "bt709"))
    assert np.abs(got - want).mean() < 1.5
    if color:  # the still of before
        [old] = ffmpeg_utils.grab_images(src, 0.5, width=320, hwaccel=None)
        assert np.abs(_bar_means(_jpeg_rgb(old, tmp_path)) - want).mean() > 4.0


@needs_ffmpeg
def test_real_untagged_hd_export_changes_no_pixel(tmp_path, monkeypatch):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    src = tmp_path / "hd.mkv"
    _bars(src, "1280x720", None)
    info = ffmpeg_utils.probe(src)
    assert (info.color_space, info.color_primaries, info.color_transfer) == (None,) * 3
    fmt = encode.OutputFormat(1280, 720, 30)
    tagged, before = tmp_path / "tagged.mp4", tmp_path / "before.mp4"

    encode.export_clip(src, info, 0.0, 1.0, tagged, fmt, False)
    monkeypatch.setattr(encode, "bt709_filter", lambda _info: "")  # the export before #290
    encode.export_clip(src, info, 0.0, 1.0, before, fmt, False)

    got = ffmpeg_utils.probe(tagged)
    assert (got.color_space, got.color_primaries, got.color_transfer) == ("bt709",) * 3
    assert _video_md5(tagged) == _video_md5(before)


# -- assemble re-encodes (#290) ------------------------------------------------------------------


def test_assemble_normalises_each_input_to_bt709():
    infos = [media_info(Path(f"00{i}.mp4"), 2.0, **t) for i, t in enumerate((BT709, {}, BT601))]
    chains, _ = assemble._normalize_inputs(infos)
    video = [c for c in chains if c.startswith("[") and ":v:0]" in c]
    assert f"fps=30,{encode.TAG_BT709},format=yuv420p," in video[0]  # BT.709: tags only
    assert f"fps=30,{encode.TAG_BT709},format=yuv420p," in video[1]
    conv = "colorspace=all=bt709:iall=smpte170m:irange=tv:range=tv:fast=1"
    assert f"fps=30,{conv},format=yuv420p," in video[2]
    other = assemble._normalize_inputs([media_info(color_space="gbr")])[0][0]
    assert "fps=30,format=yuv420p," in other  # RGB and other matrices: as before


def test_assemble_reencode_tags_the_output_bt709(monkeypatch, tmp_path):
    calls: list[list[str]] = []
    monkeypatch.setattr(assemble, "run", lambda args, **_kw: calls.append(args) or "")
    files = [tmp_path / "a.mp4"]
    assemble._encode_with(VideoEncoder("cpu"), files, "[0:v]null[v]", tmp_path / "o.mp4")
    args = calls[0]
    i = args.index("-color_range")
    assert args[i : i + 8] == ["-color_range", "tv", *encode.BT709_TAGS]


@needs_ffmpeg
def test_real_crossfade_of_bt709_untagged_and_bt601_clips_keeps_their_colours(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    files = [tmp_path / f"00{i}.mkv" for i in (1, 2, 3)]
    _bars(files[0], "1280x720", "bt709", "bt709", 2.0)  # a clip exported after #290
    _bars(files[1], "1280x720", None, "bt709", 2.0)  # an untagged HD clip from before
    _bars(files[2], "1280x720", "smpte170m", "bt601", 2.0)  # a BT.601 clip from before
    infos = [ffmpeg_utils.probe(f) for f in files]
    out = tmp_path / "final.mp4"

    assemble.crossfade(files, infos, out, assemble.crossfade_duration(infos, 0.2))

    got = ffmpeg_utils.probe(out)
    assert (got.color_space, got.color_primaries, got.color_transfer) == ("bt709",) * 3
    # The middle of each clip in the final, read as BT.709, against its source read as it
    # should be (still bars: any frame will do).
    for f, at, matrix in zip(files, (1.0, 2.8, 4.6), ("bt709", "bt709", "bt601"), strict=True):
        assert np.abs(_rgb(out, "bt709", at) - _rgb(f, matrix)).mean() < 1.5, f.name


# -- automatic grade (#290 stage 2) --------------------------------------------------------------


def _scene(n: int = 4, seed: int = 0) -> np.ndarray:
    """``n`` 72x128 float RGB frames: 40 % a gray ramp (neutral mid-tones), the rest coloured
    patches of every hue and saturation, a little near-black and near-white."""
    rng = np.random.default_rng(seed)
    frames = np.empty((n, 72, 128, 3), np.float32)
    for f in frames:
        ramp = np.linspace(0.02, 0.98, 128, dtype=np.float32)
        f[:29] = ramp[None, :, None]
        hue = rng.uniform(0, 1, (43, 128, 1)).astype(np.float32)
        sat = rng.uniform(0.2, 0.8, (43, 128, 1)).astype(np.float32)
        val = rng.uniform(0.25, 0.95, (43, 128, 1)).astype(np.float32)
        k = (np.array([5, 3, 1], np.float32) + hue * 6) % 6
        f[29:] = val - val * sat * np.clip(np.minimum(k, 4 - k), 0, 1)
    return frames


def _stats(frames: np.ndarray) -> ColorStats:
    return ColorStats(np.round(np.clip(frames, 0, 1) * 255).astype(np.uint8))


def _flat(frames: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return lo + frames * (hi - lo)


def _cast(frames: np.ndarray, r: float, g: float, b: float) -> np.ndarray:
    return np.clip(frames * np.array([r, g, b], np.float32), 0, 1)


@pytest.fixture
def lenient(monkeypatch):
    """A clipping veto of 5 points instead of 0.5, for tests of what the grade does: on these
    synthetic frames a cast's saturated colours and the underwater tint push a channel past
    the white point, 2 to 3 points more than the levels intend."""
    monkeypatch.setattr(color, "VETO_CLIPPED", 0.05)


def test_a_clip_that_already_looks_good_is_skipped():
    grade = color.estimate(_stats(_scene()), "auto")
    assert grade.skip == color.SKIP_GOOD
    assert grade.filter() == ""
    assert grade.before is not None and grade.before.m3 >= color.M3_FULL


def test_levels_stretch_to_the_luma_percentiles():
    grade = color.estimate(_stats(_flat(_scene(), 0.04, 0.9)), "auto")
    assert grade.skip is None
    assert 0.03 < grade.black <= color.MAX_BLACK
    assert color.MIN_WHITE <= grade.white < 0.92
    assert grade.predicted.black < grade.before.black
    assert grade.predicted.white > grade.before.white


def test_levels_are_clamped():
    black, white = color._levels_points(_flat(_scene(), 0.15, 0.6))
    assert (black, white) == (color.MAX_BLACK, color.MIN_WHITE)


def test_a_stretch_above_the_cap_moves_both_points_back(monkeypatch):
    # with the plan's clamps the stretch is at most 1 / 0.82; looser ones reach the cap
    monkeypatch.setattr(color, "MAX_BLACK", 0.2)
    monkeypatch.setattr(color, "MIN_WHITE", 0.7)
    black, white = color._levels_points(_flat(_scene(), 0.15, 0.6))
    assert 1 / (white - black) == pytest.approx(color.MAX_STRETCH)
    assert black < 0.15 and white > color.MIN_WHITE


def test_white_balance_corrects_part_of_a_cast_from_the_neutral_mid_tones():
    rgb = _cast(_scene()[:, :29], 1.08, 1.0, 0.9)  # the gray ramp
    r, g, b = color._gains(rgb, 0.0, underwater=False)
    avg = (1.08 + 1.0 + 0.9) / 3  # 60 % of the cast, against the channels' average
    assert (r, g, b) == pytest.approx(
        [1 + color.WB_AMOUNT * (avg / c - 1) for c in (1.08, 1.0, 0.9)], abs=1e-3
    )
    # the black point comes off before the channels are compared
    assert color._gains(rgb + 0.05, 0.05, underwater=False) == pytest.approx((r, g, b), abs=1e-3)


def test_white_balance_gains_are_clamped(monkeypatch):
    monkeypatch.setattr(color, "MAX_GAIN", 0.05)
    gains = color._gains(_cast(_scene()[:, :29], 1.08, 1.0, 0.9), 0.0, underwater=False)
    assert gains[2] == pytest.approx(1.05)  # 1.062 unclamped
    assert gains[0] == pytest.approx(0.952, abs=1e-3)  # within the clamp


def test_underwater_white_balance_is_gray_world_and_lifts_red_up_to_60_percent():
    gains = color._gains(_cast(_scene(), 0.3, 1.0, 1.0), 0.0, underwater=True)
    assert gains == pytest.approx((1 + color.MAX_RED_GAIN, 1 - color.MAX_GAIN, 1 - color.MAX_GAIN))


def test_white_balance_needs_two_percent_neutral_pixels():
    rgb = _scene()[:, 29:]  # the coloured patches only
    rgb = rgb[..., [0, 0, 2]] * np.array([1.0, 0.3, 1.0], np.float32)  # no gray left
    assert color._gains(rgb, 0.0, underwater=False) == (1.0, 1.0, 1.0)


def test_a_cast_clip_gets_a_grade_that_reduces_the_cast(lenient):
    grade = color.estimate(_stats(_cast(_flat(_scene(), 0.03, 0.93), 1.1, 1.0, 0.88)), "auto")
    assert grade.skip is None, grade
    assert grade.gains[0] < 1 < grade.gains[2]
    assert grade.predicted.residual < grade.before.residual
    assert "colorlevels=" in grade.filter() and "colorchannelmixer" not in grade.filter()


def test_vibrance_lifts_dull_colour_within_the_m3_veto():
    dull = _scene() * 0.3 + (_scene() @ color.LUMA)[..., None] * 0.7
    grade = color.estimate(_stats(_flat(dull, 0.03, 0.93)), "auto")
    assert 0 < grade.vibrance <= color.MAX_VIBRANCE
    assert grade.predicted.m3 > grade.before.m3
    assert grade.predicted.m3 <= grade.before.m3 * (1 + color.VETO_M3_RISE)
    assert "vibrance=intensity=" in grade.filter()


def _log_like() -> np.ndarray:
    return _flat(_scene() * 0.3 + 0.35, 0.08, 0.8)  # low saturation, lifted blacks


def test_flat_looking_10_bit_footage_is_skipped_as_log():
    stats = replace(_stats(_log_like()), high_bit=True)
    assert color.is_log(stats)
    grade = color.estimate(stats, "auto")
    assert (grade.skip, grade.filter()) == (color.SKIP_LOG, "")


def test_flat_looking_8_bit_footage_is_graded():
    """The rule of ``encode.warn_if_log``: 8-bit footage is not log; hazy 8-bit clips get a
    grade, the veto guarding it."""
    stats = _stats(_log_like())
    assert not stats.high_bit and not color.is_log(stats)
    grade = color.estimate(stats, "auto")
    assert grade.skip is None, grade
    assert grade.filter() != ""


def _underwater() -> np.ndarray:
    return _cast(_flat(_scene(), 0.05, 0.8), 0.35, 1.0, 0.95)


def test_underwater_borrows_red_from_green_and_lifts_red(lenient):
    stats = _stats(_underwater())
    assert color.looks_underwater(stats)
    grade = color.estimate(stats, "underwater")
    assert 0 < grade.mix <= color.MAX_MIX
    assert 1 + color.MAX_GAIN < grade.gains[0] <= 1 + color.MAX_RED_GAIN
    assert "colorchannelmixer=rr=" in grade.filter()
    graded = color.simulate(stats.frames, grade).reshape(-1, 3).mean(axis=0)
    before = color._float(stats.frames).reshape(-1, 3).mean(axis=0)
    assert graded[0] / graded[1] > before[0] / before[1] + 0.3


def test_plain_auto_never_mixes_channels_even_underwater():
    grade = color.estimate(_stats(_underwater()), "auto")
    assert grade.mix == 0
    assert all(abs(g - 1) <= color.MAX_GAIN + 1e-6 for g in grade.gains)


def test_land_footage_does_not_look_underwater():
    assert not color.looks_underwater(_stats(_scene()))


def test_mix_is_capped_and_none_once_red_is_there():
    assert color._mix(_cast(_scene(), 0.05, 1.0, 1.0)) == color.MAX_MIX
    assert color._mix(_scene()) == 0.0


def test_a_two_sided_stretch_clipping_both_tails_is_not_vetoed():
    """p0.5 / p99.5 levels clip about 0.5 % of the pixels at each end: more than 0.5 points
    in all, but that is the clipping they intend, so only clipping beyond it is vetoed."""
    frames = color._float(_stats(_flat(_scene(), 0.04, 0.9)).frames)
    grade = color.estimate(ColorStats(frames), "auto")
    assert (grade.black, grade.white) == (color.MAX_BLACK, color.MIN_WHITE)
    assert grade.skip is None, grade
    assert grade.predicted.clipped - grade.before.clipped > color.VETO_CLIPPED
    assert grade.predicted.clipped <= color._intended_clipping(frames, grade) + color.VETO_CLIPPED


def test_intended_clipping_is_the_source_s_own_plus_the_levels_tails_at_most():
    frames = color._float(_stats(_flat(_scene(), 0.04, 0.9)).frames)
    frames[:, :2, :8] = 1.0  # already clipped: 16 of 9216 pixels
    already = 16 / frames[0, ..., 0].size
    tails = color.LEVELS_PERCENTILES[0] / 100 + 1 - color.LEVELS_PERCENTILES[1] / 100
    assert color._intended_clipping(frames, AutoGrade("auto")) == pytest.approx(already, abs=1e-5)
    grade = AutoGrade("auto", black=0.06, white=0.88)
    assert already < color._intended_clipping(frames, grade) <= already + tails
    # a weaker grade's points are nearer 0 and 1 and intend less
    weaker = color._intended_clipping(frames, replace(grade, strength=0.3))
    assert weaker < color._intended_clipping(frames, grade)


def test_clipping_beyond_the_levels_intent_is_vetoed():
    """The same two-sided stretch with a strong white balance gain: red is pushed out on top
    of the levels' tails."""
    frames = color._float(_stats(_flat(_scene(), 0.04, 0.9)).frames)
    before = color.metrics(frames)
    grade = AutoGrade("auto", black=0.06, white=0.88, gains=(1.3, 1.0, 1.0))
    after = color.metrics(color.simulate(frames, grade))
    intended = color._intended_clipping(frames, grade)
    assert after.clipped - intended > color.VETO_CLIPPED
    assert color._veto(before, after, intended) == color.VETO_CLIPPING
    plain = color.metrics(color.simulate(frames, replace(grade, gains=(1.0, 1.0, 1.0))))
    assert color._veto(before, plain, intended) is None


def test_a_grade_that_clips_too_much_is_vetoed():
    frames = _flat(_scene(), 0.04, 0.95)
    # a third of every frame sits at the black point and would clip: far more than the p0.5
    # tail the levels intend
    frames[:, :, :40] = 0.05
    grade = color.estimate(_stats(frames), "auto")
    assert grade.skip == color.VETO_CLIPPING
    assert grade.filter() == ""
    assert grade.predicted.clipped - grade.before.clipped > 0.05


@pytest.mark.parametrize(
    ("after", "veto", "intended"),
    [
        ({"clipped": 0.016}, color.VETO_CLIPPING, 0.01),
        ({"clipped": 0.015}, None, 0.01),
        ({"clipped": 0.026}, color.VETO_CLIPPING, 0.02),
        ({"clipped": 0.0249}, None, 0.02),
        ({"clipped": 0.016}, color.VETO_CLIPPING, 0.0),  # never below the source's own
        ({"residual": 0.106}, color.VETO_CAST, 0.01),
        ({"residual": 0.1045}, None, 0.01),
        ({"m3": 25.1}, color.VETO_COLOURFUL, 0.01),
        ({"m3": 25.0}, None, 0.01),
    ],
)
def test_veto_thresholds(after, veto, intended):
    before = Metrics(m3=20.0, residual=0.1, clipped=0.01, black=0.05, white=0.9)
    assert color._veto(before, Metrics(**(before.to_json() | after)), intended) == veto


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError, match="unknown color mode"):
        color.estimate(_stats(_scene()), "vivid")


# -- the grade: strength, filter, manifest ------------------------------------------------------

GRADE = AutoGrade("auto", black=0.05, white=0.9, gains=(1.1, 1.0, 0.9), vibrance=0.2)


def test_filter_grades_in_bt709_16_bit_rgb_and_tags_like_a_lut():
    vf = GRADE.filter()
    assert vf.startswith("scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=gbrp16le,")
    assert vf.endswith(
        "scale=out_color_matrix=bt709:out_range=tv,"
        "setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709:range=tv"
    )
    assert vf.index("colorlevels=") < vf.index("vibrance=")
    mixed = AutoGrade("underwater", mix=0.3, gains=(1.4, 1.0, 0.95)).filter()
    assert mixed.index("colorchannelmixer=rr=0.70000:rg=0.30000") < mixed.index("colorlevels=")


def test_levels_put_white_balance_and_stretch_in_one_line_per_channel():
    vf = color._levels(0.05, 0.95, (1.1, 1.0, 0.88))
    # red reaches 1 at 0.05 + 0.9 / 1.1; blue would only at 0.05 + 0.9 / 0.88 = 1.073, so
    # its output white is lowered to (1 - 0.05) * 0.88 / 0.9 instead
    assert vf == (
        "colorlevels=rimin=0.05000:rimax=0.86818:romax=1.00000:"
        "gimin=0.05000:gimax=0.95000:gomax=1.00000:"
        "bimin=0.05000:bimax=1.00000:bomax=0.92889"
    )


def test_strength_is_folded_into_the_parameters():
    half = replace(GRADE, strength=0.5).applied()
    assert (half.black, half.white, half.vibrance) == pytest.approx((0.025, 0.95, 0.1))
    assert half.gains == pytest.approx((1.05, 1.0, 0.95))
    assert "blend" not in replace(GRADE, strength=0.5).filter()
    assert replace(GRADE, strength=0.0).filter() == ""


def test_identity_and_skipped_grades_have_no_filter():
    assert AutoGrade("auto").filter() == ""
    assert AutoGrade.skipped("auto", color.SKIP_360).filter() == ""
    assert AutoGrade("auto", black=0.05, skip=color.VETO_CAST).filter() == ""


def test_manifest_entry_records_the_grade_and_reads_it_back():
    grade = color.estimate(_stats(_cast(_flat(_scene(), 0.03, 0.93), 1.1, 1.0, 0.88)), "auto", 0.7)
    entry = grade.to_json()
    assert set(entry["color"]) == {
        "method", "version", "strength", "params", "skip", "before", "predicted",
    }  # fmt: skip
    assert entry["color"]["method"] == "auto" and entry["color"]["strength"] == 0.7
    back = AutoGrade.from_entry(json.loads(json.dumps({"file": "001_a.mp4", **entry})))
    assert back == grade
    assert back.filter() == grade.filter()
    assert (back.before, back.predicted) == (grade.before, grade.predicted)


def test_grades_compare_by_parameters_only():
    a = AutoGrade("auto", black=0.05, before=Metrics(1, 2, 3, 4, 5))
    assert a == AutoGrade("auto", black=0.05)
    assert a != AutoGrade("auto", black=0.05, strength=0.5)
    assert a != AutoGrade("underwater", black=0.05)


@pytest.mark.parametrize("entry", [{}, {"lut": "/l/a.cube"}, {"color": {"method": "off"}}])
def test_entries_without_an_automatic_grade_read_as_none(entry):
    assert AutoGrade.from_entry(entry) is None


def test_estimated_parameters_are_rounded():
    grade = color.estimate(_stats(_cast(_flat(_scene(), 0.03, 0.93), 1.1, 1.0, 0.88)), "auto")
    for v in (grade.black, grade.white, *grade.gains, grade.mix, grade.vibrance):
        assert v == round(v, 4)


def _warm() -> np.ndarray:
    return _cast(_flat(_scene(), 0.03, 0.93), 1.1, 1.0, 0.88)


def test_per_channel_levels_give_a_point_per_channel_and_luma_stays_as_it_was():
    stats = _stats(_warm())
    chan = color.estimate(stats, "auto", levels="channel")
    assert isinstance(chan.black, tuple) and isinstance(chan.white, tuple)
    assert len(chan.black) == len(chan.white) == 3 and len(set(chan.white)) > 1
    luma = color.estimate(stats, "auto")
    assert isinstance(luma.black, float) and isinstance(luma.white, float)
    assert luma == color.estimate(stats, "auto", levels="luma")


def test_unknown_levels_are_refused():
    with pytest.raises(ValueError, match="unknown levels"):
        color.estimate(_stats(_scene()), "auto", levels="rgb")


def test_per_channel_points_give_each_channel_its_own_colorlevels_and_survive_the_manifest():
    grade = AutoGrade("auto", black=(0.02, 0.05, 0.08), white=(0.9, 0.95, 0.85))
    vf = grade.filter()
    for c, lo, hi in (("r", "0.02000", "0.90000"), ("g", "0.05000", "0.95000"),
                      ("b", "0.08000", "0.85000")):  # fmt: skip
        assert f"{c}imin={lo}:{c}imax={hi}:{c}omax=1.00000" in vf
    entry = json.loads(json.dumps({"file": "001_a.mp4", **grade.to_json()}))
    assert entry["color"]["params"]["black"] == [0.02, 0.05, 0.08]
    back = AutoGrade.from_entry(entry)
    assert back == grade and isinstance(back.black, tuple) and back.filter() == vf
    # strength scales each channel's point
    half = replace(grade, strength=0.5).applied()
    assert half.black == pytest.approx((0.01, 0.025, 0.04))


def test_only_plain_auto_notes_that_frames_look_underwater():
    stats = _stats(_underwater())
    assert color.estimate(stats, "auto").underwater_hint
    assert not color.estimate(stats, "underwater").underwater_hint
    assert not color.estimate(_stats(_scene()), "auto").underwater_hint
    assert AutoGrade("auto", black=0.05, underwater_hint=True) == AutoGrade("auto", black=0.05)


def test_the_underwater_hint_survives_the_cache(monkeypatch, tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x")
    calls: list[tuple] = []
    stats = _stats(_underwater())
    monkeypatch.setattr(color, "sample_stats", lambda *a, **kw: calls.append(a) or stats)
    info = media_info(src, **BT709)
    first = color.auto_grade(src, info, 1.0, 5.0, "auto", FMT)
    again = color.auto_grade(src, info, 1.0, 5.0, "auto", FMT)
    assert len(calls) == 1 and first.underwater_hint and again.underwater_hint


@pytest.mark.parametrize(
    ("tags", "high_bit"),
    [
        ({"pix_fmt": "yuv420p10le"} | BT709, True),
        ({"pix_fmt": "yuv420p10le"}, True),  # untagged
        ({"pix_fmt": "yuv420p"} | BT709, False),
        ({"pix_fmt": "yuv420p10le"} | HLG, False),  # HDR gets the tone map, not the log check
    ],
    ids=["10-bit-709", "10-bit-untagged", "8-bit", "10-bit-hlg"],
)
def test_is_high_bit_sdr(tags, high_bit):
    assert encode.is_high_bit_sdr(media_info(**tags)) is high_bit


# -- metrics and simulation ---------------------------------------------------------------------


def test_metrics_of_gray_frames_have_no_colour_and_no_cast():
    gray = np.repeat(np.linspace(0.1, 0.9, 64, dtype=np.float32)[None, None, :, None], 3, -1)
    m = color.metrics(np.repeat(gray, 8, axis=1))
    assert (m.m3, m.residual, m.clipped) == (0, 0, 0)
    assert (m.black, m.white) == pytest.approx((0.1, 0.9), abs=1e-3)


def test_metrics_see_a_cast_clipping_and_colourfulness():
    m = color.metrics(_scene())
    cast = color.metrics(_cast(_scene(), 1.2, 1.0, 0.8))
    assert cast.residual > m.residual + 0.05
    assert cast.clipped > m.clipped
    assert color.metrics(_scene() * 0.5 + 0.25).m3 < m.m3


def test_gray_world_residual_ignores_a_common_levels_stretch():
    rgb = _cast(_flat(_scene(), 0.1, 0.8), 1.1, 1.0, 0.9)
    stretched = color.simulate(rgb, AutoGrade("auto", black=0.05, white=0.9))
    assert color.metrics(stretched).residual == pytest.approx(color.metrics(rgb).residual, rel=0.05)


def test_simulation_of_an_identity_grade_changes_nothing():
    frames = _stats(_scene()).frames
    assert np.array_equal(color.simulate(frames, AutoGrade("auto")), color._float(frames))


def test_simulated_vibrance_leaves_gray_and_full_saturation_alone():
    px = np.array([[[[0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [0.6, 0.4, 0.3]]]], np.float32)
    out = color.simulate(px, AutoGrade("auto", vibrance=0.2))[0, 0]
    assert out[0] == pytest.approx([0.5] * 3) and out[1] == pytest.approx([1, 0, 0])
    luma = px[0, 0, 2] @ color.LUMA
    assert out[2] == pytest.approx(luma + (px[0, 0, 2] - luma) * (1 + 0.2 * 0.7))


# -- sampling and cache -------------------------------------------------------------------------

FMT = encode.OutputFormat(1920, 1080, 30)


def test_sample_filter_mirrors_the_export_crop_and_bt709_chain():
    info = media_info(**BT709)
    vf, size = color.sample_filter(info, FMT, False, Crop(1364, 750, 10, 20))
    box = Crop(1364, 750, 10, 20).fit_aspect(16 / 9)
    assert vf == (
        f"crop={box.w}:{box.h}:{box.x}:{box.y},scale=256:144:flags=area:out_range=tv,"
        "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24"
    )
    assert size == (256, 144)


def test_sample_filter_converts_bt601_and_takes_the_vertical_crop():
    info = media_info(width=720, height=480, **BT601)
    vf, size = color.sample_filter(info, FMT, True)
    assert vf.startswith("crop='min(iw,ih*9/16)':'min(ih,iw*16/9)',scale=256:456:")
    assert "colorspace=all=bt709:iall=smpte170m" in vf
    assert vf.index("colorspace=") < vf.index("in_color_matrix=bt709")
    assert size == (256, 456)


@pytest.fixture
def decodes(monkeypatch):
    """``decodes(n)``: ``decode_raw`` yields ``n`` keyframes (each frame filled with its index)
    and then, for the fallback, one frame of 200; the calls are recorded."""
    calls: list[dict] = []
    count = [0]

    def fake(path, *, size, keyframes_only=False, **kw):
        calls.append({"keyframes_only": keyframes_only, **kw})
        n = count[0] if keyframes_only else 1
        for i in range(n):
            yield float(i), np.full((size[1], size[0], 3), i if keyframes_only else 200, np.uint8)

    monkeypatch.setattr(color, "decode_raw", fake)

    def run(n: int, **over) -> tuple[ColorStats, list[dict]]:
        count[0] = n
        calls.clear()
        info = media_info(**(BT709 | over))
        return color.sample_stats(Path("v.mp4"), info, 10.0, 20.0, FMT), calls

    return run


def test_sampling_reads_keyframes_of_the_inner_80_percent_on_the_cpu(decodes):
    stats, calls = decodes(5)
    assert stats.frames.shape == (5, 144, 256, 3)
    (call,) = calls
    assert call["keyframes_only"] and call["hwaccel"] is None
    assert (call["start"], call["duration"]) == pytest.approx((11.0, 8.0))


@pytest.mark.parametrize(
    ("over", "high_bit"),
    [
        ({}, False),
        ({"pix_fmt": "yuv420p10le"}, True),
        ({"pix_fmt": "p010le"}, True),
        ({"pix_fmt": "yuv420p10le", **HLG}, False),  # tone-mapped: not checked for log
    ],
)
def test_sampling_records_a_high_bit_source_like_the_log_warning(decodes, over, high_bit):
    stats, _ = decodes(2, **over)
    assert stats.high_bit is high_bit


def test_sampling_keeps_at_most_8_frames_spread_over_the_clip(decodes):
    stats, _ = decodes(100)
    picked = stats.frames[:, 0, 0, 0].tolist()
    assert len(picked) == color.SAMPLE_FRAMES
    assert picked[0] == 0 and picked[-1] >= 90
    assert min(np.diff(picked)) >= 8


def test_a_clip_without_keyframes_is_sampled_at_its_middle(decodes):
    stats, calls = decodes(0)
    assert stats.frames.shape[0] == 1 and stats.frames[0, 0, 0, 0] == 200
    assert not calls[1]["keyframes_only"] and calls[1]["start"] == pytest.approx(15.0)


@pytest.fixture
def measured(monkeypatch, tmp_path):
    """A source file and the list of ``sample_stats`` calls (which return a cast scene)."""
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x")
    calls: list[tuple] = []
    frames = _stats(_cast(_flat(_scene(), 0.03, 0.93), 1.1, 1.0, 0.88))

    def fake(*args, **_kw):
        calls.append(args)
        return frames

    monkeypatch.setattr(color, "sample_stats", fake)
    return src, calls


def test_auto_grade_is_cached_per_source_and_measurement(measured):
    src, calls = measured
    info = media_info(src, **BT709)
    first = color.auto_grade(src, info, 1.0, 5.0, "auto", FMT)
    again = color.auto_grade(src, info, 1.0, 5.0, "auto", FMT)
    assert len(calls) == 1 and again == first and again.predicted == first.predicted
    (path,) = (cache.source_dir(src) / "color").glob("*.json")
    assert AutoGrade.from_entry(cache.load_json(path)) == first

    color.auto_grade(src, info, 1.0, 5.0, "auto", FMT, strength=0.5)
    color.auto_grade(src, info, 1.0, 6.0, "auto", FMT)
    color.auto_grade(src, info, 1.0, 5.0, "underwater", FMT)
    color.auto_grade(src, info, 1.0, 5.0, "auto", FMT, crop=Crop(1280, 720, 0, 0))
    assert len(calls) == 5


def test_a_clip_that_cannot_be_decoded_is_skipped_and_not_cached(monkeypatch, tmp_path):
    src = tmp_path / "v.mp4"
    src.write_bytes(b"x")

    def broken(*_a, **_kw):
        raise FFmpegError("no frames")

    monkeypatch.setattr(color, "sample_stats", broken)
    grade = color.auto_grade(src, media_info(src), 1.0, 5.0, "auto", FMT)
    assert grade == AutoGrade.skipped("auto", color.SKIP_NO_FRAMES)
    assert not list((cache.source_dir(src) / "color").glob("*.json"))


# -- real ffmpeg: the simulation against the filters --------------------------------------------

# Largest and mean difference (8-bit codes) of the simulation from ffmpeg's filters on 8-bit
# RGB in and out: ffmpeg truncates its 16-bit results, the simulation does not round.
SIM_MAX = 1.0
SIM_MEAN = 0.3
# Through the whole export chain (BT.709 tv-range YUV 4:4:4 in and out, as 8-bit): the two
# YUV/RGB conversions add their rounding.
CHAIN_MAX = 5.0
CHAIN_MEAN = 1.0

SIM_GRADES = [
    AutoGrade("auto", black=0.05, white=0.9, gains=(1.1, 1.0, 0.9)),
    AutoGrade("auto", black=0.0, white=0.95, gains=(0.88, 1.0, 1.12)),
    AutoGrade("auto", vibrance=0.2),
    AutoGrade("underwater", mix=0.35),
    AutoGrade("underwater", mix=0.3, black=0.04, white=0.92, gains=(1.6, 1.0, 0.88), vibrance=0.15),
    AutoGrade("auto", black=0.05, white=0.9, gains=(1.1, 1.0, 0.9), vibrance=0.2, strength=0.5),
    AutoGrade("auto", black=(0.02, 0.05, 0.08), white=(0.9, 0.95, 0.85), gains=(1.1, 1.0, 0.9)),
]


def _pipe(data: np.ndarray, vf: str, in_fmt: str, out_fmt: str = "rgb24") -> np.ndarray:
    """``data`` (raw ``in_fmt`` frame bytes, ``(h, w, 3)``) through ``vf``, as ``out_fmt``."""
    h, w = data.shape[:2]
    out = subprocess.run(
        [
            *(shutil.which("ffmpeg"), "-v", "error", "-f", "rawvideo", "-pix_fmt", in_fmt),
            *("-s", f"{w}x{h}", "-i", "-", "-vf", vf, "-f", "rawvideo", "-pix_fmt", out_fmt, "-"),
        ],
        input=data.tobytes(),
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(out, np.uint8).reshape(h, w, 3)


@needs_ffmpeg
@pytest.mark.parametrize("grade", SIM_GRADES, ids=lambda g: g.filter().split(",")[2][:16])
def test_real_filters_match_the_simulation(grade):
    rgb = np.random.default_rng(3).integers(0, 256, (64, 96, 3), dtype=np.uint8)
    steps = grade.filter().split(",format=gbrp16le,")[1].split(",scale=out_color_matrix")[0]

    got = _pipe(rgb, f"format=gbrp16le,{steps},format=rgb24", "rgb24").astype(np.float32)

    diff = np.abs(got - color.simulate(rgb[None], grade)[0] * 255)
    assert diff.max() <= SIM_MAX and diff.mean() <= SIM_MEAN, (diff.max(), diff.mean())


@needs_ffmpeg
@pytest.mark.parametrize("grade", SIM_GRADES, ids=lambda g: g.filter().split(",")[2][:16])
def test_real_export_chain_matches_the_simulation(grade):
    """The whole :meth:`AutoGrade.filter` on tv-range BT.709 YUV, against the simulation of the
    same frame read through the sampling's RGB conversion."""
    rgb = np.round(_scene(1)[0] * 255).astype(np.uint8)
    to_rgb = "scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24"
    # 4:4:4, so chroma subsampling does not blur the patches; the array holds its planar bytes
    yuv = _pipe(rgb, "scale=out_color_matrix=bt709:out_range=tv", "rgb24", "yuv444p")
    src = _pipe(yuv, to_rgb, "yuv444p")  # the frame as the sampling sees it

    got = _pipe(yuv, f"{grade.filter()},format=yuv444p,{to_rgb}", "yuv444p").astype(np.float32)

    diff = np.abs(got - color.simulate(src[None], grade)[0] * 255)
    assert diff.max() <= CHAIN_MAX and diff.mean() <= CHAIN_MEAN, (diff.max(), diff.mean())


def _cast_source(path: Path) -> Path:
    """3 s of :func:`_scene` frames, 128x72 at 10 fps, low contrast with a warm cast, BT.709
    tv range, FFV1 (every frame a keyframe)."""
    frames = _cast(_flat(_scene(30), 0.03, 0.93), 1.1, 1.0, 0.88)
    tags = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
    subprocess.run(
        [
            *(shutil.which("ffmpeg"), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24"),
            *("-s", "128x72", "-r", "10", "-i", "-"),
            *("-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv444p"),
            *("-c:v", "ffv1", "-color_range", "tv", *tags, str(path)),
        ],
        input=np.round(frames * 255).astype(np.uint8).tobytes(),
        check=True,
    )  # fmt: skip
    return path


@needs_ffmpeg
def test_real_sampling_and_grade_of_a_cast_clip(tmp_path, lenient):
    src = _cast_source(tmp_path / "warm.mkv")
    info = ffmpeg_utils.probe(src)
    fmt = encode.choose_format([info], False)

    stats = color.sample_stats(src, info, 0.0, 3.0, fmt)

    assert stats.frames.shape == (color.SAMPLE_FRAMES, 144, 256, 3)  # 128x72 scaled up
    grade = color.estimate(stats, "auto")
    assert grade.skip is None, grade
    assert grade.gains[0] < 1 < grade.gains[2] and grade.black > 0.02 and grade.white < 0.95
    # the grade runs where the export puts it
    vf = fmt.video_filter(False, None, color=grade.filter())
    ffmpeg_utils.run(["ffmpeg", "-v", "error", "-i", str(src), "-vf", vf, "-f", "null", "-"])


def _underwater_source(path: Path) -> Path:
    """3 s of :func:`_underwater` frames, 128x72 at 10 fps, BT.709 tv range, FFV1."""
    frames = _cast(_flat(_scene(30), 0.05, 0.8), 0.35, 1.0, 0.95)
    tags = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
    subprocess.run(
        [
            *(shutil.which("ffmpeg"), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24"),
            *("-s", "128x72", "-r", "10", "-i", "-"),
            *("-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv444p"),
            *("-c:v", "ffv1", "-color_range", "tv", *tags, str(path)),
        ],
        input=np.round(frames * 255).astype(np.uint8).tobytes(),
        check=True,
    )  # fmt: skip
    return path


@needs_ffmpeg
def test_real_clips_color_underwater_records_the_grade_and_extend_redoes_it(
    home, tmp_path, monkeypatch, capsys, caplog, lenient
):
    monkeypatch.setenv("CUTPOINT_VIDEO_ENCODER", "cpu")
    caplog.set_level(logging.INFO)
    src = _underwater_source(tmp_path / "reef.mkv")
    seg = [{"source": str(src), "start": 0.5, "end": 2.5, "label": "a"}]
    segments = write(tmp_path / "seg.json", json.dumps(seg))
    code, doc = run_json(
        capsys, ["clips", "reef", "--segments", str(segments), "--color", "underwater"]
    )
    assert code == 0, doc["error"]
    clips_dir = home / "projects" / "reef" / "clips"
    entry = json.loads((clips_dir / "manifest.json").read_text(encoding="utf-8"))["clips"][0]
    grade = AutoGrade.from_entry(entry)
    assert grade is not None and grade.skip is None and grade.mix > 0, entry["color"]
    assert (clips_dir / "001_a.mp4").stat().st_size > 0
    assert "gaming" not in caplog.text  # no highlights.json: the style is unknown, not gaming

    seen = []

    def spy(*args):
        seen.append(args[-1])
        return encode.export_clip_checked(*args)

    def no_measuring(*_a, **_kw):
        raise AssertionError("extend measured the color again")

    monkeypatch.setattr(extend, "export_clip_checked", spy)
    monkeypatch.setattr(color, "sample_stats", no_measuring)
    out = extend.extend_clips(clips_dir, Padding(0.25, 0.25), speech=SpeechSettings(enabled=False))
    assert seen == [grade] and seen[0].filter() == grade.filter() != ""
    redone = json.loads((out / "manifest.json").read_text(encoding="utf-8"))["clips"][0]
    assert redone["color"]["params"] == entry["color"]["params"]
    assert (out / "001_a.mp4").stat().st_size > 0


@needs_ffmpeg
def test_color_eval_script_reports_every_mode_and_levels_variant(tmp_path, capsys, lenient):
    import importlib.util  # noqa: PLC0415

    path = Path(__file__).resolve().parents[1] / "scripts" / "color_eval.py"
    spec = importlib.util.spec_from_file_location("color_eval", path)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    src = _underwater_source(tmp_path / "reef.mkv")
    assert script.main([f"{src}@0.5-2.5", "--json"]) == 0

    (window,) = json.loads(capsys.readouterr().out)
    assert window["frames"] > 0
    assert {(v["mode"], v["levels"]) for v in window["variants"]} == {
        (m, lv) for m in color.MODES for lv in color.LEVELS
    }
