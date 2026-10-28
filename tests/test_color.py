"""Color handling of exports (#289, #290): probed tags, HDR/BT.2020 to SDR Rec.709, BT.601 to
BT.709, log warning."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import media_info, needs_ffmpeg
from cutpoint import assemble, encode, ffmpeg_utils
from cutpoint.crop import Crop
from cutpoint.ffmpeg_utils import FFmpegError, VideoEncoder

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


@pytest.mark.parametrize(("tags", "tin"), [(HLG, "arib-std-b67"), (PQ, "smpte2084")])
def test_hdr_source_is_tone_mapped_to_bt709(tags, tin):
    vf = encode.sdr_filter(media_info(pix_fmt="yuv420p10le", **tags), BOTH)
    assert vf.startswith(f"zscale=tin={tin}:pin=bt2020:min=bt2020nc:t=linear:npl=100:p=bt709,")
    assert "format=gbrpf32le,tonemap=tonemap=hable:desat=0," in vf
    assert vf.endswith("zscale=tin=linear:pin=bt709:t=bt709:p=bt709:m=bt709:r=tv")


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


@pytest.mark.parametrize("tags", [HLG, PQ])
def test_hdr_export_tone_maps_and_tags_the_output_bt709(monkeypatch, tags):
    args = _export_args(monkeypatch, media_info(pix_fmt="yuv420p10le", **tags))
    graph = args[args.index("-filter_complex") + 1]
    assert ",tonemap=tonemap=hable:desat=0," in graph
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
    assert "GX010001.MP4 looks like log/flat footage" in warnings[0] and "LUT" in warnings[0]


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
