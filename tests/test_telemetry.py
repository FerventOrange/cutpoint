"""Telemetry detectors on synthetic IMU, GPS and GPMF payloads: jumps, crashes, the clocks,
the GoPro SCEN/FACE streams and the joining of clips into runs. No camera files."""

from __future__ import annotations

import struct
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from cutpoint import telemetry
from cutpoint.telemetry import DT, Telemetry, TelemetryParams

P = TelemetryParams()


# ------------------------------------------------------------------------- airtime
def _jump(speed_kmh: float = 30.0, air_s: float = 0.6, landing_g: float = 3.0, gps: bool = True):
    """10 s at 200 Hz: 1 g, free fall from 3 s for ``air_s``, then a landing spike."""
    fs = 200.0
    t = np.arange(int(10 * fs)) / fs
    am = np.ones_like(t)
    air = (t >= 3.0) & (t < 3.0 + air_s)
    am[air] = 0.1
    am[(t >= 3.0 + air_s + 0.05) & (t < 3.0 + air_s + 0.1)] = landing_g
    speed = np.full(100, speed_kmh)
    gps_ok = np.full(100, 1.0 if gps else 0.0)
    return t, am, fs, speed, gps_ok


def test_a_jump_is_found_with_its_airtime_landing_and_speed():
    [jump] = telemetry._airtime_events(*_jump(), P)
    assert jump["type"] == "airtime"
    assert jump["t"] == pytest.approx(3.0, abs=0.05)
    assert jump["air_s"] == pytest.approx(0.6, abs=0.06)
    assert jump["landed"] is True and jump["landing_g"] == pytest.approx(3.0)
    assert jump["speed_kmh"] == 30.0
    assert jump["confidence"] == "high"
    assert jump["height_m"] == pytest.approx(telemetry.G * jump["air_s"] ** 2 / 8, abs=0.01)


@pytest.mark.parametrize(
    ("kwargs", "found", "confidence"),
    [
        ({"speed_kmh": 2.0}, 0, None),  # standing: the camera was dropped or handled
        ({"air_s": 0.1}, 0, None),  # a flicker, not a jump
        ({"air_s": 0.35, "landing_g": 1.2}, 1, "low"),  # short and no landing spike
        ({"air_s": 0.35}, 1, "medium"),
        ({"gps": False, "speed_kmh": 2.0}, 1, "high"),  # without a fix the speed is unknown
    ],
    ids=["standing", "flicker", "unlanded-short", "medium", "no-gps"],
)
def test_airtime_rules(kwargs, found, confidence):
    events = telemetry._airtime_events(*_jump(**kwargs), P)
    assert len(events) == found
    if found:
        assert events[0]["confidence"] == confidence
        if not kwargs.get("gps", True):
            assert events[0]["speed_kmh"] is None


# ------------------------------------------------------------------------- crashes
def _impact(stop: bool = True, gps: bool = True, tilt: bool = True):
    """20 s at 100 Hz: riding, a 6 g impact at 10 s, then a stop (or riding on)."""
    fs = 100.0
    t = np.arange(int(20 * fs)) / fs
    am = np.ones_like(t)
    am[(t >= 10.0) & (t < 10.03)] = 6.0
    grav = np.tile([0.0, 0.0, 1.0], (len(t), 1))
    if tilt:
        grav[t > 10.5] = [1.0, 0.0, 0.0]  # lying on the side: 90 degrees
    gyro = np.zeros_like(t)
    bins = np.arange(200) * DT
    speed = np.where(bins < 10.5, 30.0, 0.0 if stop else 30.0)
    vib = np.where(bins < 10.0, 0.3, 0.01 if stop else 0.3)
    if not gps:
        speed = np.full(200, np.nan)
    return t, am, grav, gyro, speed, vib


def test_an_impact_that_ends_in_a_stop_is_a_crash():
    [crash] = telemetry._crash_events(*_impact(), [], P)
    assert crash["type"] == "crash"
    assert crash["impact_t"] == pytest.approx(10.0, abs=0.03)
    assert crash["peak_g"] == 6.0
    assert crash["tilt_deg"] == pytest.approx(90, abs=1)
    assert crash["speed_before_kmh"] == 30.0 and crash["speed_after_kmh"] == 0.0
    assert crash["after_jump"] is False
    assert crash["score"] == pytest.approx(0.3 * (6.0 - 3.5) / 4.5 + 0.4 + 0.3, abs=0.01)


def test_a_hard_bump_while_riding_on_is_not_a_crash():
    assert telemetry._crash_events(*_impact(stop=False), [], P) == []


def test_without_gps_a_crash_is_an_impact_then_stillness():
    [crash] = telemetry._crash_events(*_impact(gps=False), [], P)
    assert crash["speed_before_kmh"] is None and crash["speed_after_kmh"] is None
    assert telemetry._crash_events(*_impact(gps=False, stop=False), [], P) == []


def test_a_crash_right_after_a_jump_is_marked_as_a_landing():
    landing = {"type": "airtime", "t": 9.0, "end": 9.6}
    [crash] = telemetry._crash_events(*_impact(), [landing], P)
    assert crash["after_jump"] is True


# ------------------------------------------------------------------------- GPMF payloads
def _klv(fourcc: str, size: int, records: list[bytes]) -> bytes:
    body = b"".join(records)
    head = fourcc.encode("latin1") + b"?" + bytes([size]) + struct.pack(">H", len(records))
    return head + body + b"\0" * (-len(body) % 4)


GPS_SCALE = [10_000_000, 10_000_000, 1000, 1000, 100, 1, 1000, 100, 1]


def _gps_rec(lat, lon, alt, v2d, days, secs, dop, fix) -> bytes:
    vals = [lat, lon, alt, v2d, v2d, days, secs, dop, fix]
    raw = [round(v * s) for v, s in zip(vals, GPS_SCALE, strict=True)]
    return struct.pack(">lllllllHH", *raw)


def _gps_group(t_us: int, *recs: bytes) -> dict:
    return {
        "GPS9": {
            "TimestampUs": t_us,
            "Scale": GPS_SCALE,
            "0x47505339": _klv("GPS9", 32, list(recs)),
        }
    }


def test_gps9_rows_are_scaled_and_spread_over_their_payload():
    rec = _gps_rec(47.1234567, 11.5, 2000.0, 10.0, 9000, 3600.0, 1.5, 3)
    groups = [_gps_group(0, rec, rec), _gps_group(1_000_000, rec)]

    rows = telemetry._gps9(groups)

    assert rows.shape == (3, 10)
    assert list(rows[:, 0]) == pytest.approx([0.0, 0.5, 1.0])  # evenly inside each payload
    assert list(rows[0, 1:]) == pytest.approx([47.1234567, 11.5, 2000, 10, 10, 9000, 3600, 1.5, 3])


def test_start_clock_prefers_the_first_gps_fix_over_the_camera_clock():
    no_fix = _gps_rec(47, 11, 2000, 0, 0, 0, 9.9, 0)
    fix = _gps_rec(47, 11, 2000, 0, 9000, 3600.0, 1.5, 3)
    gps = telemetry._gps9([_gps_group(0, no_fix, fix)])  # a last payload: 0.1 s apart
    default = {"Default": {"0x545a4f4e": "60", "0x43444154": "1700000000"}}

    start, tz, clock = telemetry._start_clock([default], gps)

    expected = datetime(2000, 1, 1, tzinfo=UTC) + timedelta(days=9000, seconds=3600 - 0.1)
    assert (start, tz, clock) == (expected, 60, "gps")


def test_start_clock_falls_back_to_the_camera_clock_minus_its_time_zone():
    default = {"Default": {"0x545a4f4e": "60", "0x43444154": "1700000000"}}
    start, tz, clock = telemetry._start_clock([default], np.zeros((0, 10)))
    assert clock == "camera" and tz == 60
    assert start == datetime.fromtimestamp(1_700_000_000, tz=UTC) - timedelta(minutes=60)
    assert telemetry._start_clock([{"Default": {}}], np.zeros((0, 10))) == (None, None, "none")


def test_scen_reads_the_indoor_probability():
    probs = [b"SNOW" + struct.pack(">f", 0.1), b"INDO" + struct.pack(">f", 0.75)]
    groups = [{"SCEN": {"TimestampUs": 2_000_000, "0x5343454e": _klv("SCEN", 8, probs)}}]
    t, indoor = telemetry._scen_indoor(groups)
    assert list(t) == [2.0] and list(indoor) == pytest.approx([0.75])


def test_faces_report_the_largest_box_per_payload():
    big = struct.pack(">BBHHHHHBB", 1, 90, 1, 0, 0, 32767, 32767, 0, 0)
    small = struct.pack(">BBHHHHHBB", 1, 90, 2, 0, 0, 6553, 6553, 0, 0)
    groups = [
        {"FACE": {"TimestampUs": 1_000_000, "0x46414345": _klv("FACE", 14, [small, big])}},
        {"FACE": {"TimestampUs": 2_000_000, "0x46414345": _klv("FACE", 14, [])}},
    ]
    t, area = telemetry._faces(groups)
    assert list(t) == [1.0, 2.0]
    assert list(area) == pytest.approx([0.25, 0.0], abs=0.001)


def _nested(raw: bytes, cut: int = 5) -> list[list[int]]:
    """``raw`` as telemetry-parser can hand it over (#985): a list of lists of ints."""
    return [list(raw[i : i + cut]) for i in range(0, len(raw), cut)]


def test_nested_list_payloads_parse_as_if_flat():
    rec = _gps_rec(46.5, 7.25, 1200.0, 3.0, 9100, 7200.0, 1.2, 3)
    gps = _gps_group(0, rec, rec)
    gps["GPS9"]["0x47505339"] = _nested(gps["GPS9"]["0x47505339"])
    probs = [b"INDO" + struct.pack(">f", 0.4), b"URBA" + struct.pack(">f", 0.2)]
    scen = _klv("SCEN", 8, probs)
    face = _klv("FACE", 14, [struct.pack(">BBHHHHHBB", 1, 80, 3, 0, 0, 32767, 13107, 0, 0)])
    flat = [
        _gps_group(0, rec, rec),
        {"SCEN": {"TimestampUs": 1_000_000, "0x5343454e": scen}},
        {"FACE": {"TimestampUs": 1_000_000, "0x46414345": face}},
    ]
    nested = [
        gps,
        {"SCEN": {"TimestampUs": 1_000_000, "0x5343454e": _nested(scen, 7)}},
        {"FACE": {"TimestampUs": 1_000_000, "0x46414345": _nested(face, 3)}},
    ]

    assert telemetry._gps9(nested).tolist() == telemetry._gps9(flat).tolist()
    assert telemetry._gps9(nested).shape == (2, 10)
    for fn in (telemetry._scen_indoor, telemetry._faces):
        got, want = fn(nested), fn(flat)
        assert [a.tolist() for a in got] == [a.tolist() for a in want]
    assert telemetry._scen_indoor(nested)[1].tolist() == pytest.approx([0.4])


@pytest.mark.parametrize(
    "garbage",
    ["GPS9", 7, [[1, 2], "x"], [[1, [2]]], [300, 1], [[-1]], {"a": 1}, [True, 0]],
)
def test_an_unusable_raw_payload_is_skipped(garbage, caplog):
    rec = _gps_rec(46.5, 7.25, 1200.0, 3.0, 9100, 7200.0, 1.2, 3)
    bad_gps = {"GPS9": {"TimestampUs": 1_000_000, "Scale": GPS_SCALE, "0x47505339": garbage}}
    groups = [
        _gps_group(0, rec),
        bad_gps,
        {"SCEN": {"TimestampUs": 1_000_000, "0x5343454e": garbage}},
        {"FACE": {"TimestampUs": 1_000_000, "0x46414345": garbage}},
    ]

    with caplog.at_level("DEBUG", logger="cutpoint.telemetry"):
        rows = telemetry._gps9(groups)
        scen_t, _ = telemetry._scen_indoor(groups)
        face_t, _ = telemetry._faces(groups)

    assert rows.shape == (1, 10) and rows[0, 1] == pytest.approx(46.5)  # the good one stays
    assert len(scen_t) == 0 and len(face_t) == 0
    debug = [r.getMessage() for r in caplog.records if r.levelname == "DEBUG"]
    assert [m.split(":")[0] for m in debug] == ["GPS9", "SCEN", "FACE"]


def test_a_payload_shorter_than_its_declared_repeat_is_skipped(caplog):
    rec = _gps_rec(46.5, 7.25, 1200.0, 3.0, 9100, 7200.0, 1.2, 3)
    probs = [b"INDO" + struct.pack(">f", 0.4), b"URBA" + struct.pack(">f", 0.2)]
    face = struct.pack(">BBHHHHHBB", 1, 80, 3, 0, 0, 32767, 13107, 0, 0)
    cut_gps = _gps_group(1_000_000, rec, rec)
    cut_gps["GPS9"]["0x47505339"] = cut_gps["GPS9"]["0x47505339"][:-20]  # 2 records declared
    groups = [
        _gps_group(0, rec),
        cut_gps,
        {"SCEN": {"TimestampUs": 1_000_000, "0x5343454e": _klv("SCEN", 8, probs)[:-5]}},
        {"FACE": {"TimestampUs": 1_000_000, "0x46414345": _klv("FACE", 14, [face, face])[:-9]}},
    ]

    with caplog.at_level("DEBUG", logger="cutpoint.telemetry"):
        rows = telemetry._gps9(groups)
        scen_t, _ = telemetry._scen_indoor(groups)
        face_t, _ = telemetry._faces(groups)

    assert rows.shape == (1, 10) and rows[0, 1] == pytest.approx(46.5)  # the whole one stays
    assert len(scen_t) == 0 and len(face_t) == 0
    debug = [r.getMessage() for r in caplog.records if r.levelname == "DEBUG"]
    assert [m.split(":")[0] for m in debug] == ["GPS9", "SCEN", "FACE"]
    assert all("truncated payload" in m for m in debug)


# ------------------------------------------------------------------------- clocks and runs
def _tel(source: str, start: datetime | None, seconds: float, lift_from: float | None = None):
    n = int(seconds / DT)
    lift = np.zeros(n)
    if lift_from is not None:
        lift[int(lift_from / DT) :] = 1
    grid = {
        "speed_kmh": np.full(n, 40.0), "gps_ok": np.ones(n), "alt_m": np.linspace(2000, 1900, n),
        "lift_mask": lift, "still_mask": np.zeros(n), "accel_g": np.ones(n), "faces": np.zeros(n),
    }  # fmt: skip
    meta = {"start_utc": start.isoformat(), "clock": "gps"} if start else {"clock": "none"}
    return Telemetry(source, seconds, grid, meta=meta)


def test_x3_clock_reads_the_file_name_only_without_a_better_clock():
    tel = _tel("x.insv", None, 10)
    telemetry._x3_clock(tel, Path("VID_20260101_100000_00_001.insv"))
    assert tel.meta["start_local"] == "2026-01-01T10:00:00"
    assert tel.meta["clock"] == "filename"

    gps = _tel("y.insv", datetime(2026, 1, 1, 9, tzinfo=UTC), 10)
    telemetry._x3_clock(gps, Path("VID_20260101_100000_00_001.insv"))
    assert gps.meta["clock"] == "gps" and "start_local" not in gps.meta


def test_join_runs_joins_clips_across_short_gaps_and_splits_at_lifts_and_long_gaps():
    t0 = datetime(2026, 1, 1, 10, tzinfo=UTC)
    tels = [
        _tel("c.mp4", t0 + timedelta(hours=1), 120),  # an hour later: a new run
        _tel("a.mp4", t0, 120),
        _tel("b.mp4", t0 + timedelta(seconds=180), 120, lift_from=90),  # 60 s after a ends
        _tel("d.mp4", t0 + timedelta(seconds=310), 60),  # right after b, but b ends on a lift
    ]

    out = telemetry.join_runs(tels)

    assert [c["source"] for c in out["clips"]] == ["a.mp4", "b.mp4", "d.mp4", "c.mp4"]
    assert [r["clips"] for r in out["runs"]] == [["a.mp4", "b.mp4"], ["d.mp4"], ["c.mp4"]]
    assert out["runs"][0]["riding_s"] == pytest.approx(120 + 90)
    assert out["runs"][0]["top_speed_kmh"] == 40.0
    assert out["day"]["clips"] == 4 and out["day"]["runs"] == 3
    assert out["day"]["lift_s"] == pytest.approx(30)


def test_a_nested_gps_scale_reads_as_flat_and_an_unusable_one_skips_its_payload(caplog):
    rec = _gps_rec(46.5, 7.25, 1200.0, 3.0, 9100, 7200.0, 1.2, 3)
    nested = _gps_group(0, rec)
    nested["GPS9"]["Scale"] = [GPS_SCALE[:4], GPS_SCALE[4:]]
    want = telemetry._gps9([_gps_group(0, rec)])
    assert telemetry._gps9([nested]).tolist() == want.tolist()
    one = _gps_group(0, struct.pack(">lllllllHH", *range(10, 19)))
    one["GPS9"]["Scale"] = [[10]]  # one value scales every field
    assert telemetry._gps9([one])[0, 1:].tolist() == pytest.approx([x / 10 for x in range(10, 19)])
    one["GPS9"]["Scale"] = 10  # so does a bare number
    assert telemetry._gps9([one])[0, 1:].tolist() == pytest.approx([x / 10 for x in range(10, 19)])

    bad = [[[1, 2]], ["x"], [0] * 9, [True] * 9, "scale"]
    groups = [_gps_group(0, rec)]
    for k, scale in enumerate(bad, 1):
        g = _gps_group(k * 1_000_000, rec)
        g["GPS9"]["Scale"] = scale
        groups.append(g)
    with caplog.at_level("DEBUG", logger="cutpoint.telemetry"):
        rows = telemetry._gps9(groups)

    assert rows.tolist() == want.tolist()  # only the good payload is left
    debug = [r.getMessage() for r in caplog.records if r.levelname == "DEBUG"]
    assert len(debug) == len(bad) and all(m.startswith("GPS9: unusable Scale") for m in debug)
