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
    bins = np.arange(200) * DT
    speed = np.where(bins < 10.5, 30.0, 0.0 if stop else 30.0)
    vib = np.where(bins < 10.0, 0.3, 0.01 if stop else 0.3)
    if not gps:
        speed = np.full(200, np.nan)
    return t, am, grav, speed, vib


def test_an_impact_that_ends_in_a_stop_is_a_crash():
    [crash] = telemetry._crash_events(*_impact(), P)
    assert crash["type"] == "crash"
    assert crash["impact_t"] == pytest.approx(10.0, abs=0.03)
    assert crash["peak_g"] == 6.0
    assert crash["tilt_deg"] == pytest.approx(90, abs=1)
    assert crash["speed_before_kmh"] == 30.0 and crash["speed_after_kmh"] == 0.0
    assert crash["score"] == pytest.approx(
        0.3 * (6.0 - 3.5) / 4.5 + telemetry.STOP_SHARE + 0.3, abs=0.01
    )


def test_a_hard_bump_while_riding_on_is_not_a_crash():
    assert telemetry._crash_events(*_impact(stop=False), P) == []


def test_without_gps_a_crash_is_an_impact_then_stillness():
    [crash] = telemetry._crash_events(*_impact(gps=False), P)
    assert crash["speed_before_kmh"] is None and crash["speed_after_kmh"] is None
    assert telemetry._crash_events(*_impact(gps=False, stop=False), P) == []


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
    assert tel.meta["start_utc"] is None  # per project: sync_filename_clocks sets it

    gps = _tel("y.insv", datetime(2026, 1, 1, 9, tzinfo=UTC), 10)
    telemetry._x3_clock(gps, Path("VID_20260101_100000_00_001.insv"))
    assert gps.meta["clock"] == "gps" and "start_local" not in gps.meta


def _x3(name: str, local: str):
    tel = _tel(name, None, 60)
    tel.meta.update({"start_local": local, "start_utc": None, "clock": "filename"})
    return tel


def test_x3_file_name_clock_takes_the_zone_of_a_gopro_gps_clip():
    gopro = _tel("g.mp4", datetime(2026, 1, 1, 10, tzinfo=UTC), 60)
    gopro.meta["start_local"] = "2026-01-01T05:00:00"  # local = GPS UTC - 5 h
    x3 = _x3("x.insv", "2026-01-01T05:10:00")

    out = telemetry.join_runs([x3, gopro])

    assert x3.meta["start_utc"] == "2026-01-01T10:10:00+00:00"
    assert x3.meta["clock"] == "filename_gopro"
    assert [c["source"] for c in out["clips"]] == ["g.mp4", "x.insv"]
    assert "order" not in out["clips"][1]


def test_x3_file_name_clock_without_a_gopro_keeps_local_time_and_marks_the_order(caplog):
    caplog.set_level("INFO", logger="cutpoint.telemetry")
    cam = _tel("c.mp4", datetime(2026, 1, 1, 10, tzinfo=UTC), 60)
    cam.meta["clock"] = "camera"  # no GPS: no zone to learn
    x3 = _x3("x.insv", "2026-01-01T10:10:00")

    out = telemetry.join_runs([cam, x3])

    local = datetime(2026, 1, 1, 10, 10).astimezone(UTC).isoformat()
    assert x3.meta["start_utc"] == local and x3.meta["clock"] == "filename"
    assert out["clips"][1]["order"] == "unknown vs GPS clips" and "order" not in out["clips"][0]
    assert sum("order against the other clips is unknown" in r.message for r in caplog.records) == 1

    solo = telemetry.join_runs([_x3("y.insv", "2026-01-01T10:10:00")])  # only X3 clips: ordered
    assert "order" not in solo["clips"][0]

    caplog.clear()
    quiet = telemetry.join_runs([cam, _x3("z.insv", "2026-01-01T10:10:00")], quiet=True)
    assert quiet["clips"][1]["order"] == "unknown vs GPS clips"  # still marked
    assert not caplog.records


def test_gopro_zone_offset_is_rounded_to_15_minutes():
    # fractional GPS start, local clock cut to whole seconds: 7:59:59.571 is exactly 8 h
    gopro = _tel("g.mp4", datetime(2026, 1, 1, 21, 29, 29, 429000, tzinfo=UTC), 60)
    gopro.meta["start_local"] = "2026-01-02T05:29:29"
    x3 = _x3("x.insv", "2026-01-02T05:30:00")
    telemetry.join_runs([gopro, x3])
    assert x3.meta["start_utc"] == "2026-01-01T21:30:00+00:00"

    # a local time a few minutes off still rounds to the zone: 5:28 is +5:30
    gopro = _tel("g.mp4", datetime(2026, 1, 1, 10, tzinfo=UTC), 60)
    gopro.meta["start_local"] = "2026-01-01T15:28:00"
    x3 = _x3("x.insv", "2026-01-01T15:40:00")
    telemetry.join_runs([gopro, x3])
    assert x3.meta["start_utc"] == "2026-01-01T10:10:00+00:00"


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


# ------------------------------------------------------------------------- the grids (A586)
RIDE_S, STOP_S, LIFT_S, END_S = 40.0, 55.0, 110.0, 120.0
IMU_HZ = 100.0


def _gps_fix(tr: float) -> tuple[float, float, float, int]:
    """``(speed m/s, alt m, dop, fix)`` at ``tr``: a ride at 36 km/h with bad fixes, a stop,
    a lift climbing 1 m/s at 15 km/h, then riding on with a 6 s run of bad DOP."""
    if tr < RIDE_S:
        speed, alt, dop, fix = 10.0, 2000.0 - 0.2 * tr, 1.5, 3
        if 10 <= tr < 12:
            speed, fix = 0.0, 0  # no fix for 2 s: interpolated
        elif 20 <= tr < 21:
            speed = 300 / 3.6  # a speed spike: dropped
        elif 25 <= tr < 26:
            speed, fix = 0.0, 2  # a 2D fix: dropped
        elif 30 <= tr < 31:
            speed, dop = 0.0, 9.0  # a poor DOP: dropped
        return speed, alt, dop, fix
    if tr < STOP_S:
        return 0.0, 1992.0, 1.5, 3
    if tr < LIFT_S:
        return 15 / 3.6, 1992.0 + (tr - STOP_S), 1.5, 3
    return 20 / 3.6, 2047.0, 9.0 if 112 <= tr < 118 else 1.5, 3  # a 6 s gap: left out


def _ride_groups(gps: bool = True) -> list[dict]:
    groups: list[dict] = []
    if gps:
        for k in range(int(END_S)):
            recs = []
            for j in range(10):
                speed, alt, dop, fix = _gps_fix(k + j / 10)
                recs.append(_gps_rec(46.5, 7.25, alt, speed, 9100, 3600.0 + k, dop, fix))
            groups.append(_gps_group(k * 1_000_000, *recs))
    groups.append({"WindProcessing": {"TimestampUs": 50_000_000, "Data": [[1, 80]] * 10}})
    face = struct.pack(">BBHHHHHBB", 1, 90, 1, 0, 0, 32767, 32767, 0, 0)  # a quarter of the frame
    groups.append({"FACE": {"TimestampUs": 60_000_000, "0x46414345": _klv("FACE", 14, [face])}})
    return groups


def _compute_ride(monkeypatch, gps: bool = True) -> Telemetry:
    """:func:`telemetry._compute` of :func:`_ride_groups` and an IMU that rides upright with
    turns (yaw about gravity) and vibration, and is calm while stopped and on the lift."""
    t = np.arange(int(END_S * IMU_HZ)) / IMU_HZ
    riding = (t < RIDE_S) | (t >= LIFT_S)
    noise = np.where(riding, 4.0, 0.005)[:, None]
    acc = np.tile([0.0, telemetry.G, 0.0], (len(t), 1))
    acc += np.random.default_rng(5).normal(0, 1, acc.shape) * noise
    gyr = np.zeros_like(acc)
    gyr[:, 1] = np.where(t < RIDE_S, 60 * np.sin(np.pi * t), 0.0)  # one turn each way per 2 s
    parser = type("Parser", (), {"camera": "GoPro", "model": "HERO11 Black"})()
    groups = _ride_groups(gps)
    monkeypatch.setattr(telemetry, "parse_imu", lambda path: (parser, groups, t, acc, gyr))
    tel = telemetry._compute(Path("GX01.mp4"), END_S, P)
    assert tel is not None
    return tel


def _bins(a: float, b: float) -> slice:
    return slice(int(a / DT), int(b / DT))


def test_compute_drops_bad_fixes_fills_short_gaps_and_finds_the_stop_and_the_lift(monkeypatch):
    tel = _compute_ride(monkeypatch)
    g = tel.grid
    assert tel.n == int(END_S / DT) + 1 and all(len(v) == tel.n for v in g.values())
    # bad fixes are dropped and the 2 s without a fix is interpolated
    assert g["speed_kmh"][_bins(1, 39)].tolist() == pytest.approx([36.0] * 380, abs=0.5)
    assert g["gps_ok"][_bins(0, 112)].min() == 1.0
    assert tel.summary()["top_speed_kmh"] == pytest.approx(36.0, abs=0.5)  # not the spike
    # the 6 s gap stays a gap
    assert np.isnan(g["speed_kmh"][_bins(113, 117)]).all()
    assert g["gps_ok"][_bins(113, 117)].max() == 0.0
    assert g["alt_m"][_bins(60, 100)].tolist() == pytest.approx(
        (1992.0 + np.arange(60, 100, DT) - STOP_S).tolist(), abs=0.6
    )
    # stopped, then on the lift: still only while stopped
    assert g["still_mask"][_bins(41, 54)].min() == 1.0
    assert g["still_mask"][_bins(0, 39)].max() == 0.0
    assert g["lift_mask"][_bins(60, 108)].min() == 1.0
    assert g["lift_mask"][_bins(0, 50)].max() == 0.0
    assert g["still_mask"][_bins(60, 108)].max() == 0.0
    # carving: turns at speed, nothing while stopped
    assert g["carve"][_bins(2, 38)].mean() > 0.8
    assert g["carve"][_bins(42, 54)].max() == 0.0
    # slow streams: wind held for its payload, faces for 0.9 s
    assert g["wind"][_bins(50, 51)].tolist() == pytest.approx([0.8] * 10)
    assert g["wind"][_bins(0, 49.9)].max() == 0.0
    assert g["faces"][_bins(60, 61)].tolist() == pytest.approx([0.25] * 10, abs=0.001)
    assert g["faces"][int(61 / DT)] == 0.0
    assert tel.events == [] and tel.meta["clock"] == "gps"
    assert tel.meta["camera"] == "GoPro HERO11 Black" and tel.meta["gps_samples"] == 1200


def test_compute_without_gps_finds_stillness_from_the_vibration(monkeypatch):
    tel = _compute_ride(monkeypatch, gps=False)
    g = tel.grid
    assert np.isnan(g["speed_kmh"]).all() and g["gps_ok"].max() == 0.0
    assert g["lift_mask"].max() == 0.0  # no altitude: no lift
    assert g["vibration"][_bins(2, 38)].min() > 0.04
    assert g["still_mask"][_bins(41, 54)].min() == 1.0
    assert g["still_mask"][_bins(0, 39)].max() == 0.0
    assert tel.meta["clock"] == "none" and tel.meta["gps_samples"] == 0


def test_interp_gaps_fills_only_short_inner_gaps():
    nan = np.nan
    x = np.array([nan, 1.0, nan, nan, 4.0, nan, nan, nan, nan, 9.0, nan])
    got = telemetry._interp_gaps(x, 3)
    want = [nan, 1.0, 2.0, 3.0, 4.0, nan, nan, nan, nan, 9.0, nan]  # edges and long gaps stay
    assert np.array_equal(got, want, equal_nan=True)
    assert np.isnan(x[2])  # the input is not changed
    one = np.array([nan, 5.0, nan])
    assert np.array_equal(telemetry._interp_gaps(one, 3), one, equal_nan=True)


def test_rolling_mean_and_std_skip_nans():
    x = np.array([1.0, 3.0, np.nan, 5.0, 7.0])
    assert telemetry._rolling(x, 3).tolist() == pytest.approx([2.0, 2.0, 4.0, 6.0, 6.0])
    std = telemetry._rolling(x, 3, "std")
    assert std.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.isnan(telemetry._rolling(np.full(4, np.nan), 3)).all()
    assert telemetry._rolling(x, 0).tolist()[:2] == [1.0, 3.0]  # k < 1: one bin


# ------------------------------------------------------------------ hold, median, cache
NAN = np.nan


@pytest.mark.parametrize(
    ("x", "hold", "want"),
    [
        ([1, NAN, NAN, NAN, 5], 2, [1, 1, 1, NAN, 5]),  # held for 2 bins, then the gap shows
        ([1, NAN, 3], 0, [1, NAN, 3]),  # no hold: unchanged
        ([NAN, NAN, 2, NAN], 1, [NAN, NAN, 2, 2]),  # nothing to hold before the first value
        ([1, NAN, NAN, 4, NAN, NAN, NAN], 2, [1, 1, 1, 4, 4, 4, NAN]),  # age resets at a value
        ([], 3, []),
    ],
)
def test_ffill_hold_repeats_a_value_for_at_most_hold_bins(x, hold, want):
    # a hold limit off by one, or an age that never resets, changes a bin
    src = np.array(x, float)
    got = telemetry._ffill_hold(src, hold)
    np.testing.assert_array_equal(got, np.array(want, float))
    np.testing.assert_array_equal(src, np.array(x, float))  # the input is not modified


@pytest.mark.parametrize(
    ("x", "k", "want"),
    [
        ([1, 1, 9, 1, 1], 3, [1, 1, 1, 1, 1]),  # a spike is removed
        ([1, 2, 3, 4, 5], 3, [1, 2, 3, 4, 5]),  # the edges are padded with the edge value
        ([1, 2, 3, 4], 2, [1, 1.5, 2.5, 3.5]),  # even window: pad k//2 before, the rest after
        ([1, NAN, 3, 4, 5], 3, [1, 2, 3.5, 4, 5]),  # NaN is ignored, not propagated
        ([NAN, NAN, NAN], 3, [NAN, NAN, NAN]),  # an all-NaN window stays NaN
    ],
)
def test_rolling_median_is_centered_edge_padded_and_nan_aware(x, k, want):
    got = telemetry._rolling_median(np.array(x, float), k)
    np.testing.assert_allclose(got, np.array(want, float), equal_nan=True)


@pytest.mark.parametrize(("x", "k"), [([1, 5, 2], 1), ([1, 5, 2], 0), ([1, 5], 3), ([], 3)])
def test_rolling_median_returns_a_copy_when_the_window_does_not_fit(x, k):
    src = np.array(x, float)
    got = telemetry._rolling_median(src, k)
    np.testing.assert_array_equal(got, src)
    assert got is not src


def test_extract_reads_a_written_track_back_from_the_cache(home, tmp_path, monkeypatch):
    src = tmp_path / "ride.mp4"
    src.write_bytes(b"x")
    monkeypatch.setattr(telemetry, "imu_read_path", lambda s: (src, None, src, False))
    monkeypatch.setattr("cutpoint.crash.cache_imu_grids", lambda *a: None)
    grid = {
        "speed_kmh": np.array([1.0, 2.5, NAN, 4.0]),
        "gps_ok": np.array([1.0, 1.0, 0.0, 1.0]),
        "lift_mask": np.array([0.0, 1.0, 1.0, 0.0]),
        "still_mask": np.zeros(4),
    }
    events = [{"type": "airtime", "t": 1.5, "end": 2.0, "confidence": "high"}]
    calls: list[Path] = []

    def parse(path, *a):
        calls.append(path)
        return "parsed"

    def compute(path, duration, params, parsed):
        assert parsed == "parsed"
        return Telemetry(
            source="", duration=duration, grid=dict(grid), events=events, meta={"camera": "X"}
        )

    monkeypatch.setattr(telemetry, "parse_imu", parse)
    monkeypatch.setattr(telemetry, "_compute", compute)
    first = telemetry.extract(src, 0.4)
    assert first is not None and len(calls) == 1

    # second call: no parse, no compute; the same track from the files
    monkeypatch.setattr(telemetry, "parse_imu", lambda *a: pytest.fail("parsed again"))
    monkeypatch.setattr(telemetry, "_compute", lambda *a: pytest.fail("computed again"))
    again = telemetry.extract(src, 0.4)
    assert again is not None
    assert again.source == str(src) and again.duration == 0.4
    assert again.events == events and again.meta["camera"] == "X"
    assert set(again.grid) == set(grid)
    for k, v in grid.items():
        np.testing.assert_array_equal(again.grid[k], v.astype(np.float32))  # NaN kept
    assert again.n == 4

    # other thresholds are another cache entry: parsed again
    seen = []
    monkeypatch.setattr(telemetry, "parse_imu", lambda *a: seen.append(1))
    assert telemetry.extract(src, 0.4, TelemetryParams(max_dop=1.0)) is None
    assert seen == [1]
