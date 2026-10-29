# How the 360 director works

Design notes for the `cutpoint 360` module. You do not need this to use it; it explains why a
clip is framed the way it is and which settings change that. Source: `src/cutpoint/pano/`.

## Horizon levelling

`geometry.py` fuses the camera's IMU (accelerometer and gyro at about 1 kHz) with a
complementary filter at 200 Hz. The gyro propagates the "up" vector and the accelerometer pulls
it back, weighted down when the acceleration is far from 1 g. In the display frame, yaw 0 is the
camera's forward direction, smoothed over 1.5 s so the view does not swing with every turn.
Dual-fisheye input and SDK equirects stitched without FlowState are levelled this way. Studio
exports are assumed to be levelled already.

`pano.level_strength` (default 0.7) sets how much of the camera's own tilt the view follows:
1 keeps the horizon level, 0 lets it lean with the camera in a turn. Shake is removed either
way. The lean followed is capped at `pano.max_follow_deg` (default 30 deg from the clip's
average attitude), so a fall or the camera being handled does not tip the view over.

## View layer

`views.py` decodes the `.lrv` proxy at about 0.8 frames per second and resamples it into eight
levelled horizon views (90 degrees wide, every 45 degrees of yaw) plus a downward "selfie" view at
the wearer. CLIP embeddings give beauty and group scores per view; a person detector runs on a
levelled equirect. People within 50 degrees of the wearer axis are treated as the wearer. The
results are cached per clip, and the standard visual-layer cache is written too, so `events`
does not decode again.

## Director

`director.py` plans a virtual camera path (yaw, pitch, field of view) for each chosen moment,
in the spirit of Pano2Vid and AutoCam.

- Every 0.5 s, each of 36 yaw bins (10 degrees) is scored from nearby people (by closeness,
  wearer excluded), the group prompt, beauty, and a bias toward the travel direction. A
  separate selfie state looks down at the wearer and is scored from the wearer talking or
  laughing.
- Dynamic programming chooses the path. Staying put is free, moving costs per degree, and
  anything faster than a 35 degrees-per-second pan (`max_speed`), or any change to or from the
  selfie, is a hard cut with a fixed cost. Shots shorter than 3 s (`min_hold_s`) are merged
  away.
- Within a shot, yaw is smoothed and limited in speed and acceleration. Pitch follows the framed
  person, clamped to -25..+5 degrees. Field of view is wide by default and fitted to a tracked
  friend's size.
- The wearer may be in frame when arms, poles or skis enter from the bottom edge or cover at
  most a fifth of the frame. A larger box elsewhere is treated as a person too close and framed
  out.
- Other people at least 15 degrees tall (`near_min_h_deg`, detected with a score of at least
  `near_min_score`) are framed whole or kept out, never cut by the frame edge
  (`edge_rule` `include_exclude`, the default). Where one would be cut, the director widens the
  view or shifts its yaw, whichever costs less: a shift costs `w_shift` per half view width, at
  most `include_max_shift` of it; widening costs `w_widen` for the whole way to the widest view,
  and keeps the subject at least `include_min_subject_h` of the frame tall (or as tall as it
  was). The view is never narrowed. A person framed whole keeps `include_margin` of the frame
  off the sides; one kept out stays `exclude_clear_deg` outside it. A stretch keeps one choice
  (in, or out to the same side), and stretches closer than `include_bridge_s` are fixed as one.
  The check is the one `360 qa --paths` makes (the subject and the largest person count at the
  edge too) at every 0.5 s key and between keys, and a fix is kept only when it lowers the
  edge cuts that check counts (then the other bad steps). Plain widening (as far as the
  subject allows) is tried too, and the one that leaves fewer edge cuts is kept; edge cuts
  left after that are widened away where the subject allows. Where no shift or widening
  frames everyone well, it settles for one without an edge cut the check counts, then tries
  a tighter side margin, and else cuts away: only where that removes at least
  `cut_min_gain_s` (1 s) per cut it adds (two inside a shot, one at its end) of edge cuts
  the check counts or of steps where the subject is out of view, never within
  `cut_min_gap_s` (4 s) of another cut (it joins it instead), and to a view without the
  wearer where one is as good. When the view held before the cut is itself cut at the edge,
  the cut-away starts one key earlier. `edge_rule` `legacy` keeps the earlier rule: widen, then reframe, then
  cut away, inside the moment only.
- The fix covers the whole planned span, not only the moment, so padding and tails added later
  are clean too. Keys fall on the 0.5 s grid of the moment's start. The path records that span
  as `fixed`; `extend` reuses a recorded path when that span covers the new range, and an older
  (version 4) path only when its moment does, else it plans the path anew with a WARNING.
- The moment's kind shifts the weights: laughter and speech favour the selfie, people and
  groups favour friends, beauty favours the view, speed and airtime favour forward.
- Paths are planned 10 s beyond each side of the segment so padding stays on a planned path.

Override tuning values with `pano.director` in `config.json` or `cutpoint 360 director --param
KEY=VALUE`.

`360 analyze` already plans each candidate's path with the settings `clips` uses and caches it
in `<project>/pano_candidates/` (`candidates.py`). The cache key is the source, the exact span,
16:9 or 9:16, the moment kind, the resulting settings, the path version, a hash of the source
of the modules that plan and score the path (`candidates.CODE_MODULES`: director, views,
geometry, the QA, voice, visual scoring) and the identity (name, size, modification time) of
the view, voice and orientation caches the director reads, so a path `clips` reuses is the one
it would plan. A changed span, setting, layer or planning code plans it anew.

## Renderer

`render.py` uses the ffmpeg `v360` filter driven by a `sendcmd` file. Runtime rotation commands
are incremental in ffmpeg, so the renderer tracks the rotation it has sent and sends only deltas.
By default one stabilised update goes out per frame (`pano.cmd_every`, default 1); 2 renders
faster but leaves shake on every other frame. Decoding uses the GPU decoder found here (NVDEC, or VAAPI for 8-bit H.264 on an AMD GPU) with a CPU fallback.

## Quality checks

`qa.py` (`cutpoint 360 qa`) samples rendered clips at a few frames per second and flags
occluded views, people too close, subjects cut by the frame edge, low-information and dark
frames, and horizon tilt, then measures pans and cuts from the path. A clip passes when each
flag stays under its limit; the limits are defaults in `pano.qa.QAThresholds`. `repair.py`
(`cutpoint 360 repair`) turns the flags into exclusions for the next `mix`.

Each cached candidate path also gets `diagnostics` (`candidates.py`; fields and flags in
[360 video](360.md#camera-paths-of-the-candidates)) from the path, its geometric QA, the
director's score grid and the IMU: time at the sky or above the horizon, peak pan rate, hard
cuts per minute, time with the wearer in frame and how often the view misses a clearly better
direction. A sky view is one where the horizon sits at or below the lower third line of the
frame, so the threshold follows the field of view. The flag thresholds are provisional.
