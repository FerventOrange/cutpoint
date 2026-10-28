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
  anything faster than a 45 degrees-per-second pan, or any change to or from the selfie, is a
  hard cut with a fixed cost. Shots shorter than 2.5 s are merged away.
- Within a shot, yaw is smoothed and limited in speed and acceleration. Pitch follows the framed
  person, clamped to -25..+5 degrees. Field of view is wide by default and fitted to a tracked
  friend's size.
- The wearer may be in frame when arms, poles or skis enter from the bottom edge or cover at
  most a fifth of the frame. A larger box elsewhere is treated as a person too close and framed
  out.
- The moment's kind shifts the weights: laughter and speech favour the selfie, people and
  groups favour friends, beauty favours the view, speed and airtime favour forward.
- Paths are planned 10 s beyond each side of the segment so padding stays on a planned path.

Override tuning values with `pano.director` in `config.json` or `cutpoint 360 director --param
KEY=VALUE`.

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
