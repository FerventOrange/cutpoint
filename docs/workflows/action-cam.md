# Action-cam footage

This is an optional module (extra `ski`; the full feature set also needs the `events` extra and
`cutpoint setup`). It uses telemetry in GoPro HERO 9-13 files (accelerometer, gyro, GPS) to
score moments by what happened: speed, airtime, carving, crashes. Lifts and stops are masked
unless friends are in view. The style name is `ski`, but the signals apply to any riding or
action footage that carries telemetry.

```powershell
cutpoint stage D:\Videos\trip                        # telemetry reading uses whole files: stage slow storage first
cutpoint analyze D:\Videos\trip --style ski --out trip
cutpoint events trip                                 # sound events, voice, visual layers, crash check, model tags
cutpoint clips trip --max-total 150
cutpoint assemble projects/trip/clips --loudnorm
```

`analyze --style ski` skips scene detection. As with every style, the Ollama frame rating runs
only with `--vision`. It writes `telemetry.json` (day summary, runs, per-source events). Clips are ordered by their
start time (GPS time, else the camera clock) and the day is split into runs at lift segments
and gaps over 4 minutes.

Footage without telemetry (phones, other cameras) is analysed from audio, speech and model
features only: no speed, airtime or masking.

## Crashes

```powershell
cutpoint crashes trip          # crashes.json, crash_check\*.jpg, then re-runs analyze
cutpoint clips trip --max-total 480
```

Candidates come from full-rate IMU shocks plus spoken crash cues. Each gets telemetry, visual,
dense-frame, audio and transcript sub-scores and a confidence. Crashes at or above the style's
`must_include_crash` threshold (0.6 for `ski`) are always exported with their lead-in,
aftermath and a reaction clip. `analyze --must-include-crash CONF` overrides it; 0 disables.
Check `crash_check\` by eye: weights and thresholds are hand-set, and the detector misses
tumbles when the camera points at the rider's own skis and can be fooled by shadows.

## Activity profiles

What a crash looks and sounds like depends on the sport: snow on the lens and "yard sale" when
skiing, a bike on its side and "over the bars" on a trail. `crashes` takes its visual prompts and
crash words from the project's activity profile, chosen or built without manual editing:

1. `--activity NAME` picks a profile by name.
2. Otherwise `activity.json` from an earlier run is reused.
3. Otherwise the cached keyframes are compared with CLIP to every library profile. A clear
   winner (at least half of the score and 0.2 ahead of the next) is used.
4. Otherwise the local vision model describes a few frames (activity, setting, camera mount,
   surface, hazards) and the local LLM writes a new profile from that description, with the
   built-in profiles as examples. It is checked and capped, then saved to
   `<CUTPOINT_HOME>\profiles\`, so the next video of that kind matches it in step 3.
5. If no model answers, the nearest library profile or `generic` is used (`source: fallback`),
   and the next run tries again.

`--context "..."` passes the user's own words ("enduro race in the rain, mud on the lens"). The
LLM adds crash captions, normal-riding captions and slang on top of the chosen profile; words
the profile already has keep their weights, and generated words are capped (0.6 for single
words and prefixes, 0.9 for phrases). A different `--context` rebuilds the profile.

```powershell
cutpoint profile trip --json                                  # what was chosen, and why
cutpoint profile bike-park --context "downhill mtb, chest mount, muddy"
cutpoint crashes bike-park                                    # reuses bike-park\activity.json
cutpoint crashes trip --activity ski
```

Built-in profiles: `ski`, `snowboard`, `mtb`, `moto`, `skate`, `surf`, `hike` and `generic`
(`src/cutpoint/presets/activities`). All of them inherit a shared base: dirt, mud, water or
debris on the lens, a tumbling or upside-down view, a person lying on the ground, and generic
words such as "you okay" and "fell". `ski` replaces the base captions with snowy versions.
Profiles also hold highlight and low-information captions for other layers (not used yet).

New profiles need Ollama with a Qwen model (`CUTPOINT_VISION_MODEL`, default `qwen3-vl:8b`).
Text-only requests to a model whose name starts with `qwen` skip its reasoning step; other
models are asked through the normal chat endpoint. The model is unloaded after use, so the GPU
is free for the CLIP steps.

## A varied mix

```powershell
cutpoint mix trip                      # mix_segments.json, mix_report.json, mix_sheet.jpg
cutpoint clips trip --mix              # clips_mix\ in arc order
cutpoint assemble projects/trip/clips_mix --loudnorm
```

`mix` picks runs, jumps, crashes with reactions, lift rides, views and social moments under
per-category quotas (`--quotas run=.35,...`, `--target` seconds). The arc order opens with a
lift or scenery shot, follows the day, and ends on the best crash, run or jump. `mix_report.json`
holds content checks: at least four categories, none over 40% of the length, no near-duplicates,
a scenic opener, and 180-300 s total. `--exclude-file` lists source spans to leave out.

## Learning from your own edit

If you already cut a video from the same footage, Cutpoint can fit scoring weights to it:

```powershell
cutpoint refmatch trip D:\Videos\trip-cut.mp4 --name my_style
cutpoint analyze D:\Videos\trip --style my_style --out trip      # re-rank with the learned style
```

Use `cutpoint refmatch ... --no-fit` on several projects, then `cutpoint fit proj1 proj2
--name my_style` to fit one style on all of them. Learned styles are JSON in
`<CUTPOINT_HOME>\styles\`. The fit is only as good as the edit you teach it: with one or two
edits, treat the result as a starting point.

## Limits

- Full telemetry needs GoPro HERO 9-13. The Insta360 X3 gives IMU only (no GPS, speed or lifts).
- Short clips often have no GPS fix; speed then comes from the wind meter.
- The accelerometer saturates near 8 g on GoPro, and jumps under 0.3 s are unreliable.
- Wind masks speech above about 20 km/h, so talk while riding fast is mostly lost.
- The person detector counts strangers; a lift-line crowd gives high "people" scores.
