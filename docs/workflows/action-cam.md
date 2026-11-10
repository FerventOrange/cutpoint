# Action-cam footage

This module is part of every install; the full feature set needs the `events` extra, which
Cutpoint lite leaves out, and `cutpoint setup`. It uses telemetry in GoPro HERO 9-13 files (accelerometer, gyro, GPS) to
score moments by what happened: speed, airtime, carving, crashes. Lifts and stops are masked
even with friends in view. The style is `action-cam`; the signals apply to any riding or action footage that
carries telemetry.

```powershell
cutpoint stage D:\Videos\trip                        # telemetry reading uses whole files: stage slow storage first
cutpoint analyze D:\Videos\trip --style action-cam --out trip
cutpoint events trip                                 # optional: model tags (VLM layer, needs the Ollama vision model) or forced layers
cutpoint clips trip --max-total 150
cutpoint assemble projects/trip/clips --loudnorm
```

`analyze --style action-cam`:

- skips scene detection;
- runs the reaction (sound events, voice), keyframe (visual, only with a CUDA GPU) and crash
  steps when the `events` extra and models are installed, writing `events.json` and
  `crashes.json` (`--skip` leaves steps out); the Ollama frame rating runs only with `--vision`;
- writes `telemetry.json` (day summary, runs, per-source events);
- writes these files, `crash_check\` and `action_cam_features\` only when it completes: a cancelled
  run leaves the previous ones as they were ([analyze](../commands.md#analyze));
- orders clips by start time (GPS time, else the camera clock) and splits the day into runs at
  lift segments and gaps over 4 minutes.

Footage without telemetry (phones, other cameras) is analysed from audio, speech and model
features only: no speed, airtime or masking.

## Crashes

```powershell
cutpoint crashes trip          # re-run on its own (analyze already ran it): crashes.json, crash_check\*.jpg, then re-runs analyze
cutpoint clips trip --max-total 480
```

Candidates come from full-rate IMU shocks plus spoken crash cues. Each gets telemetry, visual,
dense-frame, audio and transcript sub-scores and a confidence. Crashes at or above the style's
`must_include_crash` threshold (0.6, as for every other style) are always exported with their
lead-in, aftermath and a reaction clip. `analyze --must-include-crash CONF` overrides it; 0 forces none.
Check `crash_check\` by eye: weights and thresholds are hand-set, and the detector misses
tumbles when the camera points at the rider's own gear and can be fooled by shadows.

## Activity profiles

What a crash looks and sounds like depends on the sport: snow on the lens and a lost ski, a
bike on its side, a board in the water. `crashes` takes its visual prompts and crash words from
the project's activity profile. Cutpoint ships only `base` (the shared floor) and `generic`; it
makes no sport profiles for you. Every other profile is in your library
(`<CUTPOINT_HOME>\profiles\`): discovered from your footage, or written by you or your AI agent.
Action-cam footage goes through discovery like any other:

1. `--activity NAME` picks `generic` or a profile of your library (`cutpoint profile list`). An
   unknown name is an error.
2. Otherwise `activity.json` from an earlier run is reused.
3. Otherwise the cached keyframes are compared with CLIP to every library profile and to an
   open vocabulary of activities. A clear winner among the profiles (at least half of the
   score and 0.2 ahead of the next) is used.
4. Otherwise the local vision model describes a few frames, chosen to differ as much as
   possible (activity, setting, camera mount, surface, hazards), and the local LLM writes a new
   profile from that description, with `base` and `generic` as examples. Its name says the
   activity alone (crash words are dropped from it and from its recognition captions, so
   `ski-crash` is `ski`), crash captions that fit more than 10% of the footage's keyframes are
   dropped, and the VLM prompts keep the generic wording. It is checked and capped, then saved to
   `<CUTPOINT_HOME>\profiles\` with `"source": "discovered"` (a library file with the same
   content is reused), so the next video of that kind matches it in step 3. The result is also
   cached per footage: a new project over the same files, with the same model, `--context` and
   example profiles, gets the same profile without asking the model again. A discovery the
   model answered unusably or that timed out is remembered too: that footage is not sent to the
   model again until `--refresh-profile`; a connection error is not, so the next run asks again. In `analyze` and
   `edit` this runs whenever a local Ollama answers, without `--vision`, and is quiet (one INFO
   line when it is skipped).
5. If no model answers, the nearest library profile is used, else `generic` (`source:
   fallback`, with CLIP's three best vocabulary terms in its `description`), and the next run
   tries again. An Ollama on another machine is not asked until the Vision consent for it was given
   in the Models tab of `cutpoint ui` (WARNING `vision_consent_needed`; INFO in `analyze` and
   `edit`).

`--context "..."` passes the user's own words ("enduro race in the rain, mud on the lens"). The
LLM adds crash captions, normal-riding captions and slang on top of the chosen profile; words
the profile already has keep their weights, and generated words are capped (0.6 for single
words and prefixes, 0.75 for phrases, below the 0.8 a spoken cue needs to be a crash candidate
on its own). A different `--context` rebuilds the profile once;
`--refresh-profile` ignores `activity.json` and the profile (or failed discovery) cached for the footage and writes
it again, with `--context` or while the footage matches no library profile (otherwise delete
that profile first: `cutpoint profile delete NAME`).

```powershell
cutpoint profile trip --json                                  # what was chosen, and why
cutpoint profile bike-park --context "downhill mtb, chest mount, muddy"
cutpoint crashes bike-park                                    # reuses bike-park\activity.json
cutpoint profile list                                         # discovered and hand-made ones
cutpoint crashes trip --activity my-trails                    # a profile from the list
```

To make a profile yourself, or have your AI agent make one, start from the format of
`cutpoint profile show generic`, then check and save it:

```powershell
cutpoint profile validate my-trails.json
cutpoint profile create my-trails --from my-trails.json
```

`profile show`, `validate` and `delete` work on a name; `list` marks each profile as built-in,
discovered or hand. A project that already has an `activity.json` keeps its recorded profile.

Every profile inherits the shared base: dirt, mud, water or debris on the lens, a tumbling or
upside-down view, a person lying on the ground, and generic words such as "you okay" and
"fell". A profile can replace base captions with its own (`exclude_base`). A profile without
normal-footage captions (`incident.neg`) takes those of `generic`. `crashes` never sees less
of a crash than `--activity generic` does: per frame, the visual and dense checks use the
larger of the profile's crash score and that of `generic`.
Profiles also hold the captions of the other visual layers: `highlight` (scenic or striking
views), `group` and `person` (other people versus the filmer), `selfie`, `occluder` (a hand or
debris on the lens), `low_info` (frames with nothing to see), `scene` and `lift`, plus the
wording of the 360 vision check (`vl`). The action-cam ranking, the lift rides of `crashes`
and `mix`, the visual layer's contact sheets and the 360 views, QA, vision check and director
read them from the project's `activity.json` (the crash step writes it; until then they use
the `generic` captions). `lift` captions are optional: discovery fills them when the footage has
lift or shuttle rides, and chairlift detection is off for a profile without them.

Profiles also have a `vlm` section: the wording of the prompts of the action-cam VLM layer of
`cutpoint events` (the window prompt) and of the `--vision` frame rating, so neither names a
sport the footage is not. Its fields are `recording` (what the window shows), `video`, `audio`,
`describe`, `hype` (the 0-10 scale), `edit` (the kind of edit), `idle`, then for the rating
`rate_edit` (its kind of edit), `rate_high` and `rate_low` (what rates high and low) and
`retry_high` and `retry_low` (the shorter lists of its retry prompt), each a single line of at
most 300 characters, and `tags`, a
description for each window tag (at most 100 characters). The tag keys (`jump_air`,
`spray`, `transit` and the others) are fixed: they are the names stored in
`events.json` and read by the scorer (the older keys `powder_spray`, `fast_groomer`, `chairlift` and `lodge` are ignored: run `cutpoint events` again on an old project), so a profile can only
reword their descriptions, not rename, add or drop a tag. Only a profile made with
`cutpoint profile create` can reword them: discovered and `--context` profiles keep the generic
wording of `base.json`, as does a profile without its own wording. A value that is missing, not a string, empty or over its cap is replaced by the
`base.json` text, field by field. The `vlm` section of an `activity.json` written before
profiles had one is filled from the profile's library entry, else `base.json`.

The first run after this wording was added describes the windows and rates the frames again
for action-cam projects (the wording is part of the cache keys). The 360 plans are recomputed
once too.

Discovery needs Ollama with a Qwen model (`CUTPOINT_VISION_MODEL`, default `qwen3.5:9b`).
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
a scenic opener, and 180-300 s total. Views and lift shots are 5-7 s long. `--exclude-file` lists source spans to leave out.

## Learning from your own edit

If you already cut a video from the same footage, the development script
`scripts/refmatch.py` (from a clone; it was `cutpoint refmatch` and `cutpoint fit`) fits
scoring weights to it:

```powershell
uv run python scripts/refmatch.py match trip D:\Videos\trip-cut.mp4 --name my_style
cutpoint analyze D:\Videos\trip --style my_style --out trip      # re-rank with the learned style
```

To write or check a style by hand, see `cutpoint style` (`cutpoint style --help`).

Use `scripts/refmatch.py match ... --no-fit` on several projects, then
`scripts/refmatch.py fit proj1 proj2 --name my_style` to fit one style on all of them. Learned styles are JSON in
`<CUTPOINT_HOME>\styles\`. The fit is only as good as the edit you teach it: with one or two
edits, treat the result as a starting point.

## Limits

- Full telemetry needs GoPro HERO 9-13. The Insta360 X3 gives IMU only (no GPS, speed or lifts).
- Short clips often have no GPS fix; speed then comes from the wind meter.
- The accelerometer saturates near 8 g on GoPro, and jumps under 0.3 s are unreliable.
- Wind masks speech above about 20 km/h, so talk while riding fast is mostly lost.
- The person detector counts strangers; a lift-line crowd gives high "people" scores.
