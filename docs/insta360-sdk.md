# Bring your own Insta360 SDK (optional, for 360 footage)

This is only needed for `.insv` footage from an Insta360 X3 and the `cutpoint 360` module. Everything else in Cutpoint works without it.

## Why it is needed

`cutpoint 360 stitch` uses Insta360's Media SDK to stitch the two lens files of a clip into one equirectangular video. That gives a properly blended seam, which the built-in `dfisheye` fallback (the raw dual fisheye fed to ffmpeg `v360`) cannot do: it is preview quality and shows a hard seam along the horizon. Horizon levelling and framing come from Cutpoint's own IMU track for both sources.

Cutpoint never ships or downloads the SDK. Each user applies for their own copy. It is free, but it is distributed under Insta360's own terms, which you accept when you apply. Read them before use.

## How to apply

1. Go to <https://www.insta360.com/sdk/apply> (the link given in the public `Insta360Develop/Desktop-MediaSDK-Cpp` README) and apply for the Media SDK.
2. Approval and the download link arrive by email.

## What to download

- Windows: the Windows x64 package that contains the Media SDK (a `MediaSDK-<version>-<date>-win64` folder with `bin\MediaSDKTest.exe`, `bin\MediaSDK.dll` and `bin\models\`).
- Linux: the Media SDK tarball, which holds a `MediaSDK-<version>-linux-amd64.deb`. Installing it with `sudo dpkg -i` puts everything under `/opt/MediaSDK-<version>-linux/`.

The Camera SDK (camera connection, live control) is not needed.

## Requirements

- Windows x64, or Linux (Insta360 documents Ubuntu 22.04 x86_64; WSL is not supported).
- An NVIDIA GPU with a recent driver. Insta360 documents driver 470 or newer as the minimum; keep it current. Without working hardware encoding, stitching is much slower (see Troubleshooting).

## Where to put it

Extract the package under `<CUTPOINT_HOME>\sdk\insta360\` (default `%LOCALAPPDATA%\cutpoint\sdk\insta360\`; `cutpoint paths` prints `CUTPOINT_HOME`). Cutpoint searches that folder recursively for `MediaSDKTest.exe` and uses the newest version it finds, so no configuration is needed. On Linux it also searches `/opt/MediaSDK-*-linux/bin/`, where the `.deb` installs.

To keep the SDK somewhere else, point Cutpoint at the executable with either:

- `"pano": {"sdk_exe": "<path to MediaSDKTest.exe>"}` in `config.json`, or
- the `CUTPOINT_INSTA360_SDK` environment variable (wins over `pano.sdk_exe`).

Keep the `models\` folder next to the executable (`bin\models\`). The SDK looks there by default, and the AI stitch type needs it.

## Verify

```powershell
cutpoint 360 status                                  # no project needed
cutpoint 360 stitch <project> --dry-run              # prints the SDK commands, stitches nothing
```

`status` shows the executable it found, where it came from (`CUTPOINT_INSTA360_SDK`, `pano.sdk_exe` or auto-detected), the version, and whether it starts (Cutpoint runs it with `-help`). `--dry-run` also works without the SDK, so run `status` first.

## Configuration

All keys live in the `pano` section of `config.json`; every one is optional. See `config.example.json`.

| Key | Meaning |
|---|---|
| `sdk_exe` | Path of `MediaSDKTest.exe` (`MediaSDKTest` on Linux). `CUTPOINT_INSTA360_SDK` overrides it. Default: auto-detect under `<CUTPOINT_HOME>\sdk\insta360\` |
| `stitched_dir` | Where stitched equirects go, as `<VID stem>_NNN.equirect.mp4`. Default `<CUTPOINT_HOME>\360-stitched` |
| `stitch_args` | SDK arguments placed after `-inputs ... -output ...`; replaces the defaults. Passed through unchanged apart from the `-stitch_type` and `-output_size` checks below, both made before the run. Default: `-stitch_type optflow -output_size 5760x2880 -enable_h265_encoder -bitrate 100000000 -enable_stitchfusion` |
| `stitch_timeout_s` | Kill one clip's stitch after this many seconds (positive number). Default: no limit |

Notes:

- **`stitched_dir` size.** Stitched output is about 50 GB per hour of 5.7K footage (Cutpoint estimates 54 GB per hour when checking free space and also requires 20 GB spare). Prefer a non-system drive. When `stitched_dir` is not set and the default lands on the system drive, Cutpoint logs a warning with the expected size and carries on.
- **`stitch_args`.** The parameters are Insta360's. See the parameter table under "Using the Sample Programs" in the public [Media SDK API Reference](https://github.com/Insta360Develop/Insta360-Developer_Docs/blob/main/docs/en/sdk/x-ace-go/desktop/media.md). Example: `["-stitch_type", "aistitch", "-output_size", "5760x2880", "-enable_h265_encoder", "-bitrate", "100000000"]`.
- **Allowed stitch types.** `optflow`, `dynamicstitch` or `aistitch`. Cutpoint accepts only the stitch types listed in Insta360's public docs and rejects any other value (including the old `template`) before the run.
- **Output size.** `-output_size` must be 2:1 (for example `5760x2880`). Any other value is rejected before the run; the finished file is also checked for 2:1.
- **FlowState is not used.** Cutpoint levels and frames stitched equirects with its own IMU track, as for `dfisheye`, so do not add `-enable_flowstate` to `stitch_args`. A stitch made with it, or one without a `<stem>.equirect.json` sidecar (every stitch made before FlowState was dropped from the defaults), is rendered as it is, and its framing drifts as the camera's heading changes. Cutpoint logs a warning for such files; restitch them with `cutpoint 360 stitch <project> --clips NNN --force` (the warning names the clip). Insta360 Studio exports get the same warning.
- **Encoder.** H.264 hardware encoding is capped at 4096 px, so keep `-enable_h265_encoder` for 5.7K output.

## What `cutpoint 360 stitch` does

```powershell
cutpoint 360 stitch <project> [--clips NNN ... | --from-clips DIR] [--top N] [--max-total S] [--force] [--dry-run]
```

- Cutpoint judges success from the SDK's progress and error output plus other status markers it prints at runtime (including any requested `-enable_*` feature reported as failed or off), and verifies the output file with ffprobe.
- The full stdout and stderr of every run go to `<stitched_dir>\logs\<stem>.log`.
- Output is written to a `.partial.mp4` file first. It is deleted on failure, on timeout and on Ctrl-C, so a half-written file is never taken for a result. A finished file is checked for 2:1 and the frame count of the clip, then renamed.
- Each finished file gets a sidecar `<stem>.equirect.json` with the stitch arguments, the SDK version (from its folder name) and a schema version. Rendering reads it to decide whether the file needs Cutpoint's IMU levelling.
- Nothing is trimmed: the SDK cannot cut, so whole clips are stitched. Only clips that hold chosen moments are processed (`--clips`, the clips in a clips folder's `manifest.json`, or the clips behind what `cutpoint clips` would select).
- Existing outputs are skipped unless `--force`.

After stitching, `cutpoint clips` and `cutpoint extend` pick the stitched file automatically with `--pano-source auto` (or `sdk`).

## Never commit the SDK

The SDK is under NDA. The repository `.gitignore` blocks the common SDK paths (`sdk/`, `**/insta360/`, `MediaSDKTest*`, and the Camera SDK and metadata library names), but that is a safety net only. Keep the SDK outside the repository, preferably in `<CUTPOINT_HOME>\sdk\insta360\`, and do not paste SDK output or documentation into issues, pull requests or docs.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `Insta360 Media SDK not found` | Run `cutpoint 360 status`. Extract the package under `<CUTPOINT_HOME>\sdk\insta360\`, or set `pano.sdk_exe` / `CUTPOINT_INSTA360_SDK` to the full path of the executable. A configured path that does not exist is reported as MISSING. Until it is fixed, `--pano-source studio` (an Insta360 Studio equirect export) or `dfisheye` (preview quality) still render |
| `status` says the SDK does NOT run | Run the executable with `-help` yourself from its own folder. Missing DLLs or `models\`, an old GPU driver or an incomplete extract are the usual causes |
| `stitch FAILED: error: ...` | The text after `error:` is the SDK's own message. Open `<stitched_dir>\logs\<stem>.log` for the command line and the full output. If GPU image processing fails, try adding `-image_processing_accel cpu` to `pano.stitch_args`. If the reason starts with `exit code`, check `pano.stitch_args` against the public parameter table |
| `feature <name> FAILED` or `OFF although requested` | The SDK could not enable a requested `-enable_*` option. See the log; remove the option from `pano.stitch_args` or fix the cause |
| `timed out after ...s (pano.stitch_timeout_s)` | Raise or remove `pano.stitch_timeout_s`. Clip length, stitch type and GPU decide the time |
| Very slow stitching | The SDK may be encoding in software. Update the NVIDIA driver, keep `-enable_h265_encoder` for outputs above 4096 px, and look at the encoder line the run logs (`encoder ... (...)`). `-enable_soft_encode` or `-disable_cuda` in `pano.stitch_args` forces the slow path |
| `Invalid -output_size ...` (not 2:1) | Rejected before running: fix `-output_size` in `pano.stitch_args` (width = 2 x height) |
| `No 360 clips selected` | Pass `--clips NNN ...` or `--from-clips DIR`, or run in a project that has `highlights.json` |
| `clips folder not found: <path>` | `--from-clips` takes an absolute path, `projects\<project>\<folder>`, or a folder name inside the project (e.g. `clips_review`) |
| `<folder>: N GB free, clip NNN needs ~M GB` (error before the run) | Set `pano.stitched_dir` to a drive with room |
| `... stitched with FlowState` or `no sidecar ..., assuming FlowState` warning when rendering | The equirect predates the current defaults or was stitched with `-enable_flowstate`; its framing drifts with the camera's heading. Restitch: `cutpoint 360 stitch <project> --clips NNN --force` (the warning prints the exact command) |
