# Storage, slow drives and damaged files

## Disk space

| What | Where | Size |
|---|---|---|
| Models for `events` | `<CUTPOINT_HOME>\models`, `tools` | about 8 GB |
| Analysis cache | `<CUTPOINT_HOME>\cache` | grows with footage; safe to delete |
| Staged copies | `<CUTPOINT_HOME>\stage` | a full copy of each staged source; `cutpoint stage --clear` frees it |
| Projects | `projects_dir` | clips and finals are re-encoded video |
| 360 stitches | `pano.stitched_dir` | about 54 GB per hour of footage |

If the system drive is small, move things: `CUTPOINT_HOME` for everything, or
`projects_dir`, `stage_dir` and `pano.stitched_dir` individually
([Configuration](../configuration.md)). `cutpoint stage` refuses a copy that would leave less than
5 GB free.

## Footage on slow or network storage

Every pass over a source reads it again: audio extraction, scene proxy, OCR decode, model
frames and clip export. On a network share, a USB disk or a cloud-synced folder this adds up.
Copy the sources to a local disk once:

```powershell
cutpoint stage D:\Videos\session1        # files and/or folders; --exclude GLOB
cutpoint stage --list                    # what is staged and how long each read took
cutpoint stage --clear                   # delete the copies
```

After that, every ffmpeg and ffprobe read of a staged source uses the local copy. Projects,
caches, manifests and timelines keep the original path, so clearing the stage only makes reads
slow again. Files already staged with the same size and modification time are skipped.

Notes:

- `stage` does not resume a partial copy after a network drop; the next run restarts that file.
- If the footage is too large to stage at once, process it in parts: stage part A, run
  `analyze` and `events`, `cutpoint stage --clear`, stage part B, then run the whole set. The
  caches of part A carry over.
- Moved footage? Use `source_remap` in `config.json` to read from the new location without
  re-analysing ([Configuration](../configuration.md#source_remap-and-source_remap_exclude)).
- `crashes` and `events` skip sources that are offline or moved (warning `footage not reachable`)
  and fail with `No footage of <project> is reachable` if none is. Reconnect the footage and rerun.
- Action-cam telemetry reads whole files; stage first. Crash detection reads only the telemetry
  track and needs no staging.

## Flaky network links while writing

If the link drops during long writes, keep writes local and copy once:

1. Export clips and finals to a local folder: `cutpoint clips ... --out <local dir>`.
2. Copy to the network location under a temporary name, verify the size with ffprobe, then rename.
3. Run `assemble` with a local `--out`.

## Read errors and damaged files

A failing disk or damaged file can make ffmpeg exit successfully with a silently truncated
output. Cutpoint checks for this.

- `analyze` warns with the file name, the approximate failure time and how much was not
  analysed, and records `read_errors` for that source in `highlights.json`. Analysis of that
  source stops at the failure.
- `analyze --recover` probes a few seconds of audio at growing distances past the failure,
  and resumes at the first readable point as an extra virtual source. No copy of the video is
  made, and times stay absolute in the original file.
- `clips` checks each export for truncation, retries the read once, and re-encodes to end
  cleanly if it is still short, with a `read_error` record in `manifest.json`.
- `events --avoid "GLOB@START-END"` keeps a range from ever being decoded.
- On Windows a bad sector often shows up as `Error during demuxing: Permission denied`.

Best fix: copy the file to a healthy disk (or let `cutpoint stage` do it) and point
`source_remap` at the copy.
