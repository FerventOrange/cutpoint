# Storage, slow drives and damaged files

## Disk space

| What | Where | Size |
|---|---|---|
| Models for `events` | `<CUTPOINT_HOME>\models` | about 2 GB (the Ollama vision model is stored by Ollama) |
| Analysis cache | `<CUTPOINT_HOME>\cache` | grows with footage; safe to delete; while scene cuts are detected, a temporary proxy of up to about 0.6 MB per source second (16:9; up to 1 MB for 10-bit video, more for portrait), removed after the scan |
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
- If the original path is gone but `source_remap` maps it to an existing location, `analyze`,
  `crashes` (default `--reanalyze`) and action-cam `events` read from the new location and re-analyze.
- They instead warn `path(s) not found where analyze recorded them, even through source_remap
  ...; highlights.json left unchanged` and exit 0 when re-analysis would fail, drop a source or
  re-key one: for example a file moved out of a folder input that still exists, or a 0.1.0
  project that recorded the copy path. Rerun `analyze` on the new location, then
  `crashes`/`events` (they match sources by path).
- Projects analyzed before inputs were stored as absolute paths keep relative ones until
  `analyze` runs again.
- Action-cam telemetry reads whole files; stage first. Crash detection reads only the telemetry
  track and needs no staging.

## Preview slow to start in `cutpoint ui`

When the preview player waits for data over 5 s, a note over the picture names the likely
reason: the project folder is on a network or external drive, or the video is a fragmented MP4 (as some
cameras and screen recorders write it), which the browser opens in hundreds of small reads.
Short clips in Play sequence show it most. The note goes once playback starts. To make it faster,
keep `projects_dir` on a local disk, or remux the video into a plain MP4
(`ffmpeg -i in.mp4 -c copy -movflags +faststart out.mp4`) and use that copy.

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

## Docker volume from the 0.2.0 image

The 0.2.0 image ran as root, so an existing `cutpoint-data` volume is root-owned and newer images
fail with `Permission denied` on `/data`. Fix the volume once (use the build-arg uid:gid instead
of `1000:1000` if you changed them):

```bash
docker run --rm --user 0 --entrypoint chown \
  --mount source=cutpoint-data,target=/data cutpoint -R 1000:1000 /data
```
