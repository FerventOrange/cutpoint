# Troubleshooting

Start with `cutpoint -v <command> ...`: `-v` turns on debug logging and prints tracebacks. Errors
are logged as `ERROR` lines and the command exits with status 1 (2 for a bad command line, 130
when interrupted).

| Symptom | Page |
|---|---|
| ffmpeg not found, GPU encoder issues (NVENC, AMF, VAAPI), slow encoding, CUDA or cuDNN errors, out of GPU memory, CPU-only torch on an NVIDIA machine (`capabilities` warning) | [ffmpeg and GPU](ffmpeg-gpu.md) |
| Ollama not reachable or model not pulled (`--vision`, the `events` VLM layer, AI Chat), model downloads fail, missing module, `setup` problems | [Models and Ollama](models.md) |
| Disk full, slow or network drives, read errors, damaged files | [Storage](storage.md) |
| 360 clips, SDK not found, stitching fails | [360 and the Insta360 SDK](360.md) |

## Quick checks

```powershell
cutpoint paths                  # where Cutpoint looks for data
ffmpeg -version                 # ffmpeg is installed and on PATH
nvidia-smi                      # driver and GPU are visible (NVIDIA)
```

## Common messages

| Message or symptom | Fix |
|---|---|
| `... not available in Cutpoint lite` followed by an install command | An optional extra (`events` or `gpu`) is missing. Run the printed command: the installer again without `--lite` / `-Lite` and with `CUTPOINT_LITE` unset for a tool install, `uv sync` in a clone |
| A command works in `uv run` but not as plain `cutpoint` | The tool is not on `PATH`. Use `uv run cutpoint` in a clone, run the installer (see [Getting started](../getting-started.md#install); `--source <path>` for a clone), or use the wrappers `.\cutpoint.cmd` / `.\cutpoint.ps1` (Windows) or `./cutpoint` (Linux, needs `uv sync`) |
| PowerShell refuses to run `cutpoint.ps1` | Use `.\cutpoint.cmd`, or `uv run cutpoint` |
| `clips` refuses to write | The output folder is not empty. Use another `--out`, or `--force` to replace its clips (they are replaced only when the new export succeeds) |
| `clips` or `extend` says `<folder> is a drive root`, `is a mount point` or `is on another file system than its parent folder` | The new clips are staged next to the folder and moved in, which cannot work at a drive root or across a mount. Nothing was exported or changed. Re-run with `--out` pointing at a folder on that drive |
| `clips`, `extend` or `assemble` stops with an ERROR naming `.<folder>.cutpoint-bak` | An earlier replace was cut short (for example a power loss) and its record in that hidden folder cannot be read. Nothing is deleted. Move the files from the backup folder back into the clips folder, delete the backup folder, and re-run |
| `Another cutpoint run ... is using <folder>` or `Another cutpoint run is starting on <folder>` (an empty or unreadable lock file; `folder_busy` with `--json`) | Another run holds the folder's lock. Wait for it or stop it, then re-run. Delete `.<folder>.cutpoint-lock` only if no run is active (for example after a crash, or a run on another computer) |
| `low_memory` (`ERROR` about too little free memory for a model) | Cutpoint freed cached models and Ollama's, and memory was still short. Close other programs, then run the command the `ERROR` line names: `cutpoint resume <project>`, or the same command again when no `progress.json` was written yet. `memory_guard` in `config.json` sets the reserve in GB (`0` turns the check off). `--batch-size` does not help here: the guard already frees cached models, and the need is per model. See [Memory guard](../commands.md#memory-guard) |
| `folder_busy` on `analyze`, `edit`, `events`, `crashes`, `profile`, `360 analyze` or `clips` | A run holds `<project>/.analysis.run.cutpoint-lock`. Wait for it; a lock left by a process that is gone is taken over (on this computer; a lock from another computer is not). Delete the file only if no run is active |
| A hidden `.<folder>.cutpoint-resume` next to a clips folder | An export failed or was interrupted and kept its finished clips. The next `clips` or `extend` run with the same clips (or `cutpoint resume <project>`) encodes only the missing ones. Delete it to start over |
| `cutpoint ui` shows a project as "(interrupted)", or a **Resume analysis**, **Resume export** or **Resume one-click edit** button | The project's last run stopped. The button shows what `cutpoint resume --dry-run` reports and, once confirmed, continues the run. "Analysis is running on this project; wait for it to finish" means another run holds the project: the page checks again every 5 seconds |
| `manifest.json may not be one cutpoint wrote` | `clips --force` and `extend` do not replace a manifest cutpoint did not write. Move or rename it, or use `--out`. An unreadable manifest is refused too (it used to be replaced) |
| `assemble` joined the wrong clips or order | It uses the `.mp4` files in the folder now, in natural filename order. Rename or delete files and run it again |
| `analyze` on `gaming` or `funny` runs out of GPU memory during `transcribe` or `reactions`, or the PC is sluggish then | The HUD OCR runs in the background during those steps ([analysis steps](../commands.md#analysis-steps)). Set `CUTPOINT_ANALYZE_OVERLAP=0` to run it in its turn; the results are the same |
| Results look stale after a code or setting change | Analysis is cached per source file in `<CUTPOINT_HOME>\cache`. The cache is safe to delete, or recompute layers with `events --force LAYER` |
| Wrong clips in `highlights.json` for silent footage | Scoring is heuristic and works best with voice or mic audio. Silent gameplay relies on loudness, scene cuts and vision only. Pick segments yourself with `clips --segments` |
| Whisper hallucinated text such as "Yeah. Yeah. Yeah." | Obvious repeats are filtered. Delete `cache\<source>\transcript-*.json` to re-transcribe with the full filter |
| `--vision` ratings stay the same on every rerun | Ratings are cached per source and per frame times, model and prompt, and the cache records the Ollama model's digest: a model pulled again under the same tag is rated again, and with Ollama down the cached ratings for the tag are used. Delete `cache\<source>\vision-ratings.json` to ask Ollama again |

If none of this helps, open an issue with the command, the log lines around the error, your OS
and GPU, and the output of `cutpoint paths`. Remove private paths first. Do not paste Insta360
SDK files, documentation or output.
