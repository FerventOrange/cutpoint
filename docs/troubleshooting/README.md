# Troubleshooting

Start with `cutpoint -v <command> ...`: `-v` turns on debug logging and prints tracebacks. Errors
are logged as `ERROR` lines and the command exits with status 1.

| Symptom | Page |
|---|---|
| ffmpeg not found, NVENC errors, slow encoding, CUDA or cuDNN errors, out of GPU memory | [ffmpeg and GPU](ffmpeg-gpu.md) |
| Ollama not reachable, model downloads fail, missing module, `setup` problems | [Models and Ollama](models.md) |
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
| `No module named ...` followed by an install command | An optional extra is missing. Run the printed `uv sync --extra ...` command and list every extra you use, because `uv sync` removes the packages of extras you leave out |
| A command works in `uv run` but not as plain `cutpoint` | The tool is not on `PATH`. Use `uv run cutpoint`, install with `uv tool install cutpoint` (or `uv tool install .` from a clone), or use the wrappers `.\cutpoint.cmd` / `.\cutpoint.ps1` (Windows) or `./cutpoint` (Linux, needs `uv sync`) |
| PowerShell refuses to run `cutpoint.ps1` | Use `.\cutpoint.cmd`, or `uv run cutpoint` |
| `clips` refuses to write | The output folder is not empty. Use another `--out`, or `--force` to replace it |
| `assemble` joined the wrong clips or order | It uses the `.mp4` files in the folder now, in natural filename order. Rename or delete files and run it again |
| Results look stale after a code or setting change | Analysis is cached per source file in `<CUTPOINT_HOME>\cache`. The cache is safe to delete, or recompute layers with `events --force LAYER` |
| Wrong clips in `highlights.json` for silent footage | Scoring is heuristic and works best with voice or mic audio. Silent gameplay relies on loudness, scene cuts and vision only. Pick segments yourself with `clips --segments` |
| Whisper hallucinated text such as "Yeah. Yeah. Yeah." | Obvious repeats are filtered. Delete `cache\<source>\transcript-*.json` to re-transcribe with the full filter |

If none of this helps, open an issue with the command, the log lines around the error, your OS
and GPU, and the output of `cutpoint paths`. Remove private paths first. Do not paste Insta360
SDK files, documentation or output.
