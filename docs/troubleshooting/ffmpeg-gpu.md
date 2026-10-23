# ffmpeg and GPU problems

## ffmpeg or ffprobe not found

Cutpoint looks for ffmpeg in this order: `CUTPOINT_FFMPEG` / `CUTPOINT_FFPROBE` (full paths),
`PATH`, then the WinGet package folder.

```powershell
winget install --id Gyan.FFmpeg -e
ffmpeg -version
```

On Linux, install it with your package manager (`sudo apt install ffmpeg`, or `dnf`, `pacman`).
If the log says ffmpeg was not found, install it or point `CUTPOINT_FFMPEG` and
`CUTPOINT_FFPROBE` at the binaries. For a custom build, set the variables:

```powershell
setx CUTPOINT_FFMPEG C:\tools\ffmpeg\bin\ffmpeg.exe
setx CUTPOINT_FFPROBE C:\tools\ffmpeg\bin\ffprobe.exe
```

Open a new shell afterwards; `setx` does not change the current one. On Linux:

```bash
export CUTPOINT_FFMPEG=/opt/ffmpeg/bin/ffmpeg
export CUTPOINT_FFPROBE=/opt/ffmpeg/bin/ffprobe
```

## Which video encoder is used, and why not the GPU

Clip export, `assemble` (also with `--captions`) and 360 renders encode H.264 with the first
encoder that works, found by really encoding a short test clip with each candidate: NVENC, then
AMF on Windows, or VAAPI and AMF on Linux. If none works the log says
`No GPU video encoder works; encoding with libx264`. Otherwise it says which one it picked, for
example `Video encoder: h264_vaapi on /dev/dri/renderD128`. The output format is the same either
way; the CPU path is slower. Quality is set per encoder from the same number (CRF for libx264,
CQ for NVENC, QP for AMF and VAAPI), not by a fixed bitrate.

Check what works on this machine:

```bash
cutpoint capabilities
cutpoint capabilities --json
```

The report keeps three things apart: which GPU vendors are present (hardware only), which
encoders are listed by your ffmpeg and which of them really encode, and what transcription uses.
An AMD or NVIDIA GPU being present does not make an encoder usable; a listed encoder can still
fail (for example `h264_nvenc` or `h264_amf` listed on a machine without that driver).

To choose one yourself, set `video_encoder` in `config.json` or `CUTPOINT_VIDEO_ENCODER` to `auto`,
`nvenc`, `amf`, `vaapi` or `cpu` (see [configuration](../configuration.md#video_encoder)). A named
GPU encoder that does not work stops the command with
`video_encoder 'X' cannot encode on this machine (...)` and the ffmpeg error; it never falls back
silently. Use `cpu` to force libx264 and `auto` to go back to detection.

With `auto`, the probe only proves that a short test clip encodes. If a GPU encode then fails on
real footage, Cutpoint logs one WARNING, redoes that clip, 360 render or final video once on the
CPU (libx264) and uses the CPU for the rest of the run. A forced GPU encoder is not retried; its
error tells you to set `CUTPOINT_VIDEO_ENCODER=cpu` or `"video_encoder": "cpu"` in `config.json`.

Clips, `extend`, 360 renders and `final.mp4` are written to a hidden `.<name>.part.mp4` and
renamed when complete, so a failed or cancelled export leaves no partial file under the real
name. A hard kill (SIGKILL, or the UI cancel on Windows) can leave a hidden `.part.mp4`; commands
ignore it and `clips --force` removes it. `clips` (except into an empty drive root or mount point), in-place `extend` and `extend --out DIR --force` into a non-empty `DIR` export into a hidden
`.<folder>.cutpoint-tmp` (and `.<folder>.cutpoint-bak` while swapping); the next run on that
folder removes them and, if the swap was cut short, puts the old clips back. If the record of the swap in the backup folder cannot be read (for
example after a power loss), the run stops with an ERROR naming the backup folder and deletes
nothing; move the files from it back into the clips folder, delete the backup folder, and re-run.

Common causes when the GPU encoder is not used:

- **NVENC:** no NVIDIA GPU, a very old driver (check `nvidia-smi`), an ffmpeg build without
  NVENC, or too many simultaneous NVENC sessions on consumer GPUs.
- **VAAPI (Linux, AMD or Intel):** no `/dev/dri/renderD*` device, no permission on it (your user
  needs the `render` or `video` group), a missing Mesa/VAAPI driver, or an ffmpeg build without
  VAAPI. Every render node is tried in name order and the first one that encodes is used, so a
  machine with two GPUs (for example an iGPU and a dGPU) picks whichever node works first.
- **AMF:** needs an ffmpeg build with AMF and AMD's driver. It was not exercised on a real
  Windows machine while this was written, so run `cutpoint capabilities` there and use
  `CUTPOINT_VIDEO_ENCODER=cpu` if it fails.

Limits: only encoding runs on the AMD GPU. Video decoding for analysis, HUD OCR and 360 renders
still uses CUDA when it can and the CPU otherwise.

## Whisper runs on the CPU, or CUDA / cuDNN / cuBLAS errors

Transcription uses CUDA when it can and falls back to CPU (`small` model by default instead of
`medium`). The CUDA runtime libraries (cuBLAS, cuDNN) are in the `gpu` extra, not the base
install. With an NVIDIA GPU but no `gpu` extra, Cutpoint logs one line and uses the CPU:

```
INFO NVIDIA GPU found, but the CUDA libraries for whisper are not installed (...); transcribing on the CPU. For GPU transcription add the 'gpu' extra: <command>
```

- Add the extra with the command in that line: `uv sync --extra gpu` (plus every other extra you
  use) from a clone, `uv tool install "cutpoint[gpu]" --reinstall` for a tool install (plus your
  other extras), or `pip install "cutpoint[gpu]"`.
- Update the NVIDIA driver. On Linux the pip-installed CUDA libraries are preloaded
  automatically; if Whisper still falls back to the CPU, check the driver with `nvidia-smi`.
- AMD GPUs are not used for transcription: faster-whisper has no AMD GPU backend in Cutpoint, so
  on an AMD machine it always runs on the CPU (`cutpoint capabilities` shows `cpu`). That is
  normal, not an error.
- To force a smaller model, pass `analyze --model small`.

## Captions fail on Linux

`assemble --captions` needs fontconfig and a font; the font is chosen with `fc-match sans-serif`.

```bash
sudo apt install fontconfig fonts-dejavu-core
```

## Out of GPU memory

`cutpoint events` runs its layers one after another so only one model is loaded at a time.
Close other GPU programs (games, other models). The Qwen2.5-Omni layer is the heaviest;
lower its request size with `events --frames 12 --frame-height 540`, or skip it with
`--no-vlm`. Ollama models are unloaded before the llama.cpp server starts.

## `events` is very slow

PANNs, the voice layer and the visual layer pick the GPU automatically when the installed
torch has CUDA. With CPU-only torch they still work, but the visual layer takes roughly ten
times longer. Check with `uv run python -c "import torch; print(torch.cuda.is_available())"`.
`uv sync --extra events` installs the CUDA 12.6 builds on Windows and Linux.

A PyPI or wheel tool install (`uv tool install cutpoint`) gets CPU-only torch on Windows. The CUDA
index in `pyproject.toml` also applies to tool installs and `uv pip install` from a clone directory
or git URL, so those need no index flags (with `uv pip install`, an already-installed CPU torch also
needs `--reinstall-package torch --reinstall-package torchvision`). `pip` does not apply it. For a
PyPI or wheel install, reinstall with the PyTorch index:

```powershell
uv tool install "cutpoint[all]" --index https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match --reinstall
```

Then check the tool's own Python:
`& "$(uv tool dir)\cutpoint\Scripts\python.exe" -c "import torch; print(torch.cuda.is_available())"`
On Linux: `"$(uv tool dir)/cutpoint/bin/python" -c "import torch; print(torch.cuda.is_available())"`.

## Slow or odd video decoding

- Hardware decoding (`-hwaccel cuda`) is used for HUD OCR. If it fails on your GPU, pass
  `events --no-hwaccel`.
- 10-bit or exotic codecs may fall back to CPU decoding, which is slower but correct.
