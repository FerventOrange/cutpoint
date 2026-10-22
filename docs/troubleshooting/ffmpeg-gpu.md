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

## NVENC unavailable

Clip export uses `h264_nvenc` when it can actually encode on your machine and falls back to
`libx264` otherwise. The log says `h264_nvenc unavailable; falling back to libx264`. Output
is the same format either way; the CPU path is slower. Common causes:

- No NVIDIA GPU, or a very old driver. Update the driver and check `nvidia-smi`.
- An ffmpeg build without NVENC. Use a full build (the winget `Gyan.FFmpeg` package has it).
- Too many simultaneous NVENC sessions on consumer GPUs. Close other encoders and retry.

## Whisper runs on the CPU, or CUDA / cuDNN / cuBLAS errors

Transcription uses CUDA when it can and falls back to CPU (`small` model by default instead of
`medium`). The CUDA runtime libraries (cuBLAS, cuDNN) are installed with Cutpoint on Windows
and Linux, so you only need a current NVIDIA driver.

- Run `uv sync` again (with all the extras you use) if the libraries went missing.
- Update the NVIDIA driver. On Linux the pip-installed CUDA libraries are preloaded
  automatically; if Whisper still falls back to the CPU, check the driver with `nvidia-smi`.
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

A tool install (`uv tool install cutpoint`) gets CPU-only torch on Windows, because the CUDA
index in `pyproject.toml` only applies to a clone. Reinstall with the PyTorch index:

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
