# Models and Ollama

## Ollama is not reachable

`analyze` uses Ollama only for the optional `--vision` rating, which is off unless you pass
`--vision` (or turn it on in Settings in `cutpoint ui`, which sets `"vision": true` in
`config.json`; see [configuration](../configuration.md#vision)). If the server or model is not available
then, it logs a WARNING (a `vision_unavailable` warning with `--json`) and continues without the
rating; if some of the top candidates have cached ratings, it uses those instead. The local AI
Chat, `360 director --vl-check` and building a new activity profile use it too. To install it, see
[getting started](../getting-started.md#optional-ollama-for-frame-rating-and-ai-chat).

1. Start Ollama and pull the model: `ollama pull qwen3.5:9b` (a model-not-found message means
   it is not pulled; `ollama list` shows what is). The AI Chat uses the same model. If you set
   `CUTPOINT_VISION_MODEL` or pass `--ollama-model` (`qwen3-vl:8b`, the earlier default, still
   works), pull that model instead.
2. Check the server: open `http://127.0.0.1:11434` in a browser; it answers "Ollama is running".
3. The default URL is already `http://127.0.0.1:11434`. If you set `localhost` yourself, note that
   on Windows it can resolve to the IPv6 address `::1` while Ollama listens on IPv4 only. Use the
   IPv4 address:

   ```powershell
   setx OLLAMA_HOST http://127.0.0.1:11434
   ```

   On Linux: `export OLLAMA_HOST=http://127.0.0.1:11434` (add it to your shell profile).

   or set `"ollama_url": "http://127.0.0.1:11434"` in `config.json`, or pass
   `analyze --ollama-url http://127.0.0.1:11434`.
4. A bare `host:port` in `OLLAMA_HOST` gets `http://` added. A wildcard bind address such as
   `0.0.0.0:11434` is turned into `127.0.0.1`.
5. For another model, use `CUTPOINT_VISION_MODEL` or `analyze --ollama-model NAME`.

Replies in the AI Chat or `--vision` ratings that take very long usually mean the model runs on
the CPU (it did not fit in GPU memory); a smaller model or a GPU helps.

The `events` model layer does not use Ollama for its main model. It starts its own
`llama-server` (llama.cpp) from `<CUTPOINT_HOME>\tools\llama.cpp`; its log is
`tools\llama.cpp\llama-server.log`.

## `cutpoint setup` fails or is incomplete

- Downloads resume. Run `cutpoint setup` again, or only the failing part:
  `cutpoint setup --only omni`.
- Components: `pip` (only checks Python packages), `panns`, `llama`, `omni`, `visual`, `voice`.
- Behind a proxy or offline, download manually and place files under `CUTPOINT_HOME` using the
  folder layout in [Configuration](../configuration.md#data-folder-cutpoint_home).
- `llama` downloads llama.cpp assets: on both Windows and Linux, `auto` prefers CUDA when a
  usable NVIDIA driver is detected, otherwise Vulkan, then CPU when distribution assets are
  available (see [Configuration](../configuration.md#environment-variables)). You can override this
  with `CUTPOINT_LLAMA_BACKEND` (`auto`, `cuda`, `vulkan`, `cpu`). With an unsupported driver, or to
  use your own build, put `llama-server` on `PATH` or set `CUTPOINT_LLAMA_SERVER` to its full path.
- `<dir> was not installed by cutpoint setup (no VERSION.txt)`: a self-built or unpacked llama.cpp
  is in `tools/llama.cpp` (or `tools/llama.cpp.old`), and setup will not replace it. Move it
  elsewhere (or point `CUTPOINT_LLAMA_SERVER` at it) and run `cutpoint setup --only llama` again.
- `Could not remove tools/llama.cpp.old (a running llama-server holds files in it)`: a warning,
  not an error. On Windows a running `llama-server` does not stop the update; the new build is
  installed, but the previous one stays in `tools/llama.cpp.old`. Stop cutpoint and any running
  `llama-server`, then run `cutpoint setup --only llama` again to remove it.
- `Could not replace <dir>` or `Could not remove <dir>` (an error): another program, for example
  antivirus or a running `llama-server`, holds files in `tools/llama.cpp` or
  `tools/llama.cpp.old`. Stop cutpoint and any running `llama-server`, then run
  `cutpoint setup --only llama` again. The previous install is kept.
- `sha256 mismatch`: the llama.cpp download was corrupt and was deleted. Run
  `cutpoint setup --only llama` again; the existing install is kept.
- `panns` downloads the checkpoint from a Hugging Face copy of the Zenodo record (CC BY 4.0)
  first, then from Zenodo, and checks size and sha256. A source that fails or starts too slowly is
  skipped. To use your own copy, set `CUTPOINT_PANNS_URL`; it is tried first.
- `Could not download Cnn14_DecisionLevelMax_mAP=0.385.pth from any mirror`: every source failed;
  the message lists the reason for each. Check the network, run `cutpoint setup --only panns`
  again (partial downloads resume), or set `CUTPOINT_PANNS_URL`.
- Disk space: see [Storage](storage.md#disk-space) for sizes. Move the data folder with `CUTPOINT_HOME` or
  `CUTPOINT_MODELS_DIR` if the system drive is small.

## Missing Python packages

`events` needs the `events` extra, which Cutpoint lite leaves out. A missing module produces an
error that says the feature is not available in Cutpoint lite, with the install command: the
installer again without `--lite` / `-Lite` and with `CUTPOINT_LITE` unset for a tool install,
`uv sync` in a clone. Action-cam telemetry
and the 360 module are in every install.

## Whisper model download

The Whisper model downloads into the Hugging Face cache on first use. If that fails (offline,
proxy), pre-download it on a machine with access, or set `HF_HOME` to a folder that already has
it. Use `analyze --model small` for a smaller download.

## Wrong language or poor transcript

Force the language with `analyze --language en`. It transcribes the footage again in that
language (an earlier auto-detected transcript stays cached for other projects). A transcript
made with `--language` by an earlier Cutpoint version was cached under the plain name, so a run
without `--language` still reuses it; delete `cache\<source>\transcript-*.json` to transcribe
the source again. Music-heavy or very quiet audio produces poor transcripts; scoring then
relies more on loudness and scene cuts.
