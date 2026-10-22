# Models and Ollama

## Ollama is not reachable

`analyze` uses Ollama only for the optional `--vision` rating. If the server or model is not
available it skips the step quietly. With `--vision` you get an error instead.

1. Start Ollama and pull the model: `ollama pull qwen3-vl:8b`.
2. Check the server: open `http://127.0.0.1:11434` in a browser; it answers "Ollama is running".
3. On Windows, `localhost` can resolve to the IPv6 address `::1` while Ollama listens on IPv4
   only. Use the IPv4 address:

   ```powershell
   setx OLLAMA_HOST http://127.0.0.1:11434
   ```

   On Linux: `export OLLAMA_HOST=http://127.0.0.1:11434` (add it to your shell profile).

   or set `"ollama_url": "http://127.0.0.1:11434"` in `config.json`, or pass
   `analyze --ollama-url http://127.0.0.1:11434`.
4. A bare `host:port` in `OLLAMA_HOST` gets `http://` added. A wildcard bind address such as
   `0.0.0.0:11434` is turned into `127.0.0.1`.
5. For another model, use `CUTPOINT_VISION_MODEL` or `analyze --ollama-model NAME`.

The `events` model layer does not use Ollama for its main model. It starts its own
`llama-server` (llama.cpp) from `<CUTPOINT_HOME>\tools\llama.cpp`; its log is
`tools\llama.cpp\llama-server.log`.

## `cutpoint setup` fails or is incomplete

- Downloads resume. Run `cutpoint setup` again, or only the failing part:
  `cutpoint setup --only omni`.
- Components: `pip` (only checks Python packages), `panns`, `llama`, `omni`, `visual`, `voice`.
- Behind a proxy or offline, download manually and place files under `CUTPOINT_HOME` using the
  folder layout in [Configuration](../configuration.md#data-folder-cutpoint_home).
- `llama` downloads the newest llama.cpp build: Windows CUDA 12.4, or on Linux the oldest CUDA
  build with cudart in the newest release (12.8 today; needs a recent NVIDIA driver) when an
  NVIDIA driver is detected, otherwise Vulkan, otherwise CPU. With an older driver, or to use your
  own build, put `llama-server` on `PATH` or set `CUTPOINT_LLAMA_SERVER` to its full path.
- Disk space: about 8 GB for everything. Move the data folder with `CUTPOINT_HOME` or
  `CUTPOINT_MODELS_DIR` if the system drive is small.

## Missing Python packages

`events` needs the `events` extra; action-cam features and the 360 module need `ski`. A missing
module produces an error with the exact install command. Remember to list every extra you use
in one `uv sync` command.

## Whisper model download

The Whisper model downloads into the Hugging Face cache on first use. If that fails (offline,
proxy), pre-download it on a machine with access, or set `HF_HOME` to a folder that already has
it. Use `analyze --model small` for a smaller download.

## Wrong language or poor transcript

Force the language with `analyze --language en`. Music-heavy or very quiet audio produces
poor transcripts; scoring then relies more on loudness and scene cuts.
