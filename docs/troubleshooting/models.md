# Models and Ollama

## Ollama is not reachable

`analyze` uses Ollama only for the optional `--vision` rating, which is off unless you pass
`--vision` (or turn it on in the Models tab in `cutpoint ui`, which sets `"vision": true` in
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

The VLM layer of `cutpoint events` uses the same vision model as `--vision`: Ollama with
`qwen3.5:9b` (`CUTPOINT_VISION_MODEL` changes it), or the hosted (API or subscription) Vision
model of the Models tab once you allowed vision requests there. When neither answers, `events` logs
`VLM: vision model ... not available`, keeps the window descriptions it has cached and goes on
with the other layers. Check the points above, or pull the model with `ollama pull qwen3.5:9b`.

## `cutpoint setup` fails or is incomplete

- Downloads resume. Run `cutpoint setup` again, or only the failing part:
  `cutpoint setup --only visual`.
- Components: `pip` (only checks Python packages), `whisper`, `panns`, `visual`, `voice`.
- `CLIP weights not found in .../models/clip; run: cutpoint setup --only visual`: a step that
  uses CLIP found no weights there. Only `cutpoint setup` downloads them; other commands load
  them from disk and never download. Check `CUTPOINT_MODELS_DIR` points at the folder setup used.
- `Faster R-CNN detector weights not found in .../models/torch/hub/checkpoints; run: cutpoint
  setup --only visual`: the visual layer, `360 analyze` or `360 qa` found no detector checkpoint
  there (`TORCH_HOME/hub/checkpoints` when you set `TORCH_HOME`). Only `cutpoint setup`
  downloads it; other commands load it from disk and never download.
- Behind a proxy or offline, download manually and place files under `CUTPOINT_HOME` using the
  folder layout in [Configuration](../configuration.md#data-folder-cutpoint_home).
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

The Whisper model downloads into the Hugging Face cache on first use, with an INFO line such as
`Downloading Whisper medium (~1.5 GB) on first use (cutpoint setup --only whisper fetches it
ahead of time)`. Once it is cached, runs load it without a Hugging Face Hub request. To fetch it
before a run, use `cutpoint setup --only whisper` (`--whisper-model NAME` for another model);
an interrupted download resumes. If that fails (offline, proxy), pre-download it on a machine
with access, or set `HF_HOME` to a folder that already has it, or pass a model folder to
`--model`. Use `analyze --model small` for a smaller download.

## Wrong language or poor transcript

Force the language with `analyze --language en`. It transcribes the footage again in that
language (an earlier auto-detected transcript stays cached for other projects). A transcript
made with `--language` by an earlier Cutpoint version was cached under the plain name, so a run
without `--language` still reuses it; delete `cache\<source>\transcript-*.json` to transcribe
the source again. Music-heavy or very quiet audio produces poor transcripts; scoring then
relies more on loudness and scene cuts.
