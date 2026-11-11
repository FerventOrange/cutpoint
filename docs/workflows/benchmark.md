# Cross-category benchmark

`scripts/bench.py` measures detection and ranking on labelled footage of every category. Run it
before and after a change to scoring or detection, and put its delta in the pull request. It is
a development script, not part of the package: it runs `cutpoint analyze --style auto` on each
source, scores the result and writes a report. It never changes the manifest, the labels or the
footage.

Keep the manifest, the label files and the footage out of the repository: they describe private
footage. The paths on this page are examples (on Windows, for example `H:/cutpoint-scratch/bench`).

## Manifest

One JSON file. Paths in it are relative to the manifest's folder.

```json
{
  "schema": 1,
  "holdout_sha256": "<printed by --rehash>",
  "sources": [
    {"id": "race-night", "input": "footage/race-night.mp4", "category": "gameplay",
     "bucket": "gameplay-nokills", "labels": "labels/race-night.json", "holdout": false},
    {"id": "trail-day", "input": "footage/trail-day", "category": "action-cam",
     "bucket": "actioncam-telemetry", "holdout": false, "note": "folder of chapters"},
    {"id": "podcast-3", "input": "footage/podcast-3.mp4", "category": "talk",
     "bucket": "talk", "labels": "labels/podcast-3.json", "holdout": true}
  ]
}
```

| Field | Meaning |
|---|---|
| `id` | Unique name: letters, digits, `.`, `_`, `-`. It names the run folder |
| `input` | A file or a folder, as `analyze` takes it |
| `category` | The true footage category: `action-cam`, `gameplay`, `talk` or `general` |
| `bucket` | The group it is reported in (next table) |
| `labels` | Optional moment label file. Without it the source is scored for detection only |
| `holdout` | `true` keeps the source out of tuning (see [Holdout](#holdout)) |
| `note` | Optional free text |
| `variants` (top level) | Optional, see [Variants](#variants-a-vs-b) |

The manifest is checked before anything runs: an unknown category or bucket, a duplicate id, a
missing input or label file, or a label file whose own `holdout` disagrees with the manifest is
an error (exit 2). A label file without `holdout` is fine.

### Buckets

The labeller sets the bucket; the script does not derive it.

| Bucket | Footage |
|---|---|
| `gameplay-kills` | Gameplay with a kill feed or other HUD events |
| `gameplay-nokills` | Gameplay without them (racing, building, cleaning games) |
| `actioncam-telemetry` | Action-cam footage with GoPro or Insta360 X3 telemetry |
| `actioncam-notelemetry` | Action-cam footage without telemetry |
| `talk` | Podcasts, talking heads, commentary |
| `vlog` | Vlogs and other general footage |
| `mixed` | A source of several categories. Its `category` is the category of most of the footage by duration, set by the labeller |

## Labels

Label files are the `scripts/moment_eval.py` format and load unchanged: `video` (the file
name of the source), `keep` and `skip` (moments judged good or not good), `reference` (ranges of
a hand edit), and optional `duration`, `holdout` and `note`. Times are seconds; the bench also
takes `"mm:ss"` and `"h:mm:ss"` strings (with decimals, such as `"1:15.5"`).

```json
{"video": "race-night.mp4", "duration": "1:02:10",
 "keep": [["12:31", "12:39"], [2210.5, 2224]],
 "skip": [["40:02", "40:08"]],
 "reference": []}
```

A file with only `reference` ranges (and empty `keep` and `skip`) is fine: it is scored on the
`ref_*` metrics only. The metrics are those of `moment_eval.py` (its docstring explains them):
`keep_top10`/`keep_top20` and `skip_top10`/`skip_top20` (labels hit by the 10 or 20 best
candidates), `keep_15`/`keep_30` and `ref_15`/`ref_30` (labels hit when the best candidates fill
15% or 30% of the video), each against a random control ("beats noise").

## Holdout

Holdout sources are for the release check: never tune on them.

- By default they are neither analyzed nor reported.
- `--holdout` also runs them and reports them in a separate HOLDOUT section, and the results
  file records `"holdout_viewed": true`. The gate never uses them.
- `holdout_sha256` pins the holdout set: the sha256 of the sorted holdout ids joined by
  newlines. If the manifest's holdout ids do not give it, the bench refuses to run (exit 2), so a
  source cannot move in or out of the holdout unnoticed. With holdout sources and no hash it
  refuses too and prints the hash to add.
- To change the holdout set on purpose, edit the manifest, run `--rehash`, and store the hash it
  prints as `holdout_sha256`. `--rehash` prints the hash and stops; the script never rewrites
  the manifest.

## Running

```powershell
uv run --offline python scripts/bench.py H:/cutpoint-scratch/bench/manifest.json
```

Per source and variant, one job at a time:
`<cutpoint> analyze <input> --style auto --out <bench>/runs/<variant>/<id> --json <variant args>`.

- `<bench>` is the manifest's folder, or `--bench DIR`. The bench refuses a folder that holds a
  source file or is inside a source folder: it never writes next to the footage.
- `<cutpoint>` is `python -m cutpoint` with the interpreter that runs the script, so the
  checkout and environment you run the bench from are the ones measured. A variant's `cutpoint`
  points elsewhere.
- Every run gets `CUTPOINT_HOME=<bench>/home` (`--home DIR` to change it),
  `CUTPOINT_CACHE_DIR=<bench>/cache` (`--cache DIR` to change it, or the variant's own
  `cache`, see [Variants](#variants-a-vs-b)) and your models folder (`CUTPOINT_MODELS_DIR`,
  unless you set it). The bench home keeps discovered profiles, HUD presets and settings out of
  your own home and the same between runs. Put a `config.json` in `<bench>/home` for settings
  such as `ollama_url` or `vision`; without one the defaults apply. The shared cache keeps
  transcripts and probes between variants and runs.
- A run is reused when its stamp (`<bench>/runs/<variant>/<id>.stamp.json`) matches: the
  cutpoint command, its version, the checkout's git commit (with a hash of the uncommitted
  changes and untracked files under `src`; only for the default cutpoint), the edition, the
  variant's args, the cache folder, the sha256 of the config file the run reads (`config.json`
  of the bench home), and the size and time of the input files. `--rerun` analyzes again. A
  variant that points at another checkout records only its version, so use `--rerun` after
  changing that checkout.
- `--reuse ID=DIR` scores an existing project folder for that source instead of running it
  (for every variant).
- One bench at a time per machine: `cutpoint-bench.lock` in the temporary folder (`/tmp` on
  Linux, `%TEMP%` on Windows) holds the bench's process id for the whole run, and a second bench
  stops with an error, whatever its bench folder. A lock whose process is gone is replaced (on
  Windows the check opens the process; it never signals it). Delete the file only if no bench
  is running.
- Ctrl+C stops the running analyze and gives it 30 s to clean up before it is killed; the bench
  exits 130. Finished runs keep their stamps, so the next run reuses them. A `cutpoint` command
  that cannot start (a wrong path) is a failed job with the message, and the bench goes on.
- While each job runs, a background thread samples every 5 s (`--sample-interval`, 0 for none)
  Ollama's loaded models (`/api/ps` at `OLLAMA_HOST`, the bench home's `ollama_url` or the
  default) and `nvidia-smi` (memory used and total, utilization per GPU). A missing nvidia-smi
  or an Ollama that does not answer is recorded as unavailable. The raw samples and their
  min, max and mean are in the results file; the report has a summary per job.

Analysis takes long: run it in the background or overnight.

### Full and lite

The bench works on both editions. The edition is recorded in each results file
(`cutpoint capabilities` for a variant with its own `cutpoint`). For lite, run it from a lite
environment: a clone synced with `uv sync --no-group full`, then
`uv run --no-group full --offline python scripts/bench.py ...`. Compare a lite run against a lite
baseline: lite against full gives the deltas with a warning, but no gate.

## Variants (A vs B)

A variant is `{"name": ..., "args": [...], "cutpoint": ..., "cache": ...}`: extra `analyze`
options, optionally another cutpoint to run, and optionally a cache folder of its own (relative
to the manifest). `args` and `cutpoint` are lists, or strings split like a command line: with
POSIX rules on Linux and macOS, and on Windows with Windows rules, which keep backslashes in
paths (`H:\cutpoint-main\.venv\Scripts\cutpoint.exe` works) and take the surrounding
quotes off a quoted token (`"C:\My Tools\cutpoint.exe"`). Give them in the manifest's
`variants` or as `--variant NAME:ARGS` flags (which replace the manifest's):

```powershell
uv run --offline python scripts/bench.py manifest.json --variant base: --variant novis:--no-vision
```

```json
"variants": [
  {"name": "base", "args": []},
  {"name": "novis", "args": ["--no-vision"]}
]
```

Caches: Cutpoint's cache keys on the source and on cache versions, not on the code. A shared
cache is right when variants differ only in options (`--no-vision`); to compare two code
versions, give each variant its own `cache`, so the second one does not read what the first
one computed:

```json
"variants": [
  {"name": "main", "cutpoint": ["uv", "run", "--project", "H:/cutpoint-main", "cutpoint"],
   "cache": "cache-main"},
  {"name": "branch", "cache": "cache-branch"}
]
```

The first variant is the baseline; each other one is compared against it. Without variants
there is one, `base`. `--compare A.json B.json` compares two earlier results files the same way
and prints the comparison (it writes no file); a file that is not a bench results file is an
error (exit 2). It warns when the editions, schema, overlap rule, seed or number of control runs
differ. Each results file records per source its category, bucket, holdout flag and the sha256
of its label file: a source where these differ between the two runs was labelled differently,
so it is named in a warning and left out of the comparison and the gate. The holdout is
compared only with `--holdout` (and when both runs viewed it).

## The report

Each run writes `<bench>/results/<timestamp>-<variant>.json` per variant and
`<timestamp>-report.md`, and prints the report:

- Per variant, detection (tune split) per bucket: sources, right (sure and the labelled
  category), ambiguous, sure-wrong (sure and another category: these get the wrong style with no
  fallback), accuracy, ambiguous rate, and the confusion: per detected category the sure count and
  `+N?` ambiguous ones. Sure-wrong sources and sources without a detection (a failed run) are
  listed.
- Per variant, ranking per bucket: `keep_top10`, `keep_top20`, `skip_top10`, `skip_top20`,
  `keep_15`, `keep_30`, `ref_30`, each summed over the labelled videos as hits/labels, and how
  many videos' `keep_30` beats noise.
- With `--holdout`, the same for the holdout, apart.
- Per comparison, the per-bucket deltas (accuracy, ambiguous rate, sure-wrong count, the ranking
  sums as A -> B), the sources that changed, and the gate verdict.
- Resources: per job the status (`ok`, `reused`, `given`, `failed`), seconds, the GPU memory peak,
  the mean GPU use and Ollama's loaded VRAM and models.

## The gate

The gate is advice: it prints PASS or FAIL with the reasons, and the bench exits 0 either way.
`--strict-gate` makes a FAIL exit 3. It checks the tune split only; the holdout is checked by
hand at release. B fails when, over the sources both runs scored:

| Rule | Threshold flag (default) |
|---|---|
| A source that A scored failed or has no result in B | - |
| A source is sure-wrong in B and was not in A | - |
| A bucket's ambiguous rate rose more than the limit | `--gate-ambiguous-pp` (5 points) |
| A bucket's summed `keep_top20` or `keep_30` fell by more than the larger of N ranges and a share of A's sum | `--gate-keep-drop` (1), `--gate-keep-drop-share` (0.10) |
| A bucket's summed `skip_top20` rose by more than N | `--gate-skip-rise` (1) |
| Fewer of a bucket's videos beat noise on `keep_30` | - |

No gate is given (only deltas, with a warning) when the runs are of different editions (lite
against full), schemas or overlap rules.
