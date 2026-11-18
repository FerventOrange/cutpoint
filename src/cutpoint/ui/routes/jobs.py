"""The routes that start, list and cancel the jobs of ``cutpoint ui``."""

from __future__ import annotations

import contextlib
import re
from collections.abc import Callable
from datetime import datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from cutpoint import cache, insta360, progress, runstate
from cutpoint.assemble import (
    AUDIO_TRANSITIONS,
    OUTPUT_FORMATS,
    default_audio_transition,
    exact_clips,
)
from cutpoint.assistant import providers as ai_providers
from cutpoint.config import projects_dir, vision_assistant_settings
from cutpoint.edit import KINDS
from cutpoint.pano import x3
from cutpoint.sources import TRANSCRIPT_MODELS
from cutpoint.styles import AUTO
from cutpoint.ui.analysis import (
    ANALYZE_KEYS,
    ProjectStyle,
    _action_cam_style,
    _analyze_args,
    _auto_picks,
    _color_args,
    _compress_peaks,
    _flag,
    _keep_style,
    _no_options,
    _note_warnings,
    _opt,
    _picture_args,
    _saved_skip,
    _seconds,
    _speech_args,
    _subscription_roles,
    _vision_args,
    _vision_detect_all,
    _vision_detect_all_arg,
    analyze_options,
    crash_step_runs,
    events_unavailable,
    preset_problem,
    preset_taken,
    set_style,
    stitch_first,
    ui_styles,
)
from cutpoint.ui.common import (
    ANALYSIS_FILE,
    BUSY_TEXT,
    EVENT_TAILS,
    FINAL_EXTS,
    MAX_EXTEND_S,
    MAX_FADE_S,
    MAX_TIMING_S,
    META_FILE,
    UI_DIR,
    bad,
    cutpoint_cmd,
)
from cutpoint.ui.files import (
    _highlights,
    _inside,
    _load,
    _own,
    _ui_folder,
    _values,
    picked_filters,
    picked_inputs,
    read_meta,
    save_analysis_skip,
)
from cutpoint.ui.jobs import Job
from cutpoint.ui.projects import (
    DELETING,
    RERANKING,
    _joins_of,
    _pano_folder,
    _recorded_picks,
    _skip_after,
    clip_joins,
    current_export,
    export_joins,
    fresh_name,
    project_busy,
    project_dir,
    prune_resume_folders,
    record_export,
    resumed,
    run_state,
    validate_segments,
)
from cutpoint.ui.ranking import RERANK_BUSY_TEXT
from cutpoint.ui.routes.shared import proxy_error
from cutpoint.utils import MAX_CONTEXT_CHARS, clip_files

owned = APIRouter()
secure = APIRouter()


@owned.post("/projects/{pid}/jobs")
def start_job(request: Request, pid: str, body: dict[str, Any]):
    app, jobs, media, sources = (
        request.app,
        request.app.state.jobs,
        request.app.state.media,
        request.app.state.sources,
    )
    p = project_dir(pid)
    action = body.get("action")
    meta = read_meta(p)
    hl = _highlights(p)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    on_success: Callable[[], None] | None = None
    prepare: Callable[[], None] | None = None
    before: Callable[[Job], list[str] | None] | None = None  # a step ahead of ``args``
    on_result: Callable[[Job], None] | None = None  # after on_success, with the envelope
    written: list[Path] = []  # files ``prepare`` wrote, removed if the job does not start
    undo: list[Callable[[], None]] = []  # what ``prepare`` changed, put back the same way
    plan: list[dict[str, str]] | None = None
    if action == "analyze" and meta.get("kind") == "360":
        # The CLI's 360 flow: scan, views, analysis of the .lrv, events (no vision rating)
        style = meta.get("style")
        if not meta.get("source") or not style:
            raise bad("project has no recorded source to analyze")
        if any(body.get(k) not in (None, "", []) and body[k] is not False for k in ANALYZE_KEYS):
            raise bad("Analyze options are not available for 360 projects")
        if body.get("style") is not None:
            raise bad("the style of a 360 project cannot be changed")
        _no_options([meta["source"]], "the source")
        only = [c for c in meta.get("clips") or [] if isinstance(c, str)]
        if not all(re.fullmatch(r"\d{3}|\d{8}_\d{6}_\d{3}", c) for c in only):  # X3 clip ids
            raise bad("the clips in the project's ui/meta.json are not clip ids")
        f = insta360.parse_name(Path(meta["source"]))
        if f:  # a project made before A207 holds the bare number of its one picked file
            only = [
                x3.clip_key(f["date"], f["time"], f["num"]) if c == f["num"] else c for c in only
            ]
        live = ProjectStyle.of(meta, hl).library(style)  # a deleted style: its recorded base
        _own(p, "highlights.json")  # the analysis rewrites it
        _own(p, "pano.json")  # and the scan this
        args = cutpoint_cmd("360", "analyze", str(p), "--scan", str(_pano_folder(meta["source"])))
        args += ["--style", str(live)]
        if live != style:
            _own(p, META_FILE)  # refuse now, not when the job starts

            def prepare() -> None:
                _keep_style(p, str(live), undo)

        if only:
            args += ["--only", *only]
        plan = [{"step": s, "label": progress.LABELS[s]} for s in progress.PANO_STEPS]
    elif action == "analyze":
        ps = ProjectStyle.of(meta, hl)
        inputs, style = ps.inputs, ps.chosen
        if not inputs or not style:
            raise bad("project has no recorded source to analyze")
        _no_options(inputs, "an input")
        # The header's style button (#897): analyze again in another style (auto: detected)
        new = body.get("style")
        if new is not None and (
            not isinstance(new, str)
            or (new != AUTO and new not in ui_styles() and not _action_cam_style(new))
        ):
            raise bad(f"unknown style {new!r}")
        if new is None and (live := ps.library(style)) != style:
            new = str(live)  # a deleted style: analyze again in its base, and keep that
        if new is None and style == ps.reranked != meta.get("style"):
            # a re-rank's style: kept in meta.json, as the analysis records no re-rank and
            # the next one would go back to the chosen style (#902)
            new = str(style)
        # the options as the earlier analysis used them (inferred against its own style)
        recorded = analyze_options(p, meta, hl)
        # a re-rank (on the command line) left Auto too: its detection is kept as well
        reranked = isinstance(((hl or {}).get("settings") or {}).get("reranked"), dict)
        leaving_auto = new is not None and (style == AUTO or reranked) and new != AUTO
        style = new if new is not None else style
        _own(p, "highlights.json")  # the analysis rewrites it
        args = cutpoint_cmd("analyze", *map(str, inputs), "--style", str(style), "--out", str(p))
        # What the project was made with, else (made, or re-analyzed with these options, on
        # the command line) what the earlier analysis recorded, or the source set would change.
        st = (hl or {}).get("settings") or {}
        exclude, recursive = picked_filters(meta)
        if not picked_inputs(meta) or not (exclude or recursive):
            ex = st.get("exclude")
            exclude = [g for g in ex if isinstance(g, str)] if isinstance(ex, list) else []
            recursive = st.get("recursive") is True
        args += [f"--exclude={g}" for g in exclude]
        if recursive:
            args.append("--recursive")
        # The Analyze options; a recorded --recover is kept unless the page turns it off, or
        # the parts past a read error drop out.
        opts, skip = _analyze_args(body, recorded, style == AUTO)
        args += opts + _vision_args(st)
        if skip:  # last: its steps follow it. Saved (an older file never matches the new
            # analysis's created stamp, so without --skip there is nothing to save)
            args += ["--skip", *skip]
            _own(p, ANALYSIS_FILE)  # refuse now, not after the analysis

            def prepare() -> None:
                _ui_folder(p)  # a command-line project has no ui folder yet

            def on_success() -> None:
                save_analysis_skip(p, skip)

        if new is not None:
            _own(p, META_FILE)  # refuse now, not when the job starts
            # left Auto: the detection is kept as the "Detected earlier" copy
            kept = (hl or {}).get("detection") if leaving_auto else None

            def prepare() -> None:
                # once the job may start, so a refused one leaves the style as it was; not
                # while a run elsewhere holds the project (its analyze would stop at once)
                if project_busy(p, run_state(p, hl is not None)):
                    raise bad(BUSY_TEXT, 409)
                made: list[Path] = []
                _ui_folder(
                    p, made=made
                )  # a command-line project has no ui folder (nor meta.json) yet
                back = set_style(p, new, kept if isinstance(kept, dict) else None)

                def put_back() -> None:
                    try:
                        back()
                    finally:
                        for folder in made:  # the ui folder this created, if still empty
                            with contextlib.suppress(OSError):
                                folder.rmdir()

                undo.append(put_back)

    elif action == "edit":
        # A one-click preset: analyze (cached), clips and assemble as `edit --kind` says, in
        # the project's own style so its analysis and moments stay as they are.
        kind = body.get("kind")
        if not isinstance(kind, str) or kind not in KINDS:
            raise bad(f"kind must be one of {', '.join(KINDS)}")
        if why := preset_problem(meta, hl):
            raise bad(why)
        ps = ProjectStyle.of(meta, hl)
        inputs, style = ps.inputs, ps.chosen
        _no_options(inputs or [], "an input")
        live = ps.library(style)  # a deleted style: the edit runs in its base
        vertical = _flag(body, "vertical")
        force = _flag(body, "force")
        clips_name = "clips_vertical" if vertical else "clips"
        final_name = "final_vertical.mp4" if vertical else "final.mp4"
        _own(p, "highlights.json")  # the analysis rewrites it
        _own(p, clips_name)
        _own(p, f"{UI_DIR}/exports.json")  # refuse now, not after the edit
        if clips_name in preset_taken(p) and not force:
            raise bad(f"{clips_name} holds the clips of an earlier run; confirm to replace", 409)
        args = cutpoint_cmd(
            "edit", *map(str, inputs), "--kind", kind, "--style", str(live), "--out", str(p)
        )
        if vertical:
            args.append("--vertical")
        if force:  # the user confirmed replacing the earlier run's clips
            args.append("--force")
        args += (
            _compress_peaks(body)
            + _color_args(body)
            + _vision_args((hl or {}).get("settings") or {})
        )
        if live == AUTO:  # the analysis's --vision-detect-all, so the detection repeats
            args += _vision_detect_all_arg(_vision_detect_all(hl))
        # edit analyzes again: with the Analyze options' --skip, or it would run those steps
        if skip := _saved_skip(p, hl):
            args += ["--skip", *skip]
            _own(p, ANALYSIS_FILE)

        # a re-rank's style is kept in meta.json too: the edit's analysis records no
        # re-rank, and the next job would go back to the chosen style (#902)
        from_rerank = style == ps.reranked != meta.get("style")
        keep = live != style or from_rerank
        if keep:
            _own(p, META_FILE)  # refuse now, not when the job starts

        def prepare() -> None:
            _ui_folder(p)  # a command-line project has no ui folder for exports.json yet
            if keep:  # the style (a deleted one: the base the edit records) is kept
                det = (hl or {}).get("detection") if from_rerank else None
                _keep_style(p, str(live), undo, det if isinstance(det, dict) else None)

        def on_success() -> None:
            if not (p / clips_name / "manifest.json").is_file() or not (p / final_name).is_file():
                raise RuntimeError("edit finished without its clips and final video")
            if skip:  # the analysis edit rewrote ran with it too
                save_analysis_skip(p, skip)
            record_export(p, clips_name, created=stamp, vertical=vertical, final=final_name)

    elif action == "clips":
        picks = body.get("picks", "sequence")
        if picks not in ("sequence", "auto"):
            raise bad("picks must be 'sequence' or 'auto'")
        if picks == "auto":  # the CLI picks from highlights.json (clips --top/--ids)
            selection = _auto_picks(body, hl)
            segments = []
        else:
            if any(body.get(k) is not None for k in ("top", "ids", "max_total", "order")):
                raise bad("top, ids, max_total and order need automatic picks")
            raw = _load(p, f"{UI_DIR}/review.json")
            segments = validate_segments(
                (raw.get("segments") or []) if isinstance(raw, dict) else [],
                sources.entries(p),
                media,
            )
            if not segments:
                raise bad("choose at least one moment before exporting clips")
        extend = body.get("extend", 0)
        if (
            isinstance(extend, bool)
            or not isinstance(extend, (int, float))
            or not 0 <= extend <= MAX_EXTEND_S
        ):
            raise bad(f"extend must be 0-{MAX_EXTEND_S:g} seconds")
        extend_start = _seconds(body, "extend_start", MAX_EXTEND_S)
        extend_end = _seconds(body, "extend_end", MAX_EXTEND_S)
        vertical = _flag(body, "vertical")
        speech = _flag(body, "speech_aware")
        tails = _flag(body, "smart_tail")
        event_tails = body.get("event_tails", "auto")
        if not isinstance(event_tails, str) or event_tails not in EVENT_TAILS:
            raise bad("event_tails must be 'auto', 'on' or 'off'")
        if segments and not tails and event_tails == "on":
            # each segment's "smart_tail": false turns event tails off too
            raise bad("event tails need smart tails for the sequence's moments")
        timing = (
            _speech_args(body)
            + _opt("--tail-min", _seconds(body, "tail_min", MAX_TIMING_S))
            + _opt("--tail-max", _seconds(body, "tail_max", MAX_TIMING_S))
            + _opt("--event-window", _seconds(body, "event_window", MAX_TIMING_S, positive=True))
        )
        peaks = _compress_peaks(body) + _color_args(body)
        picture = _picture_args(body) + _opt(
            "--audio-fade", _seconds(body, "audio_fade", MAX_FADE_S)
        )
        out_name = fresh_name(p, f"clips_ui_{stamp}")
        seg_file = _own(p, f"{UI_DIR}/segments-{out_name[len('clips_ui_') :]}.json")
        _own(p, f"{UI_DIR}/exports.json")  # refuse now, not after the export

        def prepare() -> None:
            _ui_folder(p)  # a command-line project has no ui folder yet
            if not segments:  # automatic picks: no segments file
                return
            # The user's trims, in their order. Exactly them unless Speech-aware cuts or Smart
            # tails is on: then the segments leave that to the flags below.
            off = {} if speech else {"speech_aware": False}
            if not tails:
                off = {"smart_tail": False, **off}
            cache.save_json(seg_file, [{**s, **off} for s in segments])
            written.append(seg_file)

        if segments:
            args = cutpoint_cmd(
                "clips", str(p), "--segments", str(seg_file), "--out", str(p / out_name)
            )
        else:
            args = cutpoint_cmd("clips", str(p), *selection, "--out", str(p / out_name))
        if vertical:
            args.append("--vertical")
        if segments:  # an exact trim is a per-segment setting, not a flag
            args += ["--smart-tail"] if tails else []
            args += ["--speech-aware"] if speech else []
        else:  # explicit, so the style's defaults do not override what the page showed
            args.append("--smart-tail" if tails else "--no-smart-tail")
            args.append("--speech-aware" if speech else "--no-speech-aware")
        args += EVENT_TAILS[event_tails] + timing
        if extend:
            args += ["--extend", f"{float(extend):g}"]
        args += _opt("--extend-start", extend_start) + _opt("--extend-end", extend_end)
        args += picture + peaks
        if meta.get("kind") == "360":  # stitch the recordings first (pano.auto_stitch)
            chosen = {k: body.get(k) for k in ("top", "ids", "max_total")}

            def before(job: Job) -> list[str] | None:
                return stitch_first(job, p, seg_file if segments else None, chosen)

        def on_success() -> None:
            if not (p / out_name / "manifest.json").is_file():
                raise RuntimeError("clips finished without a manifest")
            record_export(p, out_name, "append", created=stamp, vertical=vertical, final=None)

    elif action == "extend":
        exp = current_export(p)
        if exp is None or not clip_files(p / exp["dir"]):
            raise bad("export clips before extending them")
        src_dir = exp["dir"]
        vertical = exp.get("vertical") is True
        pad = _opt("--start", _seconds(body, "extend_start", MAX_EXTEND_S)) + _opt(
            "--end", _seconds(body, "extend_end", MAX_EXTEND_S)
        )
        out_name = fresh_name(p, f"clips_ui_{stamp}")
        _own(p, f"{UI_DIR}/exports.json")  # refuse now, not after the export
        args = cutpoint_cmd("extend", str(p / src_dir), "--out", str(p / out_name), *pad)
        # explicit: extend snaps to speech by default, the page may have it off
        args.append("--speech-aware" if _flag(body, "speech_aware") else "--no-speech-aware")
        args += _speech_args(body)

        def on_success() -> None:
            if not (p / out_name / "manifest.json").is_file():
                raise RuntimeError("extend finished without a manifest")
            record_export(p, out_name, "append", created=stamp, vertical=vertical, final=None)

    elif action == "assemble":
        exp = current_export(p)
        if exp is None or not clip_files(p / exp["dir"]):
            raise bad("export clips before assembling")
        transition = body.get("transition", "cut")
        if transition not in ("cut", "fade"):
            raise bad("transition must be 'cut' or 'fade'")
        # the page always sends it; the default makes the audio follow the video
        audio_transition = body.get("audio_transition", default_audio_transition(transition))
        if audio_transition not in AUDIO_TRANSITIONS:
            raise bad("audio_transition must be 'cut' or 'crossfade'")
        if transition == "cut" and audio_transition == "crossfade":
            raise bad("an audio crossfade needs the fade video transition")
        exp_dir = exp["dir"]
        saved = _joins_of(_load(p, f"{UI_DIR}/review.json"))
        joins = export_joins(p / exp_dir, saved)
        files = clip_files(p / exp_dir)
        hard = [a or b for a, b in pairwise(exact_clips(p / exp_dir, files))]
        final_joins = [
            {
                "a": j["a"],
                "b": j["b"],
                # a join next to a word splice is always a cut (assemble.hard_cuts)
                "transition": "cut" if cut else j["transition"] or transition,
            }
            for j, cut in zip(clip_joins(p / exp_dir, saved), hard, strict=True)
            if j["a"] and j["b"]
        ]
        fade_dur = _seconds(body, "fade_dur", MAX_FADE_S, positive=True)
        # the saved choices, not ``joins``: the page offers Fade length from the sequence's
        # order, and a Fade pair there may not be neighbours in this export's clips
        fading = transition == "fade" or any(j["transition"] == "fade" for j in saved)
        if fade_dur is not None and not fading:
            raise bad("fade_dur needs the fade video transition")
        captions = _flag(body, "captions")
        model = body.get("caption_model")
        if model is not None and (not isinstance(model, str) or model not in TRANSCRIPT_MODELS):
            raise bad(f"caption_model must be one of {', '.join(TRANSCRIPT_MODELS)}")
        if model is not None and not captions:
            raise bad("caption_model needs captions")
        fmt = body.get("format", "mp4")
        if not isinstance(fmt, str) or fmt not in OUTPUT_FORMATS:
            raise bad(f"format must be one of {', '.join(OUTPUT_FORMATS)}")
        loudnorm = _flag(body, "loudnorm")
        if fmt != "mp4" and (loudnorm or captions):  # as the CLI refuses them (exit 2)
            raise bad(
                f"an {fmt} timeline has no audio processing or burned-in captions; "
                "turn off Normalize loudness and Burn in captions, or choose mp4"
            )
        stem = fresh_name(
            p,
            "final" + exp_dir[5:] if exp_dir.startswith("clips") else f"final_{exp_dir}",
            FINAL_EXTS,
        )
        args = cutpoint_cmd(
            "assemble",
            str(p / exp_dir),
            "--out",
            str(p / f"{stem}.{fmt}"),
            "--format",
            fmt,
            "--transition",
            transition,
            "--audio-transition",
            audio_transition,
            *_compress_peaks(body),
            *_opt("--fade-dur", fade_dur),
            *_opt("--audio-fade", _seconds(body, "audio_fade", MAX_FADE_S)),
        )
        if loudnorm:
            args.append("--loudnorm")
        if captions:
            args.append("--captions")
        if model is not None:
            args += ["--model", model]
        if joins:
            joins_file = _own(p, f"{UI_DIR}/joins-{stem}.json")
            args += ["--joins", str(joins_file)]

            def prepare() -> None:
                _ui_folder(p)  # a command-line project has no ui folder yet
                cache.save_json(joins_file, {"joins": joins})
                written.append(joins_file)

        def on_success() -> None:
            # a timeline export joins and encodes nothing: its timeline is the result
            if not (p / f"{stem}.{fmt}").is_file():
                what = "a final video" if fmt == "mp4" else f"its {fmt} timeline"
                raise RuntimeError(f"assemble finished without {what}")
            record_export(p, exp_dir, "update", final=f"{stem}.{fmt}", final_joins=final_joins)

    elif action in ("crashes", "events"):
        # The analysis refined after the fact, one CLI command each: "Describe the footage"
        # is `crashes --context` (profiles, redoes the crash detection, re-ranks), "Look
        # closer with vision" is `events` (the VLM layer, re-ranks).
        if hl is None:
            raise bad("analyze the footage first")
        if meta.get("kind") == "360":
            raise bad(f"{action} does not apply to 360 projects")
        if why := events_unavailable():
            raise bad(why, 409)
        _own(p, "highlights.json")  # the re-rank rewrites it
        if action == "crashes":
            context = body.get("context")
            if not isinstance(context, str) or not context.strip():
                raise bad("describe the footage first")
            if len(context.strip()) > MAX_CONTEXT_CHARS:
                raise bad(f"the description is at most {MAX_CONTEXT_CHARS} characters")
            if not crash_step_runs(p, meta, hl):
                raise bad("crash detection does not run for this footage")
            _own(p, "activity.json")
            _own(p, "crashes.json")
            args = cutpoint_cmd("crashes", str(p), f"--context={context.strip()}")
        else:
            _own(p, "events.json")
            args = cutpoint_cmd("events", str(p))

        def on_result(job: Job) -> None:
            _note_warnings(job)

    elif action == "resume":
        # cutpoint resume replays the stopped run of progress.json (#109) from its own cwd
        rec = runstate.read(p) if _inside(p, runstate.FILE) else None
        run = run_state(p, hl is not None)
        if rec is None or run is None or not run["resumable"]:
            raise bad("this project has no stopped run to resume", 409)
        skips = run["command"] in ("analyze", "edit") and _skip_after(
            [str(a) for a in rec.get("argv") or []]
        )
        if skips:
            _own(p, ANALYSIS_FILE)  # resumed() saves the --skip: refuse now, not after
        if run["command"] in ("clips", "edit"):
            _own(p, f"{UI_DIR}/exports.json")  # refuse now, not after the run
        if skips or run["command"] in ("clips", "edit"):

            def prepare() -> None:
                _ui_folder(p)  # as the edit's: a ui file refuses the job before it starts

        if run["command"] == "clips" and meta.get("kind") == "360":
            # stitch the recordings first, as the export it continues did
            seg_file, chosen = _recorded_picks(rec)

            def before(job: Job) -> list[str] | None:
                return stitch_first(job, p, seg_file, chosen)

        if run["command"] != "clips":
            _own(p, "highlights.json")  # the analysis rewrites it
        if run["command"] == "360 analyze":
            plan = [{"step": s, "label": progress.LABELS[s]} for s in progress.PANO_STEPS]
        args = cutpoint_cmd("resume", str(p))

        def on_result(job: Job) -> None:
            outputs = (job.result or {}).get("outputs") or {}
            if outputs.get("resumed") is True:  # not when the record had finished meanwhile
                resumed(p, rec, outputs, stamp)

    else:
        raise bad(
            "action must be 'analyze', 'edit', 'clips', 'extend', 'assemble', 'crashes', "
            "'events' or 'resume'"
        )

    def same_folder() -> None:
        # Under the jobs lock, like the settings change: a folder changed since ``p`` was
        # resolved would run the job against the old one.
        if projects_dir().resolve() != p.resolve().parent:
            raise bad("the projects folder changed; reload the page", 409)
        if p in RERANKING:  # its highlights.json is being replaced now
            raise bad(RERANK_BUSY_TEXT, 409)
        if p in DELETING:  # resolved before the delete began; its files are being moved
            raise bad("this project is being deleted", 409)
        if not p.is_dir():  # deleted after ``p`` was resolved
            raise bad("unknown project", 404)
        if prepare:
            prepare()  # only once the job may start: a refused one leaves no file behind

    # --json: the envelope's error.code tells a busy project apart (Jobs._busy) and a resume
    # names what it wrote. The jobs run in the projects folder, not where the server started.
    if action in ("analyze", "edit") and meta.get("kind") != "360":
        needs_proxy = _subscription_roles(args, next(iter(_values(args, "--style")), style))
    elif action == "resume" and run["command"] in ("analyze", "edit"):
        argv = [str(a) for a in rec.get("argv") or []]
        # no --style: auto (edit: its kind's default, funny for funny-moments)
        kind_arg = next(iter(_values(argv, "--kind")), "highlights")
        default = KINDS[kind_arg].style if run["command"] == "edit" and kind_arg in KINDS else AUTO
        style_arg = next(iter(_values(argv, "--style")), default)
        needs_proxy = _subscription_roles(argv, style_arg)
    elif action in ("crashes", "events") or (
        action == "resume" and run["command"] in ("crashes", "events")
    ):  # the profile's and the VLM layer's Vision model
        try:
            needs_proxy = ai_providers.is_subscription(vision_assistant_settings().provider)
        except (ValueError, TypeError, AttributeError):  # an invalid config.json: the job says
            needs_proxy = False
    else:
        needs_proxy = False
    if needs_proxy and before is None:

        def before(job: Job) -> list[str] | None:
            # a Vision or dialogue role on a subscription needs the proxy (#832); the step
            # is skipped with its usual warning when it does not start
            if why := proxy_error(app, logged_in_only=True):
                job.note("WARNING", why)
            return None

    args = [*args, "--json"]

    def put_back_all() -> None:  # under the jobs lock, so no newer write is overwritten
        for back in undo:
            with contextlib.suppress(OSError):
                back()

    try:
        return jobs.start(
            pid,
            action,
            args,
            on_success,
            same_folder,
            plan,
            before,
            cwd=p.parent,
            on_result=on_result,
            on_end=(lambda: prune_resume_folders(p))
            if action in ("clips", "extend", "resume")
            else None,
            on_refused=put_back_all,
        ).view()
    except HTTPException:  # for example the command could not be started
        for f in written:
            with contextlib.suppress(OSError):
                f.unlink(missing_ok=True)
        raise


@secure.get("/jobs")
def list_jobs(request: Request):
    """The kept jobs, oldest first. Only the running job and the newest carry their log: the
    page shows the log of the running job, else of the newest, and polls this every second
    while a job runs (A744: 50 kept logs of 300 lines each were sent every time)."""
    jobs = request.app.state.jobs
    kept = list(jobs.jobs.values())
    return {
        "jobs": [
            j.view(logs=j.status == "running" or i == len(kept) - 1) for i, j in enumerate(kept)
        ]
    }


@secure.post("/jobs/{job_id}/cancel")
def cancel_job(request: Request, job_id: str):
    jobs = request.app.state.jobs
    return jobs.cancel(job_id).view()
