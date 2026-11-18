"""``cutpoint ui``: local browser workspace for the standard-video highlight workflow.

A loopback-only FastAPI app that serves ``cutpoint/web`` and a small JSON
API over the projects folder. Heavy work (analyze, clips, extend, assemble, resume) runs as
ordinary ``python -m cutpoint ... --json`` child processes in the projects folder, one at a time,
so the browser and the CLI share one implementation. Source videos are only ever read.

Trust boundaries:

* Every request needs a loopback ``Host`` (and ``Origin``, when sent) on the bound port.
* Every ``/api`` call needs the per-run token (``X-Cutpoint-Token``), which only the URL printed at
  startup carries (``#t=<token>``; there is no way to fetch it from the server). Media/output URLs
  carry ``?t=`` because ``<video>`` cannot send headers: a read-only media token derived from it
  (:func:`media_token`), which only the media routes accept, so a copied video or download
  address grants no other call. A request body over :data:`MAX_BODY_BYTES` is refused.
* The page's Content-Security-Policy (:func:`page_policy`) allows only its own files, inline
  styles, ``data:`` images and the inline theme script of ``index.html`` by its hash.
* Source stills (``.../sources/<id>/frame?at=<seconds>``) are media URLs too; they are cached in
  the project's ``ui/frames`` folder under a name built from the source's content key.
* The audio envelope of a source (``.../sources/<id>/envelope``, a JSON call) is cached in ``ui/envelope``
  under the same content key; the source is always one of the project's, by id.
* Director previews of 360 moments (``.../sources/<id>/director/<name>``) are media URLs too:
  renders in the project's ``ui/pano`` folder, made from a clip's ``.lrv`` in ``pano.json`` along
  the camera path ``pano.candidates.preview_path`` gives (``POST .../director/<candidate>`` asks
  for one).
* A project id is a direct child folder of the projects folder. A project's media are its
  recorded source files, and only those under the source the user picked for it (or, for projects
  made on the command line, the inputs of its analysis). For a project the page made,
  ``highlights.json`` alone never grants access to a file; for a command-line project both the
  inputs and the source list come from it, so whoever can write that file into the projects
  folder can have any video on disk served (the trust of local file access).
* Resume (``GET .../resume``, the ``resume`` job) replays the command line the project's
  ``progress.json`` recorded, from its recorded working folder (``cutpoint resume``): a
  recorded ``--force`` or an ``--out`` outside the project too. The page shows that command
  line in its confirm dialog first; the server never changes it, and a clips folder it wrote
  outside the project is never listed as an export.
* "Show folder" (``.../reveal?target=``) opens, in the file manager and without a shell, one of
  four folders the server resolves from the project: its own sources (the default), the clips
  folder of its latest export, the project folder holding its assembled final, or the project
  folder itself (AI Chat's Open project folder, next to a reply that names a project file). The
  request names only the target, never a path.
* Choosing a source or folder (``POST /api/pick?kind=file|folder``) opens the machine's own open
  dialog (Windows: Tk's system dialog; macOS: ``osascript``; Linux: zenity, else kdialog) in a
  child process without a shell, one at a time; the server never lists folders itself. With no
  dialog (no display, Docker) the page hides those buttons and the path is typed or pasted.
* Settings (``GET``/``PUT /api/settings``) change only ``projects_dir``, ``vision``,
  ``visual_index`` and the assistant's ``provider``, ``model`` and ``url`` in ``config.json``. The projects folder never changes while a job runs or when
  ``CUTPOINT_PROJECTS_DIR`` sets it; existing projects are not moved. Every analysis the page
  starts (first or again) passes ``--vision`` or ``--no-vision`` as ``vision`` says when it
  starts (neither while ``config.json`` has no ``vision`` key: the CLI default asks the vision
  model about unsure sources and camera-tagged sources without telemetry); the page has no per-run vision box. The same goes for
  ``--visual-index`` or ``--no-visual-index`` as ``visual_index`` says. The vision box rates
  frames only: ``vision: false`` in a PUT removes the key (the default) and never writes false;
  a hand-set ``"vision": false`` stays until then.
  While ``config.json`` is unreadable or holds a value Cutpoint rejects, the project routes answer
  409 naming the file and the error; session, pick and settings keep working.
* The assistant (``/api/projects/<id>/assistant/...``) takes the token header only, never
  ``?t=``. It only proposes sequence changes, which the page applies to its working sequence; it
  writes only ``ui/assistant/chat.json``, ``ui/assistant/history.json`` and the cached timeline
  sheets in ``ui/assistant/sheets``, never ``review.json``, and runs no commands (frame grabs
  aside). A ranking in another library style (#902) is previewed without a write
  (``POST /api/projects/<id>/ranking/preview``); only ``.../ranking/apply``, which the page
  sends when the user clicks to use it, replaces ``highlights.json`` (the one before kept as
  ``highlights.before-rerank.json``) and the chosen style in ``ui/meta.json``, in this process,
  and only when ``highlights.json`` is still the one previewed (409 ``stale`` otherwise). Its
  CUDA check runs in a child process: the UI process never imports torch. Project text (labels, transcripts, file names, never
  paths; only a model on this machine is given the project folder's path) reaches the model only
  inside marked data blocks. Its GET routes only read.
* A hosted assistant provider (any whose host is not this machine) gets no project text until
  the user agrees once for it (``POST /api/assistant/consent``, an ``assistant-v3:`` key kept in
  ``config.json``); nor is its model list asked for (``GET /api/assistant/models``) before that
  and without a key. That consent also covers the whole transcripts ``cutpoint analyze`` sends to
  a hosted chat model for its dialogue reader (:mod:`cutpoint.events.dialogue`). The Vision role's
  hosted provider has its own consent (``role: vision``, a
  ``vision-v2:`` key): agreeing for one role never covers the other, and an unprefixed key from an
  older version covers neither.
  Video frames (``view_moment``, :mod:`cutpoint.assistant.watch`) go to a hosted chat provider
  only once the user allowed that provider and host for that project
  (``POST /api/projects/<id>/assistant/frames-consent``, a ``frames-v1:`` key kept per project
  folder in ``assistant.frames_consent`` of ``config.json``); a provider on this machine needs
  none. The sheets are cached in ``ui/assistant/sheets``.
  API keys come from the environment or the system keyring (``POST /api/assistant/key``,
  ``.../key/clear``); no route returns one, only where it comes from. ``cutpoint ui`` takes the
  key variables out of its environment at startup, jobs get an environment without them, and
  log records have keys redacted (:mod:`cutpoint.assistant.keys`).
"""

from __future__ import annotations

from cutpoint import paths
from cutpoint.config import projects_dir, remap_source, update_config
from cutpoint.encode import BT709_TAGS
from cutpoint.ffmpeg_utils import VideoEncoder, grab_images, probe, video_encoder
from cutpoint.segments import MAX_SEGMENTS
from cutpoint.sources import TRANSCRIPT_MODELS
from cutpoint.styles import get_style
from cutpoint.ui.analysis import (
    REFINE_WARNINGS,
    _analysis_inputs,
    _auto_picks,
    _note_warnings,
    _resolved_style,
    _style_crash,
    _subscription_roles,
    _vision_detect_all,
    analyze_options,
    detected_view,
    events_unavailable,
    new_project,
    stitch_first,
    ui_styles,
)
from cutpoint.ui.app import _Hangup, create_app, listen, media_token, quiet_resets, run
from cutpoint.ui.common import (
    ANALYSIS_FILE,
    BUSY_TEXT,
    CLIP_ORDERS,
    ENVELOPE_STALE_S,
    ENVELOPE_TIMEOUT_S,
    ENVELOPE_VERSION,
    EVENT_TAILS,
    FRAME_CACHE_MAX,
    FRAME_MAX_AT,
    MAX_BODY_BYTES,
    MAX_EXTEND_S,
    MAX_FADE_S,
    MAX_TIMING_S,
    MAX_TOTAL_S,
    MEDIA_LOOKUP_S,
    META_FILE,
    SOURCES_CACHE_MAX,
    SOURCES_CACHE_S,
    STYLE_FOR,
    UI_DIR,
    WEB_DIR,
    bad,
    cutpoint_cmd,
    job_env,
    logger,
    spawn,
)
from cutpoint.ui.files import _ui_folder, read_meta
from cutpoint.ui.jobs import Job, Jobs, parse_progress
from cutpoint.ui.media import (
    PROXY_PRUNE_GRACE_S,
    PROXY_WINDOWS_QUEUED,
    Media,
    ProxyWindow,
    _covers,
    _exit_reason,
    fragmented_mp4,
    proxy_args,
)
from cutpoint.ui.pano_previews import PANO_PREVIEW_W, PanoPreviews, _file_ident, _pano_reasons
from cutpoint.ui.picker import Picker, folder_opener, path_picker, pick_start
from cutpoint.ui.projects import (
    _REMOTE,
    DELETING,
    RERANKING,
    MediaLookups,
    Sources,
    _crash_events,
    clip_joins,
    export_joins,
    fresh_name,
    moment_id,
    moments_view,
    on_network,
    project_busy,
    project_dir,
    prune_resume_folders,
    run_state,
    source_entries,
    source_problem,
    validate_segments,
)
from cutpoint.ui.ranking import (
    CUDA_PROBE_S,
    PREVIEW_TOP,
    CudaProbe,
    PlainJSON,
    RankingHook,
    RankingRefused,
    cuda_probe,
    ranking_key,
    ranking_preview,
    ranking_swap,
)
from cutpoint.ui.settings import (
    LOGIN_TIMEOUT_S,
    _vision_provider_from_data,
    assistant_view,
    check_assistant,
    check_projects_dir,
    consent_view,
    key_status,
    settings_view,
)
from cutpoint.utils import find_videos
