// cutpoint editing workspace: ranked moments, trim preview, sequence, export and assemble.
import { api, startSession, whenExpired } from './api.js';
import { initNewProject, pickPath } from './picker.js';
import { $, AI_OTHER, aiModelChoices, baseName, colorNote, filtersOn, fmt, fmtLen, h, momentMatches, naturalCompare, NO_FILTERS, parseTime, positionCompare, round2, spanText, svg } from './util.js';

const MIN_CLIP = 0.1; // seconds
const MARGIN = 10; // seconds: the least rail context around a moment; playback and trims use the whole source
const MIN_VIEW = 2; // seconds: the rail's deepest zoom
const LONG_CLIP = 600; // seconds: longer clips get a note under the trim fields
const LABEL_MAX = 200; // characters; ui.py validate_segments rejects longer labels
const THUMB_W = 160; // pixels requested for sequence thumbnails
const FILM_TILE = 78; // pixels: the rail's filmstrip holds about one keyframe still per this width (#261)
const FILM_STEP = 0.25; // seconds: filmstrip stills sit on multiples of this times a power of 2, so pans and 2x zooms reuse them
const FILM_EXACT = 1; // seconds: a finer grid asks for exact stills; keyframe ones would repeat the same keyframe
const FILM_WAIT = 150; // ms the view stays still before new filmstrip stills are asked for
const FILM_FAILS = 3; // failed stills after which a source gets no filmstrip this session
const JOB_NAMES = { analyze: 'Analysis', edit: 'One-click edit', clips: 'Clip export', extend: 'Export extension', assemble: 'Assembly' };
const live = (status) => status === 'running' || status === 'cancelling'; // the job still holds the server
const SIDE_KEY = 'cutpoint.sideWidth'; // localStorage: the sequence column width the user dragged to
const SIDE_MIN = 290; // pixels; the column's CSS default is 290-380
const SIDE_MAX = 640;
const MOM_KEY = 'cutpoint.momentsWidth'; // localStorage: the moments column width the user dragged to
const MOM_MIN = 260; // pixels; the column's CSS default is 260-340
const MOM_MAX = 640;
const STAGE_MIN = 360; // pixels the preview keeps when either column widens (app.css: the stage track)
const SPLIT_STEP = 16; // pixels per arrow key on a column splitter; Shift moves four steps
const VOL_KEY = 'cutpoint.volume'; // localStorage: {volume: 0-1, muted} for every preview video
const NERD_KEY = 'cutpoint.nerdStats'; // localStorage: "1" while Stats for Nerds is on (off by default)
const ADV_KEY = 'cutpoint.advancedExport'; // localStorage: "1" while Show advanced export options is on (off by default)
const THEME_KEY = 'cutpoint.theme'; // localStorage: "light" or "dark"; absent for Blue, the default. index.html reads it too
const THEMES = ['blue', 'light', 'dark'];
const TAB_KEY = 'cutpoint.sideTab'; // localStorage: the sequence column's tab; absent for Sequence, the default
const ORDER_KEY = 'cutpoint.momentOrder'; // localStorage: "position" for video order; absent for Score, the default
const ORDER_NAMES = { score: 'score', position: 'position in video' }; // the moments list's orders (#469)
// localStorage: the moments list's quick filters the user set ({seq, ready, must}); absent for none. The source file is per project
const FILTERS_KEY = 'cutpoint.momentFilters';
// localStorage: the Export & Assembly Options the user changed, by option ({link: false, ...}); the rest keep their defaults
const OPTS_KEY = 'cutpoint.exportOptions';
// Export & Assembly Options for a viewer who changed none; Compress peaks starts from config.json's
// compress_peaks when it sets one, else Light. Speech-aware cuts and Smart tails were one Smart edges
// box before: a stored smart_edges is the default of both.
const OPT_DEFAULTS = {
  vertical: false, captions: false, speech_aware: true, smart_tail: true, loudnorm: true, link: true,
  transition: 'cut', audio_transition: 'cut', compress_peaks: 'light',
  picks: 'sequence', order: '', event_tails: 'auto', kind: 'highlights', crop_mode: '', caption_model: '',
};
// The Export tab's number and text fields, by the option they send; blank sends nothing (the CLI's default)
const OPT_FIELDS = {
  top: '#optTop', max_total: '#optMaxTotal', ids: '#optIds', speech_max_extend: '#optSpeechExtend',
  speech_max_lead: '#optSpeechLead', tail_min: '#optTailMin', tail_max: '#optTailMax',
  event_window: '#optEventWindow', extend_start: '#optExtendStart', extend_end: '#optExtendEnd',
  crop: '#optCropBox', size: '#optSize', audio_fade: '#optAudioFade', fade_dur: '#optFadeDur',
  final_audio_fade: '#optFinalAudioFade', // sent as Assemble's audio_fade (assembleOptions)
};
const TEXT_FIELDS = new Set(['crop', 'size']); // OPT_FIELDS sent as typed, not as numbers
// The Export tab's selects that keep their choice, by their storage key; '' (Default) sends nothing
const OPT_SELECTS = { order: '#optOrder', crop_mode: '#optCrop', caption_model: '#optCaptionModel' };
const PEAK_STOPS = ['off', 'light', 'strong']; // the Compress Audio Peaks slider's stops, left to right
const PEAK_NAMES = { off: 'None', light: 'Light', strong: 'Max' };
const COLOR_STOPS = ['off', 'auto', 'underwater']; // the Color slider's stops, left to right (--color)
const COLOR_NAMES = { off: 'Off', auto: 'Auto', underwater: 'Underwater' };
const LINKED_AUDIO = { cut: 'cut', fade: 'crossfade' }; // Link Transitions: the audio transition of each video one
const FRAME = 0.04; // seconds: "Play sequence" moves on this close to a clip's end
const GAP = 0.05; // seconds: a clip starting this close to the previous one's end plays on without a cut
// The camera path flags of a 360 moment (#517, cutpoint.pano.candidates.FLAGS) as short badge words
const PANO_FLAGS = { sky: 'Sky', whip: 'Whip pans', spin: 'Spinning', cuts: 'Many cuts', wearer: 'Wearer', off_target: 'Off target', qa_fail: 'QA fail' };

const S = {
  styles: [],
  canReveal: false, // the server can open a folder in this machine's file manager
  canPick: false, // the server can show this machine's own open dialog
  savedOpts: {}, // the Export & Assembly Options this browser stored (initExportOptions)
  colorDefault: 'off', // config.json's color (GET /api/settings): Color's default when this browser saved none
  projects: [],
  id: null,
  project: null,
  segments: [], // working sequence (may hold unsaved edits)
  savedKey: '[]',
  drafts: {}, // trims of moments not (yet) in the sequence; whole segments for removed non-moments; ``at``: index a removed clip had
  focus: null,
  filter: '',
  order: 'score', // the moments list's order: 'score' (rank #1 first) or 'position' (initMomentOrder)
  filters: { ...NO_FILTERS }, // the moments list's quick filters (initMomentFilters); util.js momentMatches applies them
  jobs: [],
  active: null, // the running job, if any
  tracked: new Map(), // job id -> last seen status
  logOpen: false,
  saving: false,
  selected: new Set(), // sequence ids ticked for batch actions
  nerdStats: false, // Settings > Stats for Nerds: the scoring overlay on the preview
  nerdStatsWas: false, // nerdStats when Settings opened, restored by Cancel or Escape
  advExport: false, // Settings > Show advanced export options: the Export tab's .adv controls
  advExportWas: false, // advExport when Settings opened, restored by Cancel or Escape
  themeWas: 'blue', // Settings > Theme when Settings opened, restored by Cancel or Escape
  renaming: null, // sequence id whose label is being edited
  renameDraft: '',
  renameErr: '',
  expired: false, // the server rejected the session token: nothing polls or retries any more
  stepStates: null, // {job, states: {step: state}}: the analysis steps last announced (renderSteps)
  stepSince: {}, // "job/step" -> ms the UI first saw the step running (elapsed time under reduced motion)
  stepTimer: null, // ticks that elapsed time once a second
  view: null, // {id, lo, hi}: the part of the source the trim rail shows, separate from the trim
  seqPlay: null, // {order: [ids], i, wait}: "Play sequence" runs; order is fixed at its start, wait: the clip's video is not ready
  undo: [], // {segments, drafts} before each edit, manual or applied from AI Chat, newest last (at most UNDO_MAX)
  redo: [],
  undoTag: null, // {tag, at}: the last edit's kind, so a held key or a drag is one undo step
  // AI Chat (#346): the change previewed in the Sequence tab, the moments list and the rail until Apply
  // or Discard: {id, summary, after, diff, base_key, seqKey}; seqKey: the sequence it was made on (baseOf its diff)
  proposal: null,
  chat: null, // {pid, provider, consent, turns: [{id, user, status, error, events}], error}: the open project's chat
  ai: null, // the assistant settings Settings shows (GET /api/settings' assistant); never a key
  turn: null, // {id, next}: the turn being polled
  sending: false,
  applying: false,
  pendingProposal: null, // a proposal that came in while Apply waited: previewed once Apply is done
  shown: null, // Set of the moment ids the assistant showed (show_moments), highlighted in the list
  history: [], // the applied assistant changes (GET history), oldest first
};

// Two stage players: the visible one (``v``) plays, the other preloads the next clip of "Play sequence".
const els = [$('#video'), $('#video2')];
let v = els[0];
const spare = () => (v === els[0] ? els[1] : els[0]);
const seek = $('#seek');
let openDialog;
let showTab = () => {}; // showTab('sequence', focus): select a tab of the sequence column (initSideTabs)
let pendingSeek = null;
let cued = null; // {id, start, vt}: the clip the spare player holds, at video time vt
let seekWant = null; // {t, fast}: the latest scrub target while a seek is in flight
let wasIn = true; // the playhead was on the rail at the last frame or view change (tick follows it off the edge)

// ---------- state helpers ----------

const pick = (s) => ({ id: s.id, source: s.source, start: s.start, end: s.end, label: s.label });
const key = () => JSON.stringify(S.segments.map(pick));
const dirty = () => key() !== S.savedKey;
const cands = () => S.project?.highlights?.candidates ?? [];
const srcOf = (path) => S.project?.sources?.find((s) => s.path === path);
const sourceDur = (path) =>
  S.project?.highlights?.sources?.find((s) => s.path === path)?.duration ?? srcOf(path)?.duration;

function current() {
  const id = S.focus;
  if (!id) return null;
  const seg = S.segments.find((s) => s.id === id);
  const cand = cands().find((c) => c.id === id);
  const base = seg ?? (cand ? { ...cand, ...S.drafts[id] } : S.drafts[id]?.source && S.drafts[id]);
  if (!base) return null;
  return { id, seg, cand, source: base.source, start: base.start, end: base.end, label: base.label || id };
}

/** A window proxy ``w`` ({lo, hi}) plays source time ``t``: not in its last frame, unless it ends with the source. */
const covers = (w, t, dur) => t >= w.lo && (t < w.hi - FRAME || (!!dur && w.hi >= dur - FRAME && t <= w.hi));

/**
 * The file the preview plays source time ``t`` of ``src`` from: the media url, the source time at
 * its 0 (``offset``) and the source range it covers. The whole source once it plays (as is or
 * from its proxy); before that a ready window proxy covering ``t`` (#323), else null.
 */
function playable(src, t, raw = false) {
  const d = !raw && directed(src, t);
  if (d) return d;
  const dur = sourceDur(src.path) ?? (v.dataset.url === src.url && Number.isFinite(v.duration) ? v.duration : null);
  if (src.status === 'ready') return { url: src.url, offset: 0, lo: 0, hi: dur };
  const w = (src.windows ?? []).filter((x) => t != null && covers(x, t, dur)).sort((a, b) => b.hi - a.hi)[0];
  return w ? { url: w.url, offset: w.lo, lo: w.lo, hi: w.hi } : null;
}
/**
 * The director preview of the focused 360 moment (#517) when it is rendered and plays source time
 * ``t``: a flat view along the camera path the server's ``preview_path`` gave (the one 360 analysis
 * planned, one planned now, or a level forward view), in place of the raw fisheye, played like a
 * window proxy. Not during "Play sequence", which plays the raw preview.
 */
/** Director preview urls the browser could not play (a 404 after a re-plan): never tried again. */
const directorBad = new Set();
function directed(src, t, c = current()) {
  const d = c?.cand?.pano?.preview;
  if (S.seqPlay || d?.state !== 'ready' || directorBad.has(d.url) || !src || t == null || c.cand.source !== src.path) return null;
  return covers(d, t, sourceDur(src.path)) ? { url: d.url, offset: d.lo, lo: d.lo, hi: d.hi, director: c.id } : null;
}
/** The window ``w`` plays all of ``seg`` ({start, end}); one ending with the source covers an end past its centisecond-floored ``hi``. */
const coversSeg = (w, seg, dur) =>
  covers(w, seg.start, dur) && (covers(w, seg.end, dur) || (!!dur && w.hi >= dur - FRAME && seg.end <= dur));
/**
 * The file "Play sequence" plays ``seg`` of ``src`` from (call once ``readyFor``): the stage player's
 * own when it covers all of it, so contiguous clips play on without a swap, else ``playable``'s.
 */
function fileFor(src, seg) {
  const r = loaded();
  if (r && ofSource(src) && !v.dataset.director) { // a director preview never plays a sequence
    const dur = sourceDur(src.path);
    if (r.hi == null ? v.dataset.url === src.url : coversSeg(r, seg, dur))
      return { url: v.dataset.url, offset: r.lo, lo: r.lo, hi: r.hi ?? dur };
  }
  return playable(src, seg.start);
}
/** ``src`` plays all of ``seg``: as a whole (as is or from its proxy), or from one ready window proxy. */
function readyFor(src, seg) {
  if (src?.status === 'ready') return true;
  if (!src || !seg) return false;
  const dur = sourceDur(src.path);
  return (src.windows ?? []).some((w) => coversSeg(w, seg, dur));
}
const durOf = (c) => {
  const src = srcOf(c.source);
  return src?.status === 'ready' ? playable(src, 0, true).hi : sourceDur(c.source);
};
/** The source time at the stage player's 0: a window proxy starts later in its source. */
const offsetOf = () => Number(v.dataset.offset) || 0;
/** The source range the stage player's file covers; ``hi`` null: to the end of the source. Null when nothing is loaded. */
const loaded = () => (v.dataset.url ? { lo: offsetOf(), hi: v.dataset.hi ? Number(v.dataset.hi) : null } : null);
/**
 * The stage player holds a file of ``src``: its original or whole proxy, one of its window proxies,
 * or the director preview of the focused moment (#517; not during "Play sequence").
 */
const ofSource = (src) => {
  const u = v.dataset.url;
  if (!u) return false;
  const base = src.url.split('?')[0];
  if (v.dataset.director) return !S.seqPlay && v.dataset.director === S.focus && u.startsWith(`${base}/director/`);
  return u === src.url || u.startsWith(`${base}/window/`);
};
/** The stage player's file plays all of clip ``c`` ({source, start, end}): a file of its source, whole or a window covering it. */
const holdsClip = (c) => {
  const r = loaded();
  const src = srcOf(c.source);
  return !!r && !!src && ofSource(src) && (r.hi == null || coversSeg(r, c, sourceDur(c.source)));
};
const inLoaded = (t, dur) => {
  const r = loaded();
  return !!r && (r.hi == null || covers(r, t, dur));
};
/** The playhead in source time: a held scrub target or one waiting for the video (both source time), so the rail does not jump back. */
const playhead = (c = current()) => seekWant?.t ?? pendingSeek ?? (v.currentTime || 0) + (c ? offsetOf() : 0);

/** A rail view spanning at least MIN_VIEW, shifted (not cut) to stay inside the source. */
function clampView(lo, hi, dur) {
  let span = Math.max(hi - lo, MIN_VIEW);
  if (dur) span = Math.min(span, dur);
  lo = Math.max(0, dur ? Math.min(lo, dur - span) : lo);
  return { lo, hi: lo + span };
}

/** The default view: the clip and its analyzed range, padded by half the clip (at least MARGIN). */
function fitView(c) {
  const dur = durOf(c);
  const pad = Math.max(MARGIN, 0.5 * (c.end - c.start));
  const lo = Math.max(0, Math.min(c.start, c.cand?.start ?? c.start) - pad);
  const hi = Math.max(c.end, c.cand?.end ?? c.end) + pad;
  return clampView(lo, dur ? Math.min(dur, hi) : hi, dur);
}

const hms = (t) => {
  const s = Math.round(t);
  return `${Math.floor(s / 3600)}:${String(Math.floor(s / 60) % 60).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
};

function validate(s, e, dur) {
  if (!Number.isFinite(s) || !Number.isFinite(e)) return 'Enter a time like 83.5 or 1:23.5.';
  if (s < 0 || (dur && e > dur + 1e-6))
    return `Keep both times inside the source, 0:00.00 to ${fmt(dur)}.`;
  if (e - s < MIN_CLIP - 1e-6) return `A clip must be at least ${MIN_CLIP} s long.`;
  return '';
}

const banner = (msg) => {
  if (S.expired) return; // the expired-session message stays, announced once
  const b = $('#banner');
  b.textContent = msg || '';
  b.hidden = !msg;
};
const fail = (e) => banner(e.message || String(e));

// A restarted server issues a new token: say so once and stop every timer, so the page neither
// hammers the server nor waits on "Loading video…" forever.
whenExpired((e) => {
  banner(e.message);
  S.expired = true;
  clearTimeout(S.prepTimer);
  clearTimeout(S.partTimer);
  clearTimeout(S.jobTimer);
  clearTimeout(S.seqTimer);
  clearTimeout(S.chatTimer);
  clearTimeout(S.panoTimer);
  clearInterval(S.stepTimer);
  S.stepTimer = null;
  seqStop('');
  drop(''); // the banner already says why; a second status line would be announced too
  const st = $('#jobStatus'); // an earlier failed poll may have left "Can't reach the server…"
  st.textContent = '';
  st.dataset.status = '';
});

function preserveFocus(fn) {
  const a = document.activeElement;
  const id = a?.closest?.('[data-id]')?.dataset.id;
  const act = a?.dataset?.act;
  const listId = a?.closest?.('ol')?.id;
  fn();
  if (!id || !act || !listId) return;
  const li = $(`#${listId} [data-id="${CSS.escape(id)}"]`);
  const btn = li?.querySelector(`[data-act="${act}"]`);
  (btn && !btn.disabled ? btn : (li?.querySelector('button:not(:disabled)') ?? $(`#${listId} button:not(:disabled)`)))?.focus();
}

const setEmpty = (el, ...kids) => {
  el.replaceChildren(...kids);
  el.hidden = !kids.length;
};

// ---------- projects ----------

async function loadProjects() {
  const sel = $('#projectSelect');
  try {
    S.projects = (await api('/api/projects')).projects;
  } catch (e) {
    // e.g. an invalid config.json (409): creating a project would fail the same way until a load
    // succeeds. Settings stays usable to fix it.
    S.projects = [];
    sel.replaceChildren(h('option', { value: '', text: 'Projects unavailable', disabled: true }));
    sel.value = '';
    $('#newBtn').disabled = true;
    throw e;
  }
  $('#newBtn').disabled = false;
  sel.replaceChildren(
    ...(S.projects.length
      ? S.projects.map((p) => h('option', { value: p.id, text: p.name }))
      : [h('option', { value: '', text: 'No projects yet', disabled: true })]),
  );
  sel.value = S.id ?? '';
}

async function openProject(id) {
  if (id !== S.id && dirty() && !confirm('Discard the unsaved changes to this sequence?')) {
    $('#projectSelect').value = S.id ?? '';
    return;
  }
  try {
    adopt(await api(`/api/projects/${encodeURIComponent(id)}`), id !== S.id);
    banner('');
  } catch (e) {
    fail(e);
  }
}

async function refreshProject() {
  const id = S.id;
  if (!id) return;
  try {
    const p = await api(`/api/projects/${encodeURIComponent(id)}`);
    if (p.id === S.id) adopt(p, false);
  } catch (e) {
    fail(e);
  }
}

/** Forget the per-project edit state; ``segments`` become the saved sequence. */
function resetWorkspace(segments) {
  seqStop('');
  S.drafts = {};
  S.focus = null;
  S.selected.clear();
  S.renaming = null;
  S.segments = segments;
  S.savedKey = key();
  S.undo = [];
  S.redo = [];
  S.undoTag = null;
  S.proposal = null;
  S.pendingProposal = null;
  S.shown = null;
  S.filters.source = ''; // a path of the project left
}

/** Leave the open project (its folder is no longer listed): back to the empty workspace. */
function closeProject() {
  S.id = null;
  S.project = null;
  resetWorkspace([]);
  clearTimeout(S.prepTimer);
  drop('');
  history.replaceState(null, '', location.pathname + location.search);
  renderAll(true);
  loadAssistant();
}

function adopt(p, switched) {
  S.project = p;
  if (switched) {
    S.id = p.id;
    delete $('#momentsEmpty').dataset.refocus;
    resetWorkspace(structuredClone(p.review?.segments ?? []));
    $('#projectSelect').value = p.id;
    history.replaceState(null, '', `#${encodeURIComponent(p.id)}`);
  } else if (!dirty()) {
    // Polling must never clobber unsaved edits; saved ones follow the server.
    S.segments = structuredClone(p.review?.segments ?? []);
    S.savedKey = key();
  }
  if (!current()) S.focus = S.segments[0]?.id ?? cands()[0]?.id ?? null;
  clearTimeout(S.prepTimer);
  if (switched) startColor();
  if (switched) syncTiming(); // Crop and Output size are off for 360 projects
  renderAll(switched);
  if (switched) loadAssistant();
}

// ---------- rendering: moments ----------

function renderMoments() {
  const p = S.project;
  const hl = p?.highlights;
  const list = cands();
  $('#analyzeBtn').hidden = !hl; // before analysis the empty state's own button replaces it
  $('#revealHeadBtn').hidden = !hl || !S.canReveal; // likewise its own Show folder
  $('.moments').classList.toggle('bare', !hl);
  $('#analyzeBtn').textContent = 'Analyze again';
  $('#filterWrap').hidden = !list.length;
  $('#momentsCount').textContent = hl ? `${list.length} found, ${hl.style ?? p.style} style` : '';
  const stats = hl ? p.analysis : null; // null when the length or the time is unknown
  $('#analyzeStats').hidden = !stats;
  $('#analyzeStats').textContent = stats ? `Analyzed ${spanText(stats.video_s)} of video in ${spanText(stats.took_s)}` : '';
  const empty = $('#momentsEmpty');
  // the big button had focus (disabling it while analysis runs moves focus to body): keep it in this column
  const hadFocus = empty.contains(document.activeElement) || (!!empty.dataset.refocus && document.activeElement === document.body);
  empty.classList.toggle('fresh', !!p && !hl);
  if (!p) setEmpty(empty);
  else if (!hl) renderUnanalyzed(empty, p.sources?.length ?? 0, p.source_error ? sourceErrorText(p) : '', p.kind === '360');
  else if (!list.length)
    setEmpty(empty, h('p', {}, 'Analysis found no moments. Try another style, or add footage and analyze again.'));
  else setEmpty(empty);
  if (!p || hl) {
    delete empty.dataset.n;
    delete empty.dataset.refocus;
  }

  const sources = [...new Set(list.map((c) => c.source))];
  const multi = sources.length > 1;
  // Sort and filter: the source file choice only with several sources, and only one of this project's
  const f = { ...S.filters, source: multi && sources.includes(S.filters.source) ? S.filters.source : '' };
  syncFilterMenu(sources, f, !!list.length);
  const segOf = (c) => S.segments.find((s) => s.id === c.id);
  const items = list
    .map((c, i) => ({ c, rank: i + 1 }))
    .filter(({ c }) => {
      const seg = segOf(c);
      return momentMatches({ ...c, picked: !!seg, ready: readyFor(srcOf(c.source), seg ?? { ...c, ...S.drafts[c.id] }) }, S.filter, f);
    });
  if (S.order === 'position') items.sort((a, b) => positionCompare(a.c, b.c)); // the badge keeps the score rank
  const listHadFocus = $('#candList').contains(document.activeElement);
  // A card: rank, title and Add/Remove on top (the button sits over the main button's top row: no
  // nested buttons), the times, the file with several sources, the strip, then the transcript with
  // the preview readiness at the bottom right.
  preserveFocus(() => {
    $('#candList').replaceChildren(
      ...items.map(({ c, rank }) => {
        const seg = segOf(c);
        const picked = !!seg;
        const badge = prepBadge(srcOf(c.source), seg ?? { ...c, ...S.drafts[c.id] });
        const tip = c.pano?.flags?.length ? `tip-${c.id}` : null;
        return h('li', { class: `cand${picked ? ' picked' : ''}`, 'data-id': c.id, 'aria-current': c.id === S.focus ? 'true' : null },
          h('button', { type: 'button', class: 'cand-main', 'data-act': 'focus', 'aria-describedby': tip, onclick: () => focusMoment(c.id) },
            h('span', { class: 'cand-top' },
              h('span', { class: 'rank tc', text: `#${rank}` }),
              h('span', { class: 'cand-title', text: c.label || c.cli_id || c.id }),
              c.must_include ? h('span', { class: 'tag', text: 'must include' }) : null),
            h('span', { class: 'cand-time tc' },
              // the CLI id (h001) as text, for Candidate ids and clips --ids (#487)
              c.cli_id ? [h('span', { class: 'cand-id' }, h('span', { class: 'vh', text: 'Candidate ' }), c.cli_id), ' '] : null,
              `${fmt(c.start)} to ${fmt(c.end)}`,
              h('span', { 'aria-hidden': 'true', text: ' · ' }), h('span', { class: 'vh', text: ', ' }),
              `${(c.end - c.start).toFixed(1)} s`),
            multi ? h('span', { class: 'cand-src' }, h('span', { class: 'vh', text: 'in ' }), baseName(c.source)) : null,
            flagBadges(c.pano),
            sourceStrip(c),
            c.transcript || badge
              ? h('span', { class: 'cand-foot' }, c.transcript ? h('span', { class: 'cand-text', text: c.transcript }) : null, badge)
              : null),
          h('button', {
            type: 'button', class: 'btn small', 'data-act': 'toggle',
            'aria-label': `${picked ? 'Remove' : 'Add'} ${c.label || c.cli_id || c.id} ${picked ? 'from' : 'to'} sequence`,
            onclick: () => (picked ? removeSeg(c.id) : addSeg(c.id)),
          }, picked ? 'Remove' : 'Add'),
          // the reasons of the camera path badges (#517): shown on hover and keyboard focus, read as the card's description
          tip ? h('span', { class: 'flag-tip', role: 'tooltip', id: tip, text: `Camera path looks poor: ${c.pano.reasons.map((r) => r.text).join('; ')}.` }) : null,
        );
      }),
    );
  });
  if (list.length && !items.length)
    $('#candList').replaceChildren(h('li', { class: 'empty', text: `No moments match ${filtersOn(f) ? 'these filters' : 'this filter'}.` }));
  // the focused card left the list (Add with Not in sequence, say) and no card is left to take focus
  if (listHadFocus && !$('#candList').contains(document.activeElement) && !$('#filterWrap').hidden) $('#filterBtn').focus();
  markMoments();
  if (hadFocus && !empty.contains(document.activeElement))
    ([$('#analyzeBig'), $('#candList button'), $('#analyzeBtn'), $('#revealHeadBtn')].find((b) => b && !b.hidden && !b.disabled) ?? $('#momentsTitle')).focus();
  renderBusy();
}

/**
 * A card's read-only strip (as the preview's overview): the whole source, the moment in gold and
 * the parts no ready file plays yet hatched. Empty while the source's length is unknown. The time
 * line above states the times, so it is hidden from screen readers; the hover title keeps the score.
 */
function sourceStrip(c) {
  const dur = sourceDur(c.source);
  const pct = (t) => `${((t / dur) * 100).toFixed(2)}%`;
  const at = (a, b) => {
    const lo = Math.max(0, Math.min(dur, a));
    return { style: `left:${pct(lo)};width:${pct(Math.max(0, Math.min(dur, b) - lo))}` };
  };
  // A moment at the source's end keeps its min-width (app.css), flush with the right edge.
  const lo = Math.max(0, Math.min(dur, c.start));
  const clip = `left:min(${pct(lo)}, calc(100% - 4px));width:${pct(Math.max(0, Math.min(dur, c.end) - lo))}`;
  return h('span', { class: 'cand-strip', 'aria-hidden': 'true', title: `Score ${c.score.toFixed(3)}` },
    ...(dur ? [
      h('span', { class: 'rail-gaps' }, ...sourceGaps(srcOf(c.source), dur).map(([a, b]) => h('span', at(a, b)))),
      h('span', { class: 'cand-strip-clip', style: clip }),
    ] : []));
}

/**
 * How far the preview of ``seg`` ({start, end}) is, for a source the browser plays from a proxy
 * (#323): 'ready' (the whole proxy or a ready window covers it), 'encoding' (its own window is
 * encoding, or it has none queued and this project's whole proxy is) or 'queued' (its window
 * waits, or the whole proxy does). Null for sources that play as is or are not asked for.
 */
function prepState(src, seg) {
  if (!src || (!src.proxy && !['needs-proxy', 'preparing'].includes(src.status))) return null;
  if (readyFor(src, seg)) return 'ready';
  const dur = sourceDur(src.path);
  const mine = (src.pending ?? []).filter((w) => coversSeg(w, seg, dur));
  if (mine.some((w) => w.state === 'encoding')) return 'encoding';
  if (mine.length) return 'queued';
  if (src.progress != null) return 'encoding';
  return src.status === 'preparing' ? 'queued' : null;
}

const PREP = { ready: ['check', 'Ready'], encoding: ['hourglass', 'Encoding'], queued: ['clock', 'Queued'] };

/** The readiness of a row's preview as a small icon and word (never colour alone); null when there is nothing to say. */
function prepBadge(src, seg) {
  const state = prepState(src, seg);
  if (!state) return null;
  const [name, word] = PREP[state];
  return h('span', { class: 'prep', 'data-state': state, title: `Preview ${word.toLowerCase()}` },
    icon(name), h('span', { class: 'vh', text: 'Preview ' }), word);
}

/**
 * The camera path flags of a 360 moment (#517) as small badges (words, never colour alone), or null.
 * Their reasons are the card's tooltip and description (``flag-tip``) and the preview's note.
 */
function flagBadges(pano) {
  if (!pano?.flags?.length) return null;
  return h('span', { class: 'cand-flags' },
    h('span', { class: 'vh', text: 'Camera path looks poor: ' }),
    ...pano.flags.map((f, i) => h('span', { class: 'flag', 'data-flag': f, text: PANO_FLAGS[f] ?? f },
      i < pano.flags.length - 1 ? h('span', { class: 'vh', text: ', ' }) : null)));
}

/** Folder with level bars, flanked by sound waves; decorative. Fixed 158:80 box so CSS cannot distort it. */
function folderArt() {
  const line = (x, len) => ['line', { class: 'wave', x1: x, x2: x, y1: 42 - len / 2, y2: 42 + len / 2 }];
  const waves = [[4, 8], [9, 16], [14, 26], [19, 18], [24, 32], [29, 12]];
  const el = svg('fresh-art', [
    ['path', { d: 'M42 6h20l7 8h47a4 4 0 0 1 4 4v50a4 4 0 0 1-4 4H42a4 4 0 0 1-4-4V10a4 4 0 0 1 4-4z' }],
    ['path', { d: 'M38 23h82' }],
    ['rect', { x: 63, y: 50, width: 5, height: 11, rx: 1 }],
    ['rect', { x: 72, y: 39, width: 5, height: 22, rx: 1 }],
    ['rect', { x: 81, y: 42, width: 5, height: 19, rx: 1 }],
    ['rect', { x: 90, y: 34, width: 5, height: 27, rx: 1 }],
    ...waves.map(([x, len]) => line(x, len)),
    ...waves.map(([x, len]) => line(154 - x, len)),
  ], '0 0 158 80');
  for (const [k, v] of Object.entries({ width: 158, height: 80, preserveAspectRatio: 'xMidYMid meet' })) el.setAttribute(k, v);
  return el;
}

/** Why the picked source folder lists no videos: it could not be read (``source_error``). */
const sourceErrorText = (p) =>
  `The source folder cannot be found or read (${p.source_error}). Reconnect the drive, or set a source remap in Settings.`;

/**
 * Keep focus in the empty state ``el`` while the analysis it starts disables its buttons (focus
 * would drop to the body), only until the user moves on: their next click or focus change ends it.
 */
function keepEmptyFocus(el) {
  el.dataset.refocus = '1';
  const stop = new AbortController();
  const drop = () => { delete el.dataset.refocus; stop.abort(); };
  document.addEventListener('pointerdown', drop, { capture: true, signal: stop.signal });
  document.addEventListener('focusin', drop, { capture: true, signal: stop.signal });
}

/** Project open but not analyzed: source count (or why the folder cannot be read) and the one button that starts analysis; ``pano``: a 360 project (n counts its recordings). */
function renderUnanalyzed(el, n, error, pano) {
  const state = `${n}|${error}|${pano}`;
  if (el.dataset.n === state) return; // polling re-renders; keep the button (and its focus)
  el.dataset.n = state;
  setEmpty(el,
    folderArt(),
    error
      ? h('p', { class: 'fresh-error', text: error })
      : h('p', { class: `fresh-count${n ? '' : ' none'}` },
        pano
          ? (n ? `360 footage: ${n} recording${n === 1 ? '' : 's'} found` : 'No 360 recordings found')
          : (n ? `${n} video${n === 1 ? '' : 's'} found in the sources` : 'No videos found in the sources')),
    h('h3', { text: 'No moments analyzed yet' }),
    h('p', {
      text: pano
        ? 'Analyze footage to read the 360 recordings, transcribe audio and rank the strongest moments. Clips are reframed when you export them. Vision rating is not used for 360 footage.'
        : 'Analyze footage to transcribe audio and rank the strongest moments.',
    }),
    h('button', {
      type: 'button', id: 'analyzeBig', class: 'btn primary', 'aria-describedby': 'analyzeWhy analyzeNote',
      onclick: async (e) => {
        e.currentTarget.disabled = true; // no second start while the request is in flight
        keepEmptyFocus(el);
        await startJob('analyze'); // vision: the server applies the Settings value
        if (![...S.tracked.values()].some(live)) renderBusy(); // it failed: enable again
      },
    }, 'Analyze footage'),
    // the Analyze options (#483) for the first analysis; Analyze footage alone uses none
    h('button', {
      type: 'button', id: 'analyzeOptsBtn', class: 'btn outline', 'aria-haspopup': 'dialog', hidden: pano,
      onclick: () => analyzeAgain(true),
    }, 'Analysis options…'),
    h('button', {
      type: 'button', id: 'revealBtn', class: 'btn outline', hidden: !S.canReveal,
      onclick: (e) => revealFolder(e.currentTarget),
    }, icon('folder'), 'Show folder'),
    h('p', { id: 'analyzeWhy', class: 'fresh-why', 'aria-live': 'polite' }),
    h('div', { id: 'freshSteps', class: 'steps', hidden: true }), // the running analysis's steps (renderSteps)
    h('p', { id: 'freshStepsStatus', class: 'vh', role: 'status' }),
    h('p', { id: 'analyzeNote', class: 'fresh-note', text: 'This process can take a long time on large footage.' }),
  );
}

/** Ask the server to show a project folder in the file manager: 'sources', 'clips' or 'outputs'. */
async function revealFolder(btn, target = 'sources') {
  btn.disabled = true; // one request at a time
  try {
    await api(`/api/projects/${encodeURIComponent(S.id)}/reveal?target=${target}`, { method: 'POST' });
    banner('');
  } catch (e) {
    fail(e);
  } finally {
    btn.disabled = false;
    if (document.activeElement === document.body && btn.isConnected) btn.focus(); // disabling dropped it
  }
}

/** Show moment ``id`` on the stage; during "Play sequence" a clip of it jumps there, any other moment stops it. */
function focusMoment(id) {
  const j = S.seqPlay?.order.indexOf(id) ?? -1;
  if (j >= 0 && seqSeg(j)) return seqGo(j);
  seqStop();
  showMoment(id);
}

/** ``keepVideo``: the player is already at the moment (a sequence step), so it is not sought. */
function showMoment(id, keepVideo = false) {
  S.focus = id;
  for (const li of [...$('#candList').children, ...$('#seqList').children, ...$('#removedList').children]) {
    if (li.dataset.id === id) li.setAttribute('aria-current', 'true');
    else li.removeAttribute('aria-current');
  }
  renderStage(true, keepVideo);
}

// ---------- rendering: stage ----------

function trimField(which, label) {
  const input = h('input', {
    id: `t-${which}`, class: 'tc', inputMode: 'decimal', autocomplete: 'off', spellcheck: false,
    onchange: () => {
      const ok = setRange(which, parseTime(input.value));
      if (ok) seekTo(trimSeek(which));
    },
  });
  const nudge = (d) => () => {
    const c = current();
    if (c && setRange(which, c[which] + d)) seekTo(trimSeek(which));
  };
  const key = which === 'start' ? '[' : ']';
  $(`#${which}Field`).append(
    h('label', { for: input.id, text: label }),
    input,
    h('div', { class: 'stepper' },
      h('button', { type: 'button', class: 'btn small', 'aria-label': `-0.5 s, ${label} earlier by half a second`, onclick: nudge(-0.5), text: '-0.5 s' }),
      h('button', {
        type: 'button', class: 'btn small accent', 'aria-label': `Use playhead as ${label}`, title: `Use playhead as ${label} "${key}"`,
        'aria-keyshortcuts': key, onclick: () => setRange(which, playhead()), text: 'Use playhead',
      }),
      h('button', { type: 'button', class: 'btn small', 'aria-label': `+0.5 s, ${label} later by half a second`, onclick: nudge(0.5), text: '+0.5 s' })),
  );
  return input;
}

const trimInputs = {};

/** Where a typed or nudged trim seeks: its point; during "Play sequence" a new end gets 2 s of pre-roll, so the cut is seen. */
function trimSeek(which) {
  const c = current();
  return S.seqPlay && which === 'end' ? Math.max(c.start, c.end - 2) : c[which];
}

function setTrimMsg(text, bad = false) {
  const m = $('#trimMsg');
  m.textContent = text;
  m.classList.toggle('error', bad);
}

/** ``widen``: a point that left the rail widens it to show both; the trim handles keep the zoom and pan instead. */
function setRange(which, t, widen = true) {
  const c = current();
  if (!c) return false;
  t = round2(t);
  const s = which === 'start' ? t : c.start;
  const e = which === 'end' ? t : c.end;
  const err = validate(s, e, durOf(c));
  if (err) {
    renderTrim(true);
    setTrimMsg(err, true);
    return false;
  }
  if (s !== c.start || e !== c.end) remember(`trim:${c.id}:${which}`);
  if (c.seg) {
    c.seg.start = s;
    c.seg.end = e;
    changed();
  } else {
    S.drafts[c.id] = { ...S.drafts[c.id], start: s, end: e };
    renderRemoved();
  }
  const w = S.view?.id === c.id && S.view;
  if (widen && w && (s < w.lo || e > w.hi)) { // a trim point left the rail: widen it to show both again
    const pad = 0.05 * (Math.max(w.hi, e) - Math.min(w.lo, s));
    S.view = { id: c.id, ...clampView(s < w.lo ? s - pad : w.lo, e > w.hi ? e + pad : w.hi, durOf(c)) };
  }
  renderTrim(true);
  return true;
}

function renderStage(seekToStart, keepVideo = false) {
  if (trimDrag && dragStale(trimDrag)) dropDrag(); // the reused handle loses its amber drag state at once
  const c = current();
  $('#stageBody').hidden = !c;
  $('#toggleBtn').hidden = !c;
  $('#stageMeta').textContent = c ? `${c.label} from ${baseName(c.source)}` : '';
  renderFullPrep(c);
  renderNerdStats(c);
  renderPanoNote(c);
  if (!S.project)
    setEmpty($('#stageEmpty'),
      h('p', {}, 'No project is open.'),
      h('p', { class: 'muted' }, 'Create one from a video file or a folder of footage on this computer. Source files are never changed.'),
      h('button', { type: 'button', class: 'btn primary', onclick: () => openDialog(S.styles), text: 'New project' }));
  else if (!c)
    setEmpty($('#stageEmpty'), h('p', {}, S.project.highlights ? 'Pick a moment to preview it.' : 'Analyze the footage to rank its best moments, then preview and trim them here.'));
  else setEmpty($('#stageEmpty'));
  if (!c) {
    v.pause();
    return;
  }
  if (seekToStart) S.view = null; // a new moment starts on its fit view
  const inSeq = !!c.seg;
  $('#toggleBtn').textContent = inSeq ? 'Remove from sequence' : 'Add to sequence';
  renderTranscript(c.cand);
  ensureVideo(seekToStart && !keepVideo ? c : null);
  renderTrim(!!seekToStart);
}

/**
 * Stats for Nerds (#353): the moment's score, rank, analyzed times and signals over the top-left of
 * the preview, only while the setting is on; off, the overlay is hidden and empty.
 */
function renderNerdStats(c = current()) {
  const el = $('#signals');
  const cand = S.nerdStats ? c?.cand : null;
  el.hidden = !cand;
  if (!cand) return (el.textContent = '');
  const sig = cand.signals;
  const bits = sig
    ? Object.entries(sig).filter(([, n]) => typeof n === 'number' && n > 0).map(([k, n]) => `${k} ${n.toFixed(2)}`)
    : [];
  const all = cands();
  el.textContent = [
    `Score ${cand.score.toFixed(2)}, rank ${all.indexOf(cand) + 1} of ${all.length}`,
    `Analyzed ${fmt(cand.start)} to ${fmt(cand.end)}`,
    bits.length ? `Signals: ${bits.join(', ')}` : '',
    sig?.keywords?.length ? `Keywords: ${sig.keywords.join(', ')}` : '',
  ].filter(Boolean).join('\n');
}

// ---------- 360 director previews (#517) ----------

let panoAsked = null; // "<project>:<candidate>:<path>" whose director preview the page last asked for

/** Ask the server once for the director preview of 360 moment ``c``: it plans the camera path (the cached one when current) and renders it in the background. */
function askDirector(c) {
  const cand = c?.cand;
  if (S.project?.kind !== '360' || !cand?.pano || cand.pano.preview || S.seqPlay) return;
  const k = `${S.id}:${cand.cli_id}:${cand.pano.path}`; // a new plan (Analyze again) is asked for again
  if (panoAsked === k) return;
  panoAsked = k;
  postDirector(cand.cli_id);
}

/** Ask for (or with ``cancel``, stop) the director preview of candidate ``cid`` (its CLI id). */
async function postDirector(cid, cancel = false) {
  try {
    await api(`/api/projects/${encodeURIComponent(S.id)}/director/${encodeURIComponent(cid)}${cancel ? '/cancel' : ''}`, { method: 'POST' });
    refreshProject();
  } catch (e) {
    fail(e);
  }
}

/** The render's progress (or, while its path is planned, the elapsed time) beside its state; not announced, as it changes on every poll. */
function panoTime(pv) {
  if (pv?.state !== 'rendering') return '';
  if (pv.progress == null) return ` ${Math.round(pv.elapsed)} s`;
  return ` ${Math.round(pv.progress * 100)}% (about ${Math.round(pv.seconds)} s of video)`;
}

// What a ready director view shows: the ``kind`` of the server's ``preview_path`` (with its ``reason``)
const PANO_KINDS = { cached: 'the camera path the analysis planned', planned: 'a camera path planned now', level_forward: 'a level forward view' };

/**
 * Under the preview of a 360 moment: what it shows (the director view, or the raw fisheye and why),
 * a render's state with Cancel or a retry, and the camera path's flags with their reasons as text.
 * The one status region keeps its node and changes text only when the status does; the button and
 * the flag list are rebuilt only when they change.
 */
function renderPanoNote(c) {
  const el = $('#panoNote');
  clearTimeout(S.panoTimer);
  const cand = S.project?.kind === '360' ? c?.cand : null;
  const status = el.querySelector('.pano-status');
  const actEl = el.querySelector('.pano-act');
  const flagsEl = el.querySelector('.pano-flags');
  if (!cand) {
    el.hidden = true;
    status.textContent = '';
    delete status.dataset.want;
    actEl.replaceChildren();
    flagsEl.replaceChildren();
    flagsEl.hidden = true;
    delete el.dataset.key;
    return;
  }
  const pano = cand.pano;
  const pv = pano?.preview;
  const state = pv?.state ?? (pano ? 'asking' : 'none');
  // a render is followed until it is done (the request's own answer refreshes after 'asking')
  if (!S.expired && (state === 'rendering' || state === 'queued')) S.panoTimer = setTimeout(refreshProject, 1000);
  const on = v.dataset.director === c.id;
  let text;
  let act = null; // [label, cancel]
  if (S.seqPlay) text = 'Play sequence shows the raw 360 preview.';
  else if (state === 'none') text = 'Raw 360 preview: this moment is not in a 360 clip of the project.';
  else if (state === 'asking') text = 'Raw 360 preview: checking the camera path…';
  else if (state === 'ready' && directorBad.has(pv.url))
    text = 'Raw 360 preview: the browser could not play the director view.';
  else if (state === 'ready')
    text = on ? `Director view: ${PANO_KINDS[pv.kind] ?? PANO_KINDS.cached}, ${fmt(pv.lo)} to ${fmt(pv.hi)}.${pv.reason ? ` Why: ${pv.reason}.` : ''}`
      : `Raw 360 preview here: the director view covers ${fmt(pv.lo)} to ${fmt(pv.hi)}.`;
  else if (state === 'rendering' || state === 'queued') {
    text = `${state === 'queued' ? 'Director view queued' : 'Rendering the director view'}; the raw 360 preview plays meanwhile.`;
    act = ['Cancel', true];
  } else if (state === 'cancelled') {
    text = 'Raw 360 preview: the director view was cancelled.';
    act = ['Render director view', false];
  } else {
    text = `Raw 360 preview: the director view could not be rendered (${pv.reason}).`; // failed
    act = ['Retry director view', false];
  }
  const reasons = pano?.reasons ?? [];
  // shown before its status changes, so the first status is announced too (on the next frame)
  const wasHidden = el.hidden;
  el.hidden = false;
  status.dataset.want = text;
  const put = () => {
    if (status.dataset.want === text && status.textContent !== text) status.textContent = text;
  };
  if (wasHidden) requestAnimationFrame(put);
  else put();
  el.querySelector('.pano-time').textContent = panoTime(pv);
  const keyNow = JSON.stringify([c.id, act, reasons]);
  if (el.dataset.key === keyNow) return;
  el.dataset.key = keyNow;
  const cid = cand.cli_id;
  const hadFocus = el.contains(document.activeElement);
  actEl.replaceChildren(...(act ? [h('button', { type: 'button', class: 'btn small', onclick: () => postDirector(cid, act[1]), text: act[0] })] : []));
  flagsEl.replaceChildren(...(reasons.length ? [
    h('span', { class: 'pano-flags-title', text: 'Camera path looks poor:' }),
    h('ul', {}, ...reasons.map((r) => h('li', {}, h('span', { class: 'flag', 'data-flag': r.flag, text: PANO_FLAGS[r.flag] ?? r.flag }), ` ${r.text}`)))] : []));
  flagsEl.hidden = !reasons.length;
  if (hadFocus) (el.querySelector('button') ?? $('#playBtn')).focus(); // Cancel became Render director view
}

/** "Full preview 42%" beside the moment while its source's whole proxy builds; window proxies play meanwhile. */
function renderFullPrep(c) {
  const src = c && srcOf(c.source);
  const el = $('#stagePrep');
  let text = '';
  if (src?.status === 'preparing') text = src.progress != null ? `Full preview ${Math.floor(src.progress * 100)}%` : 'Full preview queued';
  el.textContent = text;
  el.hidden = !text;
}

// Words Whisper was unsure of (analyze lists them as [index in transcript.split(' '), p]) get a
// dotted underline and their confidence; without the list the transcript stays plain text.
function renderTranscript(cand) {
  const el = $('#transcript');
  const text = cand?.transcript;
  const unsure = new Map(Array.isArray(cand?.transcript_unsure) ? cand.transcript_unsure : []);
  if (!text || !unsure.size) {
    el.textContent = text || 'No speech was transcribed for this moment.';
    return;
  }
  el.replaceChildren(...text.split(' ').flatMap((word, i) => {
    const sep = i ? ' ' : '';
    const p = unsure.get(i);
    if (typeof p !== 'number') return [sep + word];
    const pct = p < 0.005 ? '<1%' : `${Math.round(p * 100)}%`;
    return [sep, h('span', { class: 'unsure', title: `Whisper confidence ${pct}` }, word, h('span', { class: 'vh' }, ' (unsure)'))];
  }));
}

function renderTrim(force = false) {
  const c = current();
  if (!c) return;
  drawRail(c);
  for (const which of ['start', 'end']) {
    const input = trimInputs[which];
    if (force || document.activeElement !== input) input.value = fmt(c[which]);
  }
  const changedFromOrig = c.cand && (c.start !== c.cand.start || c.end !== c.cand.end);
  $('#resetBtn').hidden = !changedFromOrig;
  const len = c.end - c.start;
  setTrimMsg(`Clip length ${len.toFixed(2)} s${c.cand ? `, analyzed as ${fmt(c.cand.start)} to ${fmt(c.cand.end)}` : ''}.${
    len > LONG_CLIP ? ' This clip is longer than 10 minutes, so exporting it takes a while.' : ''}`);
}

/** The rail (the view, with the clip and analyzed range on it) and the overview strip of the whole source. */
function drawRail(c = current()) {
  if (!c) return;
  const dur = durOf(c);
  S.view = { id: c.id, ...(S.view?.id === c.id ? clampView(S.view.lo, S.view.hi, dur) : fitView(c)) };
  const { lo, hi } = S.view;
  const span = hi - lo;
  $('#winLo').textContent = fmt(lo);
  $('#winHi').textContent = fmt(hi);
  seek.min = lo;
  seek.max = hi;
  const place = (el, a, b, from, len) =>
    Object.assign(el.style, { left: `${(((a - from) / len) * 100).toFixed(2)}%`, width: `${(((b - a) / len) * 100).toFixed(2)}%` });
  place($('#railSel'), c.start, c.end, lo, span);
  drawGaps(c, dur, lo, hi, place);
  drawFilm(c);
  const o = $('#railOrig');
  o.hidden = !c.cand;
  if (c.cand) place(o, c.cand.start, c.cand.end, lo, span);
  const pan = $('#pan');
  pan.disabled = !dur;
  $('#overview').classList.toggle('off', !dur);
  if (dur) {
    pan.max = dur;
    pan.value = lo + span / 2;
    pan.setAttribute('aria-valuetext', `${hms(lo)} to ${hms(hi)} of ${hms(dur)}`);
    place($('#ovView'), lo, hi, 0, dur);
    place($('#ovClip'), c.start, c.end, 0, dur);
  }
  const t = playhead(c);
  wasIn = t >= lo && t <= hi;
  // a marker the view cuts off loses its edge border on that side
  const sel = $('#railSel');
  sel.classList.toggle('cut-l', c.start < lo);
  sel.classList.toggle('cut-r', c.end > hi);
  drawHandles(c, lo, hi);
  drawGhost(c);
  const zIn = $('#zoomIn');
  const zOut = $('#zoomOut');
  const had = document.activeElement; // read first: disabling a focused button drops its focus
  zIn.disabled = span <= MIN_VIEW + 1e-6;
  zOut.disabled = !!dur && span >= dur - 1e-6;
  // a button that just reached its limit hands focus to its partner instead of dropping it
  if (zIn.disabled && had === zIn) zOut.focus();
  if (zOut.disabled && had === zOut) zIn.focus();
  updatePlayhead();
}

const handles = { start: $('#startHandle'), end: $('#endHandle') };
const HANDLE_NAMES = { start: 'Clip start', end: 'Clip end' };

/** Where a trim handle may go (#352): inside the source, MIN_CLIP from the other point. ``max`` is Infinity while the source length is unknown. */
function handleRange(which, c) {
  const dur = durOf(c);
  return which === 'start'
    ? { min: 0, max: round2(c.end - MIN_CLIP) }
    : { min: round2(c.start + MIN_CLIP), max: dur ? Math.floor(dur * 100) / 100 : Infinity };
}

/** Place the trim handles on the rail (a point off the view parks its handle at that edge) and set their slider values. */
function drawHandles(c, lo, hi) {
  for (const which of ['start', 'end']) {
    const el = handles[which];
    const t = c[which];
    const { min, max } = handleRange(which, c);
    el.style.left = `${(Math.max(0, Math.min(1, (t - lo) / (hi - lo))) * 100).toFixed(2)}%`;
    el.classList.toggle('off-view', t < lo || t > hi);
    el.setAttribute('aria-valuemin', min);
    el.setAttribute('aria-valuemax', Number.isFinite(max) ? max : Math.max(t, hi));
    el.setAttribute('aria-valuenow', t);
    el.setAttribute('aria-valuetext', `${HANDLE_NAMES[which]}, ${fmt(t)}`);
  }
}

/**
 * Ghost handles (#346): the trim the previewed AI Chat proposal gives the open clip, dashed on the
 * rail in their own element, which never takes the pointer. Hidden when the proposal leaves its
 * times as they are; the Sequence tab states the old and new times in words.
 */
function drawGhost(c = current()) {
  const el = $('#railGhost');
  const w = c && S.view?.id === c.id ? S.view : null;
  const to = w && S.proposal?.diff.find((r) => r.id === c.id)?.after;
  el.hidden = !to || (to.start === c.start && to.end === c.end);
  if (el.hidden) return;
  const at = (t) => Math.max(0, Math.min(1, (t - w.lo) / (w.hi - w.lo)));
  el.style.left = `${(at(to.start) * 100).toFixed(2)}%`;
  el.style.width = `${((at(to.end) - at(to.start)) * 100).toFixed(2)}%`;
  el.classList.toggle('cut-l', to.start < w.lo);
  el.classList.toggle('cut-r', to.end > w.hi);
}

/** Move a trim point, clamped to its handle's range, through the same path as the time fields and "Use playhead". */
function moveHandle(which, t, widen = true) {
  const c = current();
  if (!c) return false;
  const { min, max } = handleRange(which, c);
  t = round2(Math.max(min, Math.min(max, t)));
  return t === c[which] || setRange(which, t, widen);
}

/**
 * Hatch the parts of the source no ready file plays yet (#323), on the rail and the overview:
 * while its proxy builds, all but its ready window proxies. Rebuilt only when they or the view change.
 */
function drawGaps(c, dur, lo, hi, place) {
  const src = srcOf(c.source);
  const gaps = sourceGaps(src, dur || Math.max(hi, ...(src?.windows ?? []).map((w) => w.hi)));
  const fill = (el, from, to) => {
    const sig = `${from}|${to}|${gaps.join(';')}`;
    if (el.dataset.sig === sig) return;
    el.dataset.sig = sig;
    el.replaceChildren(...gaps.filter(([a, b]) => b > from && a < to).map(([a, b]) => {
      const g = h('span');
      place(g, Math.max(a, from), Math.min(b, to), from, to - from);
      return g;
    }));
  };
  fill($('#railGaps'), lo, hi);
  if (dur) fill($('#ovGaps'), 0, dur);
  else $('#ovGaps').replaceChildren();
}

// The filmstrip behind the rail (#261): keyframe stills (``key=1``, the keyframe at or before each
// time) on a grid of FILM_STEP * 2^k seconds, so panning reuses every still and a 2x zoom every one
// or every other one. Below FILM_EXACT, keyframes are usually further apart than the grid, so the
// few stills of a deep zoom are exact ones (2-3x the cost each) rather than one keyframe repeated.
// Purely decorative: a failed still is left out and the ticks show through.
const film = { src: null, sig: '', step: 0, tiles: new Map(), timer: 0 }; // tiles: time -> <img> of film.src
const filmFails = new Map(); // source path -> failed stills this session
const filmOff = () => matchMedia('(prefers-reduced-data: reduce)').matches || !!navigator.connection?.saveData;

/** Lay the filmstrip out over the current view: keep and move the stills it shares, drop the rest, ask for new ones once the view is still. */
function drawFilm(c = current()) {
  const el = $('#railFilm');
  const src = c && srcOf(c.source);
  const width = $('#rail').clientWidth;
  const on = !!src?.url && !!S.view && !!width && !filmOff() && (filmFails.get(c.source) ?? 0) < FILM_FAILS;
  const sig = on ? `${src.url}|${S.view.lo}|${S.view.hi}|${width}` : '';
  if (sig === film.sig) return; // a trim or a poll: the view has not moved
  film.sig = sig;
  filmGen += 1; // stills still queued for an earlier view are dropped
  clearTimeout(film.timer);
  if (!on) {
    film.tiles.clear();
    el.replaceChildren();
    film.src = null;
    return;
  }
  const { lo, hi } = S.view;
  const span = hi - lo;
  const dur = durOf(c) || hi;
  let step = FILM_STEP;
  while (step < span / Math.ceil(width / FILM_TILE)) step *= 2;
  if (film.src !== c.source) {
    film.tiles.clear();
    el.replaceChildren();
    film.src = c.source;
  }
  film.step = step;
  const times = [];
  for (let t = Math.floor(lo / step) * step; t < Math.min(hi, dur); t += step) times.push(t);
  const want = new Set(times);
  for (const [t, img] of film.tiles) {
    if (want.has(t)) continue;
    img.remove();
    film.tiles.delete(t);
  }
  let missing = false;
  const key = step >= FILM_EXACT;
  for (const t of times) {
    let img = film.tiles.get(t);
    if (img && !img.src && img.dataset.key !== String(key)) { // not loading yet: ask in this zoom's mode
      img.remove();
      film.tiles.delete(t);
      img = null;
    }
    img ??= filmTile(c.source, src, t, key); // a loaded still stays, whichever mode
    if (!img) continue; // failed before: the ticks show through
    Object.assign(img.style, { left: `${(((t - lo) / span) * 100).toFixed(2)}%`, width: `${((step / span) * 100).toFixed(2)}%` });
    if (!img.isConnected) el.append(img);
    if (!img.src && img.dataset.queued !== 'loading') missing = true;
  }
  if (missing) film.timer = setTimeout(loadFilm, FILM_WAIT);
}

/** A filmstrip still of ``src`` at ``t``, loaded at once when the browser has it, or null when it failed before. */
function filmTile(path, src, t, key) {
  const [base, query] = src.url.split('?'); // query carries the session token, as for <video>
  const url = `${base}/frame?${query}&at=${t}&w=${THUMB_W}${key ? '&key=1' : ''}`;
  const state = thumbs.get(url);
  if (state === 'failed') return null;
  const img = h('img', { decoding: 'async', alt: '' });
  img.draggable = false; // a dragged image would carry its URL, session token included
  img.dataset.src = url;
  img.dataset.key = String(key);
  const done = (ok) => {
    thumbs.set(url, ok ? 'ok' : 'failed');
    if (img.dataset.queued === 'loading') {
      img.dataset.queued = '';
      thumbLoading -= 1;
      pumpThumbs();
    }
    if (ok) {
      img.classList.add('ok');
      return;
    }
    img.remove();
    if (film.tiles.get(t) === img) film.tiles.delete(t);
    filmFails.set(path, (filmFails.get(path) ?? 0) + 1);
    if (filmFails.get(path) >= FILM_FAILS && film.src === path) drawFilm(); // stop: clears the strip
  };
  img.onload = () => done(true);
  img.onerror = () => done(false);
  if (state === 'ok') img.src = url; // seen before: the browser cache serves it
  film.tiles.set(t, img);
  return img;
}

/** Queue the filmstrip's missing stills, nearest the playhead first, ahead of the sequence thumbnails. */
function loadFilm() {
  const t0 = playhead();
  const half = film.step / 2;
  const waiting = [...film.tiles].filter(([, img]) => !img.src && img.dataset.queued !== 'loading');
  waiting.sort(([a], [b]) => Math.abs(a + half - t0) - Math.abs(b + half - t0));
  filmQueue.length = 0;
  for (const [, img] of waiting) filmQueue.push({ img, gen: filmGen });
  pumpThumbs();
}
new ResizeObserver(() => drawFilm()).observe($('#rail')); // a wider rail holds more stills

/** The [from, to] parts of ``src`` up to ``end`` no ready file plays yet: none unless its proxy builds. */
function sourceGaps(src, end) {
  const gaps = [];
  if (!src || !['needs-proxy', 'preparing'].includes(src.status)) return gaps;
  let at = 0;
  for (const w of [...(src.windows ?? [])].sort((a, b) => a.lo - b.lo)) {
    if (w.lo > at) gaps.push([at, w.lo]);
    at = Math.max(at, w.hi);
  }
  if (at < end) gaps.push([at, end]);
  return gaps;
}

/** Move the rail view to [lo, hi] (kept inside the source) and redraw it. */
function setView(lo, hi) {
  const c = current();
  if (!c) return;
  S.view = { id: c.id, ...clampView(lo, hi, durOf(c)) };
  drawRail(c);
}

/** Zoom by ``f`` (2: out, 0.5: in), keeping ``at`` (default: the playhead, or the middle when off the rail) in place. */
function zoomBy(f, at) {
  const w = S.view;
  if (!w) return;
  const t = at ?? playhead();
  const a = t >= w.lo && t <= w.hi ? t : (w.lo + w.hi) / 2;
  const dur = durOf(current());
  const span = Math.max(MIN_VIEW, Math.min((w.hi - w.lo) * f, dur || Infinity));
  const lo = a - ((a - w.lo) / (w.hi - w.lo)) * span;
  setView(lo, lo + span);
}

/** Bring ``t`` onto the rail when it is off it: an instant jump that puts it 10% from the left edge. */
function follow(t) {
  const w = S.view;
  if (!w || (t >= w.lo && t <= w.hi)) return;
  const span = w.hi - w.lo;
  setView(t - 0.1 * span, t + 0.9 * span);
}

// ---------- video ----------

function screenMsg(text) {
  const el = $('#screenMsg');
  if (el.textContent !== text) el.textContent = text; // the same text again (a poll): not re-announced
  el.hidden = !text;
}

// Source proxies are made on demand: only the focused source is prepared, and only it is polled.
// Keys "<project>:<path>": whole proxies asked for. Window requests dedupe only against the last
// one (``lastPart``, the key plus "@<lo>-<hi>"): the server may drop a window (superseded, stopped,
// evicted, failed), so asking for an earlier range again must reach it.
const prepareRequested = new Set();
let lastPart = null;
let nextFor = null; // the window proxy url whose successor ``requestNext`` asked for
// Playback paused itself at a window proxy's end (``partEnd``): {source, t, url}, the source path
// and time it waits at and the file it stopped in. A file covering ``t`` that loads in its place
// plays on by itself (``canplay``); a pause, a scrub, a play or any other wait drops it.
let resumeAt = null;
// The wait ``awaitPart`` holds: {source, t, lo, hi}, the source time it waits at and the range it
// asked a window for. A refresh re-running ``ensureVideo`` during it asks for that range again
// (the ``lastPart`` dedupe then drops it), never for the moment's: playback waiting at a window's
// end inside a long moment needs the window there, not one spanning the moment. ``retried``: it
// asked once more after the server listed no window for it; ``failed``: that failed too, so the
// stage shows the error and Retry.
let waitFor = null;
/** Drop a pending resume; a paused video then shows Play (a pause on a paused video fires no event). */
function cancelResume() {
  if (!resumeAt) return;
  resumeAt = null;
  if (v.paused) showPlaying(false);
}
// seconds before a playing window proxy's end the next one is asked for: a window encodes in a few
// seconds, about twice that while the whole proxy builds, and the page polls every 2 s
const EARLY = 20;
const PART_MSG = 'Preparing this part of the source…';
const MOMENT_MSG = 'Preparing this moment\'s preview…';

/** Forget the prepare requests of ``src`` (whole and window), so they are made again. */
function forgetPrepared(src) {
  const k = `${S.id}:${src.path}`;
  prepareRequested.delete(k);
  if (lastPart?.startsWith(`${k}@`)) lastPart = null;
}

/**
 * Queue the proxy of ``src``; with ``range`` ({lo, hi}), a window proxy around it first (the server
 * queues both). ``queue``: the window waits behind the queued ones instead of superseding them
 * ("Play sequence" asks for its next clips so; its poll refreshes the project).
 */
async function prepareSource(src, range, queue = false) {
  const whole = `${S.id}:${src.path}`;
  const k = range && !queue ? `${whole}@${range.lo}-${range.hi}` : null;
  const had = prepareRequested.has(whole);
  if (had && (!range || (k && k === lastPart))) return;
  prepareRequested.add(whole); // a window request queues the whole proxy too
  if (k) lastPart = k;
  const sid = src.id ?? decodeURIComponent(src.url.split('?')[0].split('/').pop());
  try {
    await api(`/api/projects/${encodeURIComponent(S.id)}/sources/${encodeURIComponent(sid)}/prepare`, { method: 'POST', body: queue ? { ...range, queue } : range });
    if (k) S.partAck = k; // the server has this window request: listings from now on reflect it
    if (!queue) refreshProject();
  } catch (e) {
    if (!had) prepareRequested.delete(whole);
    if (k && lastPart === k) lastPart = null;
    fail(e);
  }
}

/**
 * Ask for a window proxy around [lo, hi] of ``src`` unless the one encoding covers it; ``delay`` ms
 * debounces scrubbing. A covering window still queued (behind "Play sequence"'s) is asked for again:
 * a focus moves it to the front. The ``lastPart`` dedupe still holds: the page's last focus is already
 * first in the server's queue, and queued requests never set it.
 */
function requestPart(src, lo, hi, delay = 0) {
  clearTimeout(S.partTimer);
  const range = { lo: round2(Math.max(0, lo)), hi: round2(Math.max(0, hi)) };
  const pend = (src.pending ?? []).filter((p) => p.lo <= range.lo && p.hi >= range.hi);
  if (pend.some((p) => p.state === 'encoding')) return;
  const id = S.id;
  if (delay) S.partTimer = setTimeout(() => S.id === id && prepareSource(src, range), delay);
  else prepareSource(src, range);
}

/** Load ``p`` (from ``playable``) into the stage player, to show source time ``t``. */
function loadFile(src, p, t) {
  v.dataset.url = p.url;
  v.dataset.offset = p.offset;
  if (p.url === src.url) delete v.dataset.hi;
  else v.dataset.hi = p.hi;
  if (p.director) v.dataset.director = p.director;
  else delete v.dataset.director;
  seekWant = null;
  pendingSeek = t;
  screenMsg('Loading video…');
  v.src = p.url;
  waitFor = null;
  if (resumeAt && src.path !== resumeAt.source) resumeAt = null; // another source: no resume
  showPlaying(!!resumeAt); // a new src stops playback without a pause event; a resume plays on
  renderPanoNote(current()); // director view or raw 360 preview (#517)
}

/** Nothing ready covers source time ``t``: hold the playhead there, paused, until a window proxy around [lo, hi] is built. */
function awaitPart(src, t, delay = 0, lo = t, hi = t) {
  // A wait elsewhere (a scrub, another moment) cancels a resume; the same wait again (a refresh
  // re-running ensureVideo) keeps it. ``partEnd`` arms it after this.
  if (resumeAt && (src.path !== resumeAt.source || Math.abs(t - resumeAt.t) > FRAME)) cancelResume();
  v.pause();
  seekWant = null;
  pendingSeek = t;
  screenMsg(PART_MSG);
  waitFor = { source: src.path, t, lo, hi };
  requestPart(src, lo, hi, delay);
  updatePlayhead();
}

function ensureVideo(seekC) {
  clearTimeout(S.prepTimer);
  if (S.expired) return;
  // shown only when this pass ends in a failure; set once, so a focused Retry keeps its focus
  const hide = !stageVideo(seekC);
  if (hide && document.activeElement === $('#retryBtn')) $('#playBtn').focus(); // not lost to <body>
  $('#retryBtn').hidden = hide;
}

/** ``ensureVideo``'s work; true when it ends in a failure Retry can fix. */
function stageVideo(seekC) {
  const c = current();
  const src = c && srcOf(c.source);
  // drop() forgets pending seeks; a moment waiting for its video keeps its start or the user's scrub
  const want = seekC ? seekC.start : pendingSeek;
  const wait = (msg) => {
    drop(msg);
    pendingSeek = want;
  };
  if (!src) {
    wait('This source file is not part of the project, so it cannot be previewed. You can still trim by timecode.');
    return;
  }
  askDirector(c);
  // A 360 moment's rendered director preview (#517) plays in place of the raw fisheye; a raw file
  // playing on switches at the next pause (as to a whole proxy)
  const dAt = want ?? (ofSource(src) ? playhead(c) : c.start);
  const d = directed(src, dAt, c);
  if (d) {
    // the raw preview is still prepared, for the source outside the director preview's span
    if (src.status !== 'ready' && !prepareRequested.has(`${S.id}:${src.path}`)) prepareSource(src);
    const rawOn = ofSource(src) && !v.paused && !v.error && inLoaded(dAt, durOf(c));
    if (v.dataset.url !== d.url && !rawOn) loadFile(src, d, dAt);
    else if (seekC) seekTo(seekC.start);
    return;
  }
  if (src.status === 'unknown') {
    // not probed yet (the project view never probes): preparing probes it and refreshes. Still
    // unknown after an earlier prepare (the file changed, or the server restarted): prepare again.
    // A window around the moment comes with it, in case the source needs a proxy.
    const k = `${S.id}:${src.path}`;
    const range = { lo: round2(c.start), hi: round2(c.end) };
    if (!prepareRequested.has(k)) prepareSource(src, range);
    else
      S.prepTimer = setTimeout(() => {
        forgetPrepared(src);
        prepareSource(src, range);
      }, 2000);
    return wait('Loading video…');
  }
  if (src.status === 'needs-proxy' || src.status === 'preparing') {
    // The whole proxy builds; a window proxy around the moment (or the playhead) plays meanwhile.
    const k = `${S.id}:${src.path}`;
    if (src.status === 'needs-proxy' && prepareRequested.has(k))
      // requested before, but not building (the server restarted): prepare again
      S.prepTimer = setTimeout(() => {
        forgetPrepared(src);
        prepareSource(src);
      }, 2000);
    else S.prepTimer = setTimeout(refreshProject, 2000);
    const mine = ofSource(src);
    const at = want ?? (mine ? playhead(c) : c.start);
    const p = playable(src, at);
    // "Play sequence" plays a clip only from a file covering all of it: a window ending inside it would cut it short
    const fits = !S.seqPlay || readyFor(src, c);
    // and keeps the loaded file only when it covers all of the clip too (``fileFor`` picks the same)
    const keep = mine && inLoaded(at, durOf(c)) && (!S.seqPlay || holdsClip(c));
    if (fits && (keep || p)) {
      if (src.status === 'needs-proxy' && !prepareRequested.has(k)) prepareSource(src); // no window request brings it
      if (!keep) loadFile(src, p, at);
      else if (seekC) seekTo(seekC.start);
      return;
    }
    const moment = !!S.seqPlay || (at >= c.start && at <= c.end);
    // the same wait again (a refresh, not a moment click): its own range, e.g. [at, at] at a
    // window's end or a scrub, also while no file is loaded yet
    const again = !seekC && !S.seqPlay && waitFor?.source === src.path && Math.abs(at - waitFor.t) <= FRAME;
    const [lo, hi] = again ? [waitFor.lo, waitFor.hi] : moment ? [c.start, c.end] : [at, at];
    const kept = again ? waitFor : null;
    const wasRetried = !!kept?.retried;
    if (kept && !kept.retried && lastPart === `${k}@${round2(Math.max(0, lo))}-${round2(Math.max(0, hi))}`
      && !(src.pending ?? []).some((w) => w.lo <= round2(Math.max(0, lo)) && w.hi >= round2(Math.max(0, hi)))) {
      // the server listed no window for the range this wait asked for (it failed or was superseded):
      // ask once more, past the lastPart dedupe; once per wait, so a failing encode is not retried forever
      kept.retried = true;
      lastPart = null;
      S.partAck = null;
    }
    const range = { lo: round2(Math.max(0, lo)), hi: round2(Math.max(0, hi)) };
    // failed only once the server took the re-ask: a listing fetched before it still shows the old failure
    const failed = wasRetried && S.partAck === `${k}@${range.lo}-${range.hi}`
      && !(src.pending ?? []).some((w) => w.lo <= range.lo && w.hi >= range.hi)
      && (src.failed ?? []).find((w) => w.lo <= range.lo && w.hi >= range.hi);
    if (failed) {
      // the window asked for once more failed too: stop waiting, say so, offer Retry (``retryBtn``)
      kept.failed = true;
      cancelResume(); // playback stopped for good: Play, not Pause
      screenMsg(`This part of the source could not be prepared for preview${failed.reason ? ` (${failed.reason})` : ''}. Retry it, or trim by timecode.`);
      return true;
    }
    if (kept) kept.failed = false;
    if (mine && !S.seqPlay) {
      awaitPart(src, at, 0, lo, hi); // keeps the last frame
      if (kept) waitFor = kept;
      return;
    }
    requestPart(src, lo, hi);
    wait(S.seqPlay ? `Preparing preview of ${baseName(c.source)}…` : moment && !again ? MOMENT_MSG : PART_MSG);
    if (kept) waitFor = kept; // drop() forgot it; the wait goes on
    return;
  }
  if (src.status === 'failed') {
    wait('The browser-playable preview could not be built for this source. Retry it, or trim by timecode.');
    return true;
  }
  forgetPrepared(src); // ready: a later change to the file prepares again
  const p = playable(src, 0);
  // A window proxy of this source playing on switches to the whole proxy at the next pause or
  // seek; during "Play sequence" only while it covers all of the clip.
  const playingOn = v.dataset.hi && ofSource(src) && !v.paused && !v.error && inLoaded(want ?? playhead(c), durOf(c))
    && (!S.seqPlay || holdsClip(c));
  if (v.dataset.url !== p.url && !playingOn) {
    loadFile(src, p, want ?? (ofSource(src) ? playhead(c) : c.start));
  } else if (seekC) seekTo(seekC.start);
}

/** Stop a video element and release its file. */
function unload(el) {
  el.pause();
  el.removeAttribute('src');
  delete el.dataset.url;
  delete el.dataset.offset;
  delete el.dataset.hi;
  delete el.dataset.director;
  el.load();
}

function drop(msg) {
  resumeAt = null;
  waitFor = null;
  seekWant = null;
  pendingSeek = null;
  unload(v);
  showPlaying(false); // load() drops the queued pause event, so a playing video would keep the Pause icon
  screenMsg(msg);
}

/**
 * Seek to source time ``t`` (kept inside the source) and bring it onto the rail. One seek is in
 * flight at a time: while the video seeks, only the latest target is kept and applied on
 * ``seeked``, so dragging stays smooth. ``fast``: a keyframe seek (fastSeek) while dragging.
 */
function seekTo(t, fast = false) {
  const c = current();
  if (!c) return;
  const dur = durOf(c);
  t = Math.max(0, dur ? Math.min(t, dur) : t);
  if (resumeAt && Math.abs(t - resumeAt.t) > FRAME) cancelResume(); // a scrub, not the resume's own seek
  if (!S.seqPlay && !stays(c, t, dur)) {
    wasIn = true;
    follow(t);
    updatePlayhead();
    return;
  }
  if (waitFor) { // the target is on playable media: a wait (e.g. at a window's end) is over
    if (waitFor.failed) {
      if (document.activeElement === $('#retryBtn')) $('#playBtn').focus(); // not lost to <body>
      $('#retryBtn').hidden = true;
      screenMsg('');
    }
    waitFor = null;
  }
  if (v.readyState < HTMLMediaElement.HAVE_METADATA) {
    pendingSeek = t; // applied on loadedmetadata
    wasIn = true;
    follow(t);
    updatePlayhead();
    return;
  }
  wasIn = true;
  pendingSeek = null; // a target held while no file covered it
  if ($('#screenMsg').textContent === PART_MSG) screenMsg('');
  if (v.seeking) seekWant = { t, fast };
  else {
    seekWant = null;
    const vt = t - offsetOf();
    if (fast && v.fastSeek) v.fastSeek(vt);
    else v.currentTime = vt;
  }
  follow(t);
  updatePlayhead();
}

/**
 * Before a seek to ``t``: when the stage file does not cover it (a window proxy, #323) or the
 * whole source became playable, load the ready file covering it, else hold the playhead there
 * and ask for a window around it. True when the loaded file seeks there itself.
 */
function stays(c, t, dur) {
  const src = srcOf(c.source);
  const d = src && directed(src, t, c);
  if (d) {
    if (v.dataset.url === d.url) return true;
    loadFile(src, d, t); // back on the moment's director preview (#517)
    return false;
  }
  if (src && v.dataset.director) { // off the director preview's span: the raw preview there
    const p = playable(src, t);
    if (p) loadFile(src, p, t);
    else awaitPart(src, t, 300);
    return false;
  }
  if (!src || !['ready', 'needs-proxy', 'preparing'].includes(src.status)) return true;
  if (!v.dataset.url && src.status === 'ready') return true; // ensureVideo is loading it
  if (v.dataset.url && inLoaded(t, dur) && (src.status !== 'ready' || !v.dataset.hi)) return true;
  const p = playable(src, t);
  if (p) loadFile(src, p, t);
  else awaitPart(src, t, 300);
  return false;
}

/**
 * Playback reached the end of the window proxy it plays: go on in a ready file covering what
 * follows, else pause there until one is built, and play on once it loads (``resumeAt``).
 */
function partEnd() {
  const c = current();
  const src = c && srcOf(c.source);
  const r = loaded();
  const dur = c && durOf(c);
  if (!src || !r || r.hi == null || (dur && r.hi >= dur - FRAME)) return false;
  const t = Math.min(playhead(c), r.hi);
  if (t < r.hi - FRAME && !v.ended) return false;
  const p = playable(src, t);
  if (p && p.url !== v.dataset.url) {
    loadFile(src, p, t);
    v.play().catch(() => {});
  } else {
    awaitPart(src, t);
    resumeAt = { source: src.path, t, url: v.dataset.url };
    showPlaying(true); // still playing, waiting: a click pauses (cancels the resume)
  }
  return true;
}

/**
 * A window proxy plays within EARLY s of its end (not the source's): ask for the window after it
 * now, as a focus on its end, so playback goes on there without a stall. Once per window file.
 */
function requestNext() {
  const r = loaded();
  const c = current();
  const src = c && srcOf(c.source);
  if (!src || !r || r.hi == null || src.status === 'ready' || nextFor === v.dataset.url) return;
  const dur = durOf(c);
  if ((dur && r.hi >= dur - FRAME) || r.hi - playhead(c) > EARLY) return;
  nextFor = v.dataset.url;
  if (!playable(src, r.hi)) requestPart(src, r.hi, r.hi);
}

/** ``playing``: called every frame during playback, so the slider's spoken value waits for a pause. */
function updatePlayhead(playing = false) {
  const c = current();
  const t = playhead(c);
  seek.value = t;
  seek.classList.toggle('off-view', !!S.view && (t < S.view.lo || t > S.view.hi)); // no thumb parked at an edge
  if (!playing) seek.setAttribute('aria-valuetext', fmt(t));
  $('#clock').textContent = S.seqPlay ? seqClock(t) : c ? `${fmt(t)} in source` : '';
}

let tickId = 0;
const loop = () => {
  if (!tickId) tickId = requestAnimationFrame(tick); // one loop, also across player swaps
};

function tick() {
  tickId = 0;
  if (v.paused || !seqCheck() || (!S.seqPlay && partEnd())) return;
  if (!S.seqPlay) requestNext();
  // playback runs on past the clip end; the rail jumps along when the playhead runs off it, not
  // when the user panned away from it
  const t = playhead();
  const w = S.view;
  const inV = !w || (t >= w.lo && t <= w.hi);
  if (!inV && wasIn) follow(t);
  else wasIn = inV;
  updatePlayhead(true);
  loop();
}

/** During "Play sequence": move on at the clip end, read live so trims apply at once. False when nothing plays on. */
function seqCheck() {
  const sp = S.seqPlay;
  if (!sp || sp.wait || v.paused) return true;
  const seg = seqSeg(sp.i);
  if (!seg) {
    seqStop('The playing clip left the sequence. Sequence stopped.');
    return false;
  }
  if (playhead() >= seg.end - FRAME) seqGo(sp.i + 1);
  return !v.paused && !S.seqPlay?.wait;
}

function togglePlay() {
  const c = current();
  if (!c || !v.src) return;
  if (!v.paused) return v.pause();
  if (resumeAt) { // paused at a window's end, about to play on: this is a pause
    resumeAt = null;
    return showPlaying(false);
  }
  if (pendingSeek != null && !inLoaded(pendingSeek, durOf(c))) return; // waiting for that part's window proxy
  const dur = durOf(c);
  if (!S.seqPlay && (v.ended || (dur && playhead(c) >= dur - 0.05))) seekTo(c.start); // at the end of the source: replay the clip
  v.play().catch(() => {});
}

/** Listen on both players; only the visible one (``v``) drives the stage. */
const onStage = (type, fn) => els.forEach((el) => el.addEventListener(type, (e) => el === v && fn(e)));

onStage('loadedmetadata', () => {
  drawRail(); // the source length may only be known now
  const t = pendingSeek;
  pendingSeek = null;
  if (t != null) seekTo(t);
});
onStage('canplay', () => {
  screenMsg('');
  if (S.seqPlay?.wait) { // the clip's video the sequence waited for is ready
    S.seqPlay.wait = false;
    v.play().catch(() => {});
  } else if (resumeAt && v.dataset.url !== resumeAt.url && current()?.source === resumeAt.source && pendingSeek == null && inLoaded(resumeAt.t, durOf(current()))) {
    resumeAt = null; // the file after the window playback stopped at: play on
    v.play().catch(() => showPlaying(false));
  }
});
onStage('playing', () => screenMsg(''));
onStage('timeupdate', () => document.hidden && seqCheck()); // frames stop in a hidden tab; timeupdate still fires
onStage('waiting', () => screenMsg('Buffering…'));
onStage('ended', () => {
  if (S.seqPlay) !S.seqPlay.wait && seqGo(S.seqPlay.i + 1); // a source shorter than its clip
  else partEnd(); // a window proxy's end
});
onStage('error', () => {
  seekWant = null;
  if (!v.dataset.url) return;
  if (v.dataset.director) {
    // A director preview that fails (its path was re-planned since the listing) plays raw meanwhile;
    // one refresh per url picks up the new preview, and a url that failed is not tried again.
    const bad = v.dataset.url;
    const first = !directorBad.has(bad);
    directorBad.add(bad);
    delete v.dataset.director;
    delete v.dataset.url;
    ensureVideo(); // Retry shows if the raw preview fails too
    if (first) refreshProject();
    return;
  }
  const url = v.dataset.url;
  const msg = `The browser could not play this video${v.error?.message ? `: ${v.error.message}` : ''}.`;
  // The element hides the status; one API call tells an expired token (banner only) from a bad
  // file, so the message is shown, and announced, only when the session is still valid.
  const show = () => {
    if (!S.expired && v.dataset.url === url) screenMsg(msg);
  };
  // A window proxy goes once the whole proxy is built: the refresh switches to that instead.
  (v.dataset.hi ? refreshProject() : startSession()).then(show, show);
});
/** The play button is an icon; its name and tooltip say what a click does. */
function showPlaying(playing) {
  const name = playing ? 'Pause' : 'Play';
  const b = $('#playBtn');
  b.setAttribute('aria-label', name);
  b.title = `${name} (Space)`;
  b.replaceChildren(icon(playing ? 'pause' : 'play'));
}

// One preview sounds at a time: the stage (a moment or "Play sequence") and the Export tab's final
// video, which plays on while its tab is hidden, pause each other.
const finalVideo = $('#finalVideo');
finalVideo.addEventListener('play', () => {
  if (S.seqPlay) seqStop();
  else {
    cancelResume(); // a resume at a window's end would start the stage again
    v.pause();
  }
});
onStage('play', () => {
  finalVideo.pause();
  resumeAt = null;
  showPlaying(true);
  follow(playhead()); // Play shows the playhead, also after a pan away from it
  loop();
});
onStage('pause', () => {
  showPlaying(!!resumeAt); // waiting at a window's end to play on
  updatePlayhead();
  const c = current();
  const src = c && srcOf(c.source);
  const d = src && directed(src, playhead(c), c);
  if (S.seqPlay) return;
  if ((v.dataset.hi && src?.status === 'ready') || (d && d.url !== v.dataset.url)) ensureVideo(null); // the whole proxy or a director preview is ready: switch to it now
});
onStage('seeked', () => {
  const w = seekWant;
  if (w) seekTo(w.t, w.fast); // the latest scrub target that waited for this seek
  else updatePlayhead();
});
onStage('click', togglePlay);
// the spare player seeks to the clip it preloads as soon as it knows the source
els.forEach((el) => el.addEventListener('loadedmetadata', () => {
  if (el !== v && cued) el.currentTime = cued.vt;
}));
$('#playBtn').addEventListener('click', togglePlay);
$('#retryBtn').addEventListener('click', () => {
  const c = current();
  const src = c && srcOf(c.source);
  if (!src) return;
  if (waitFor?.failed && waitFor.source === src.path) { // a window proxy failed twice: ask again
    const w = waitFor;
    w.failed = false;
    w.retried = false;
    lastPart = null;
    S.partAck = null;
    if (document.activeElement === $('#retryBtn')) $('#playBtn').focus(); // not lost to <body>
    $('#retryBtn').hidden = true;
    screenMsg(PART_MSG);
    requestPart(src, w.lo, w.hi);
    return;
  }
  prepareRequested.delete(`${S.id}:${src.path}`);
  prepareSource(src);
});
seek.addEventListener('input', () => seekTo(Number(seek.value), true));
seek.addEventListener('change', () => seekTo(Number(seek.value))); // released: an exact seek
seek.addEventListener('keydown', (e) => {
  const d = { ArrowLeft: -0.005, ArrowDown: -0.005, ArrowRight: 0.005, ArrowUp: 0.005, PageDown: -0.1, PageUp: 0.1 }[e.key];
  if (d == null || !S.view) return;
  // arrows: 0.5% of the rail (at least 0.05 s), so a zoomed-out long source moves too; PageUp/PageDown: 10%
  e.preventDefault();
  const span = S.view.hi - S.view.lo;
  seekTo(playhead() + Math.sign(d) * Math.max(0.05, Math.abs(d) * span));
});
$('#zoomOut').addEventListener('click', () => zoomBy(2));
$('#zoomIn').addEventListener('click', () => zoomBy(0.5));
$('#zoomFit').addEventListener('click', () => {
  const c = current();
  const f = c && fitView(c);
  if (f) setView(f.lo, f.hi);
});
const pan = $('#pan');
const panTo = (mid) => S.view && setView(mid - (S.view.hi - S.view.lo) / 2, mid + (S.view.hi - S.view.lo) / 2);
pan.addEventListener('input', () => panTo(Number(pan.value)));
pan.addEventListener('keydown', (e) => {
  const w = S.view;
  const dur = Number(pan.max);
  const span = w ? w.hi - w.lo : 0;
  const d = { ArrowLeft: -0.1, ArrowDown: -0.1, ArrowRight: 0.1, ArrowUp: 0.1, PageDown: -1, PageUp: 1 }[e.key];
  if (!w || (d == null && e.key !== 'Home' && e.key !== 'End')) return;
  e.preventDefault(); // arrows pan by 10% of the view, PageUp/PageDown by a whole view
  panTo(e.key === 'Home' ? 0 : e.key === 'End' ? dur : (w.lo + w.hi) / 2 + d * span);
});
// mouse convenience: the wheel pans the rail, Ctrl+wheel zooms around the pointer
$('#rail').addEventListener('wheel', (e) => {
  const w = S.view;
  if (!w) return;
  e.preventDefault();
  const r = e.currentTarget.getBoundingClientRect();
  const span = w.hi - w.lo;
  const d = (e.deltaX || e.deltaY) * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? r.width : 1);
  if (e.ctrlKey) zoomBy(2 ** Math.max(-1, Math.min(1, d / 100)), w.lo + ((e.clientX - r.left) / r.width) * span); // a wheel notch: x2
  else setView(w.lo + (d / r.width) * span, w.hi + (d / r.width) * span);
}, { passive: false });

// The trim handles (#352). Keys: arrows 0.1 s, Shift+arrows 1 s, PageUp/PageDown 5 s, Home/End
// to the limit. Space is left to the global handler, so it still plays and pauses.
const HANDLE_STEPS = { ArrowLeft: -0.1, ArrowDown: -0.1, ArrowRight: 0.1, ArrowUp: 0.1, PageDown: -5, PageUp: 5 };
const DRAG_SEEK_MS = 100; // the least time between preview seeks while dragging
const PAN_EDGE = 24; // pixels: a drag this close to (or past) a rail edge pans the view along
let trimDrag = null; // {which, el, id, cid, pid, x, x0, moved, grab, raf, seekAt}

/** End a drag without moving the point. */
function dropDrag() {
  const d = trimDrag;
  cancelAnimationFrame(d.raf);
  trimDrag = null;
  d.el.classList.remove('dragging');
}

/** The drag's clip is no longer the open one (Play sequence moved on, or another moment or project opened). */
const dragStale = (d) => current()?.id !== d.cid || S.id !== d.pid;

/** Source time at client x on the rail, clamped to the view. */
function railTime(x) {
  const rail = $('#rail');
  const left = rail.getBoundingClientRect().left + rail.clientLeft;
  const f = Math.max(0, Math.min(1, (x - left) / rail.clientWidth));
  return S.view.lo + f * (S.view.hi - S.view.lo);
}

/** One drag step per frame: pan when held at an edge, move the point under the pointer, seek at most every DRAG_SEEK_MS. */
function dragFrame(final = false) {
  const d = trimDrag;
  d.raf = 0;
  if (dragStale(d)) return dropDrag();
  const c = current();
  if (!S.view) return;
  const rail = $('#rail');
  const left = rail.getBoundingClientRect().left + rail.clientLeft;
  const x = d.x - d.grab;
  const over = x < left + PAN_EDGE ? x - left - PAN_EDGE : x > left + rail.clientWidth - PAN_EDGE ? x - left - rail.clientWidth + PAN_EDGE : 0;
  let panned = false;
  const { min, max } = handleRange(d.which, c);
  // up to 2% of the view a frame, faster the further past the edge; not once the point is held at its limit
  if (over && !final && (over < 0 ? c[d.which] > min : c[d.which] < max)) {
    const { lo, hi } = S.view;
    const shift = Math.sign(over) * Math.min(1, Math.abs(over) / (2 * PAN_EDGE)) * 0.02 * (hi - lo);
    setView(lo + shift, hi + shift);
    panned = S.view.lo !== lo;
  }
  moveHandle(d.which, railTime(x), false);
  const now = performance.now();
  if (!final && now - d.seekAt >= DRAG_SEEK_MS) {
    d.seekAt = now;
    seekTo(trimSeek(d.which), true);
  }
  if (panned) d.raf = requestAnimationFrame(() => dragFrame()); // keeps panning while the pointer is held still
}

for (const [which, el] of Object.entries(handles)) {
  el.addEventListener('keydown', (e) => {
    const c = current();
    if (!c || e.ctrlKey || e.metaKey || e.altKey) return;
    const step = HANDLE_STEPS[e.key];
    const { min, max } = handleRange(which, c);
    let t;
    if (step != null) t = c[which] + (e.shiftKey && e.key.startsWith('Arrow') ? step * 10 : step);
    else if (e.key === 'Home') t = min;
    else if (e.key === 'End' && Number.isFinite(max)) t = max;
    else return;
    e.preventDefault();
    if (!moveHandle(which, t, false)) return; // keeps the zoom: only the moved point is brought onto the view
    follow(current()[which]);
    seekTo(trimSeek(which));
  });
  el.addEventListener('pointerdown', (e) => {
    const c = current();
    if (!c || !S.view || trimDrag || (e.pointerType === 'mouse' && e.button !== 0)) return;
    e.preventDefault(); // no text selection or touch scrolling
    el.focus();
    el.setPointerCapture(e.pointerId);
    const rail = $('#rail');
    const { lo, hi } = S.view;
    const at = rail.getBoundingClientRect().left + rail.clientLeft + ((c[which] - lo) / (hi - lo)) * rail.clientWidth;
    // grab offset: a press beside the point does not make it jump (none when the point is off the view)
    const grab = c[which] >= lo && c[which] <= hi ? e.clientX - at : 0;
    trimDrag = { which, el, id: e.pointerId, cid: c.id, pid: S.id, x: e.clientX, x0: e.clientX, moved: false, grab, raf: 0, seekAt: 0 };
    el.classList.add('dragging');
  });
  el.addEventListener('pointermove', (e) => {
    const d = trimDrag;
    if (!d || d.el !== el || e.pointerId !== d.id) return;
    d.x = e.clientX;
    d.moved ||= Math.abs(e.clientX - d.x0) >= 3; // a plain click (also on a parked handle) moves nothing
    if (d.moved && !d.raf) d.raf = requestAnimationFrame(() => dragFrame());
  });
  el.addEventListener('pointerup', (e) => {
    if (trimDrag?.el === el && e.pointerId === trimDrag.id) trimDrag.x = e.clientX;
  });
  el.addEventListener('lostpointercapture', (e) => { // after pointerup or pointercancel
    const d = trimDrag;
    if (!d || d.el !== el || e.pointerId !== d.id) return;
    if (dragStale(d)) return dropDrag();
    cancelAnimationFrame(d.raf);
    if (d.moved) dragFrame(true);
    dropDrag();
    seekTo(trimSeek(which)); // released: an exact seek
  });
}

// ---------- sequence ----------

function addSeg(id) {
  const d = S.drafts[id];
  const c = cands().find((x) => x.id === id) ?? (d?.source && d);
  if (!c || S.segments.some((s) => s.id === id)) return;
  remember();
  // A removed clip goes back to the index it had, clamped to the current length; clips moved or
  // removed since then shift it, which is simpler than tracking neighbours. New clips go last.
  const at = Math.min(d?.at ?? S.segments.length, S.segments.length);
  S.segments.splice(at, 0, { id, source: c.source, start: d?.start ?? c.start, end: d?.end ?? c.end, label: d?.label ?? c.label ?? id });
  delete S.drafts[id];
  renderAfterSeqChange();
  return at;
}

/** Keep a removed clip's trim and index ``at`` for re-adding; a clip that is no moment is kept
 * whole, so the stage and the Removed list can still offer it after it leaves the sequence. */
function keepDraft(seg, at) {
  S.drafts[seg.id] = cands().some((c) => c.id === seg.id)
    ? { start: seg.start, end: seg.end, label: seg.label, at }
    : { source: seg.source, start: seg.start, end: seg.end, label: seg.label, at };
}

function removeSeg(id) {
  const at = S.segments.findIndex((s) => s.id === id);
  if (at < 0) return;
  remember();
  keepDraft(S.segments[at], at);
  S.segments.splice(at, 1);
  renderAfterSeqChange();
}

function moveSeg(id, to) {
  const from = S.segments.findIndex((s) => s.id === id);
  to = Math.max(0, Math.min(S.segments.length - 1, to));
  if (from < 0 || from === to) return;
  remember();
  const [seg] = S.segments.splice(from, 1);
  S.segments.splice(to, 0, seg);
  $('#seqLive').textContent = `${seg.label || seg.id} moved to position ${to + 1} of ${S.segments.length}`;
  renderAfterSeqChange();
}

function renderAfterSeqChange() {
  renderMoments();
  renderSequence();
  renderSave();
  renderExport();
  if (!current()) S.focus = cands()[0]?.id ?? S.segments[0]?.id ?? null;
  renderStage(false);
}

function changed() {
  renderSequence();
  renderSave();
  renderExport();
}

// ---------- undo ----------
// One stack for every edit of the working sequence, by hand or applied from AI Chat (#346): a
// snapshot of the segments and drafts before each one. Ctrl+Z undoes, Ctrl+Y or Shift+Ctrl+Z redoes.

const UNDO_MAX = 50;
const UNDO_JOIN_MS = 1000; // edits of one kind this close together (a held key, a drag) are one step
const snapshot = () => ({ segments: structuredClone(S.segments), drafts: structuredClone(S.drafts) });

/** Keep the sequence as it is before an edit; ``tag`` joins a quick run of the same edit (one trim point of one clip) into one step. */
function remember(tag = null) {
  const now = performance.now();
  const last = S.undoTag;
  S.undoTag = tag ? { tag, at: now } : null;
  if (tag && last?.tag === tag && now - last.at < UNDO_JOIN_MS) return;
  S.undo.push(snapshot());
  if (S.undo.length > UNDO_MAX) S.undo.shift();
  S.redo = [];
  renderUndo();
}

/** Move one snapshot from ``from`` to the sequence, keeping the current one on ``to``. */
function restore(from, to, done) {
  if (!from.length) return;
  to.push(snapshot());
  const s = from.pop();
  S.segments = s.segments;
  S.drafts = s.drafts;
  S.undoTag = null;
  renderAfterSeqChange();
  renderUndo();
  $('#sideLive').textContent = done;
}
const undo = () => restore(S.undo, S.redo, 'Undone');
const redo = () => restore(S.redo, S.undo, 'Redone');

function renderUndo() {
  const had = document.activeElement; // read first: disabling a focused button drops its focus
  $('#undoBtn').disabled = !S.undo.length;
  $('#redoBtn').disabled = !S.redo.length;
  if (had?.disabled && (had.id === 'undoBtn' || had.id === 'redoBtn'))
    ([$('#undoBtn'), $('#redoBtn')].find((b) => !b.disabled) ?? $('#seqTitle')).focus();
}

let dragId = null;

const ICONS = {
  edit: [['path', { d: 'M3 17.25V21h3.75L17.81 9.94l-3.75-3.75L3 17.25zM20.71 7.04a1 1 0 0 0 0-1.41l-2.34-2.34a1 1 0 0 0-1.41 0l-1.83 1.83 3.75 3.75 1.83-1.83z' }]],
  trash: [['path', { d: 'M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z' }]],
  up: [['path', { d: 'M7.41 15.41 12 10.83l4.59 4.58L18 14l-6-6-6 6z' }]],
  down: [['path', { d: 'M7.41 8.59 12 13.17l4.59-4.58L18 10l-6 6-6-6z' }]],
  film: [['path', { d: 'M18 4l2 4h-3l-2-4h-2l2 4h-3l-2-4H8l2 4H7L5 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V4h-4z' }]],
  folder: [['path', { d: 'M20 6h-8l-2-2H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm0 12H4V8h16v10z' }]],
  gear: [['path', { d: 'M19.14 12.94c.04-.3.06-.61.06-.94 0-.32-.02-.64-.07-.94l2.03-1.58a.49.49 0 0 0 .12-.61l-1.92-3.32a.49.49 0 0 0-.59-.22l-2.39.96c-.5-.38-1.03-.7-1.62-.94l-.36-2.54a.48.48 0 0 0-.48-.41h-3.84c-.24 0-.43.17-.47.41l-.36 2.54c-.59.24-1.13.57-1.62.94l-2.39-.96a.48.48 0 0 0-.59.22L2.74 8.87a.47.47 0 0 0 .12.61l2.03 1.58c-.05.3-.09.63-.09.94s.02.64.07.94l-2.03 1.58a.49.49 0 0 0-.12.61l1.92 3.32c.12.22.37.29.59.22l2.39-.96c.5.38 1.03.7 1.62.94l.36 2.54c.05.24.24.41.48.41h3.84c.24 0 .44-.17.47-.41l.36-2.54c.59-.24 1.13-.56 1.62-.94l2.39.96c.22.08.47 0 .59-.22l1.92-3.32a.47.47 0 0 0-.12-.61l-2.01-1.58zM12 15.6A3.6 3.6 0 1 1 12 8.4a3.6 3.6 0 0 1 0 7.2z' }]],
  volume: [['path', { d: 'M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02zM14 3.23v2.06c2.89.86 5 3.54 5 6.71s-2.11 5.85-5 6.71v2.06c4.01-.91 7-4.49 7-8.77s-2.99-7.86-7-8.77z' }]],
  muted: [['path', { d: 'M16.5 12c0-1.77-1.02-3.29-2.5-4.03v2.21l2.45 2.45c.03-.2.05-.41.05-.63zm2.5 0c0 .94-.2 1.82-.54 2.64l1.51 1.51C20.63 14.91 21 13.5 21 12c0-4.28-2.99-7.86-7-8.77v2.06c2.89.86 5 3.54 5 6.71zM4.27 3 3 4.27 7.73 9H3v6h4l5 5v-6.73l4.25 4.25c-.67.52-1.42.93-2.25 1.18v2.06c1.38-.31 2.63-.95 3.69-1.81L19.73 21 21 19.73l-9-9L4.27 3zM12 4 9.91 6.09 12 8.18V4z' }]],
  play: [['path', { d: 'M8 5v14l11-7z' }]],
  pause: [['path', { d: 'M6 19h4V5H6v14zm8-14v14h4V5h-4z' }]],
  grip: [[9, 5], [15, 5], [9, 12], [15, 12], [9, 19], [15, 19]].map(([cx, cy]) => ['circle', { cx, cy, r: 1.9 }]),
  check: [['path', { d: 'M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z' }]],
  hourglass: [['path', { d: 'M6 2v6h.01L6 8.01 10 12l-4 4 .01.01H6V22h12v-5.99h-.01L18 16l-4-4 4-3.99-.01-.01H18V2H6zm10 14.5V20H8v-3.5l4-4 4 4zm-4-5-4-4V4h8v3.5l-4 4z' }]],
  funnel: [['path', { d: 'M4.25 5.61C6.27 8.2 10 13 10 13v6c0 .55.45 1 1 1h2c.55 0 1-.45 1-1v-6s3.72-4.8 5.74-7.39A1 1 0 0 0 18.95 4H5.04c-.83 0-1.3.95-.79 1.61z' }]],
  clock: [['path', { d: 'M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67z' }]],
};
const icon = (name) => svg(`icon icon-${name}`, ICONS[name]);

// Thumbnails are stills the server cuts (and caches) on request; remember how each URL went so a
// re-render neither flashes the skeleton again nor retries a source that cannot be read.
const thumbs = new Map(); // url -> 'ok' | 'failed'
// New stills load a few at a time so they cannot take every connection the player's range requests need.
// The rail's filmstrip goes first; its queue only holds the latest view's stills (``filmGen``).
const THUMB_LOADS = 3;
const thumbQueue = []; // <img> elements waiting for their data-src
const filmQueue = []; // {img, gen}: filmstrip stills waiting, dropped once drawFilm moved on (gen < filmGen)
let filmGen = 0;
let thumbLoading = 0;

function pumpThumbs() {
  while (thumbLoading < THUMB_LOADS && (filmQueue.length || thumbQueue.length)) {
    const next = filmQueue.shift();
    if (next && next.gen !== filmGen) continue; // queued for an earlier view
    const img = next ? next.img : thumbQueue.shift();
    if (!img.isConnected || img.src) continue; // a re-render (or the filmstrip) dropped it, or it already loads
    thumbLoading += 1;
    img.dataset.queued = 'loading';
    img.src = img.dataset.src;
  }
}

function thumbUrl(s) {
  const src = srcOf(s.source);
  if (!src?.url) return null;
  const [path, query] = src.url.split('?'); // query carries the session token, as for <video>
  const at = Math.round(Math.min(s.start + 0.5, (s.start + s.end) / 2) * 10) / 10; // skip black first frames
  return `${path}/frame?${query}&at=${at}&w=${THUMB_W}`;
}

function thumb(s) {
  const url = thumbUrl(s);
  const state = url ? thumbs.get(url) : 'failed';
  const box = h('span', { class: `seq-thumb${state === 'failed' ? ' failed' : state ? '' : ' loading'}` }, icon('film'));
  if (state !== 'failed') {
    const done = (ok) => {
      thumbs.set(url, ok ? 'ok' : 'failed');
      box.classList.remove('loading');
      box.classList.toggle('failed', !ok);
      if (img.dataset.queued === 'loading') {
        img.dataset.queued = '';
        thumbLoading -= 1;
        pumpThumbs();
      }
    };
    const img = h('img', {
      decoding: 'async', alt: '', width: 64, height: 36,
      onload: () => done(true), onerror: () => done(false),
    });
    img.draggable = false; // a dragged image would carry its URL, session token included
    if (state === 'ok') img.src = url; // seen before: the browser cache serves it
    else {
      img.dataset.src = url;
      thumbQueue.push(img);
      queueMicrotask(pumpThumbs); // after the render has put the row into the page
    }
    box.append(img);
  }
  return box;
}

/** Why a label cannot be saved (mirrors ui.py: at most 200 printable characters), or ''. */
function labelError(text) {
  if (text.length > LABEL_MAX) return `Use at most ${LABEL_MAX} characters.`;
  if (/[\p{C}\p{Zl}\p{Zp}]|(?! )\p{Zs}/u.test(text)) return 'Use letters, digits, punctuation and plain spaces only.';
  return '';
}

function startRename(id) {
  const seg = S.segments.find((s) => s.id === id);
  if (!seg) return;
  S.renaming = id;
  S.renameDraft = seg.label || seg.id;
  S.renameErr = '';
  renderSequence();
  const input = $(`#seqList [data-id="${CSS.escape(id)}"] input[data-act="rename"]`);
  input?.focus();
  input?.select();
}

/**
 * End the rename of ``input``'s row. Enter and Escape re-render the list; a blur (``inPlace``)
 * swaps only this row's editor back to its title, so the focus already moving on (Tab, a click on
 * another control) is not lost to a rebuild.
 */
function finishRename(commit, input, inPlace = false) {
  const seg = S.segments.find((s) => s.id === S.renaming);
  const text = S.renameDraft.trim();
  if (seg && commit && text && text !== seg.label) {
    const err = labelError(text);
    if (err) {
      S.renameErr = err;
      input.setAttribute('aria-invalid', 'true');
      input.closest('.seq-text').querySelector('.seq-rename-err').textContent = err;
      return;
    }
    remember();
    seg.label = text;
    S.renaming = null;
    $('#seqLive').textContent = `Clip renamed to ${text}`;
    if (inPlace) closeRename(seg, input);
    else changed();
    if (S.focus === seg.id) renderStage(false);
    return;
  }
  S.renaming = null; // cancelled, unchanged or empty: keep the old name
  if (inPlace && seg) closeRename(seg, input);
  else renderSequence();
}

/** Put the title and pencil back into the row of ``input`` without rebuilding the list. */
function closeRename(seg, input) {
  const li = input.closest('li');
  const old = li.querySelector('.seq-main');
  const { main, pencil } = seqMain(seg, [...old.querySelectorAll(':scope > .rank, :scope > .seq-thumb')]);
  old.replaceWith(main);
  li.querySelector('.seq-edit-slot').replaceWith(pencil);
  const name = seg.label || seg.id;
  for (const [act, label] of Object.entries(ROW_LABELS)) {
    li.querySelector(`[data-act="${act}"]`)?.setAttribute('aria-label', label(name));
  }
  renderSave();
  renderExport();
}

/** Ask with the shared dialog; resolves true when the user confirms. Focus returns to the trigger. */
function confirmAction(title, text, ok = 'Remove') {
  const d = $('#confirmDialog');
  const back = document.activeElement;
  $('#confirmTitle').textContent = title;
  $('#confirmText').textContent = text;
  $('#confirmOk').textContent = ok;
  d.returnValue = '';
  d.showModal();
  $('#confirmCancel').focus();
  return new Promise((resolve) => {
    d.addEventListener('close', () => {
      if (back?.isConnected) back.focus();
      resolve(d.returnValue === 'ok');
    }, { once: true });
  });
}

/**
 * Analyze options (#483): filled with what the project's analysis used (``analyze_options``), so
 * Analyze again repeats it unless changed; the disclosure starts open when one is set. Hidden
 * for 360 projects; Must-include crash only for ski styles (``must_include_crash_default``).
 */
function fillAnalyzeOptions(p) {
  const o = p.analyze_options ?? {};
  const crashDefault = p.must_include_crash_default;
  $('#analyzeOpts').hidden = p.kind === '360';
  $('#anModel').value = o.model ?? '';
  $('#anLanguage').value = o.language ?? '';
  $('#anMaxCandidates').value = o.max_candidates ?? '';
  $('#anCrash').value = o.must_include_crash ?? '';
  $('#anCrash').placeholder = crashDefault == null ? '' : `Style (${crashDefault})`;
  for (const el of [$('#anCrashLabel'), $('#anCrash')]) el.hidden = crashDefault == null;
  $('#anNoScenes').checked = !!o.no_scenes;
  $('#anRecover').checked = !!o.recover;
  for (const box of document.querySelectorAll('input[name=anSkip]')) box.checked = !!o.skip?.includes(box.value);
  $('#analyzeOpts').open = Object.values(analyzeOptions()).some((v) => (Array.isArray(v) ? v.length > 0 : v != null && v !== false));
}

/** The Analyze options as the analyze job takes them: every key, blank as null (no flag). */
function analyzeOptions() {
  const text = (id) => $(id).value.trim() || null;
  const num = (id) => ($(id).value.trim() && !$(id).hidden ? Number($(id).value) : null);
  return {
    model: text('#anModel'),
    language: text('#anLanguage'),
    max_candidates: num('#anMaxCandidates'),
    must_include_crash: num('#anCrash'),
    no_scenes: $('#anNoScenes').checked,
    recover: $('#anRecover').checked,
    skip: [...document.querySelectorAll('input[name=anSkip]:checked')].map((box) => box.value),
  };
}

/**
 * "Analyze again" (``fresh``: the first analysis, from the empty state's Analysis options):
 * confirm in a dialog with the Analyze options; vision follows Settings, as for a first analysis.
 */
function analyzeAgain(fresh = false) {
  const p = S.project;
  if (!p) return;
  const d = $('#analyzeDialog');
  const back = document.activeElement;
  $('#analyzeTitle').textContent = fresh ? 'Analyze footage' : 'Analyze again?';
  $('#analyzeText').textContent = fresh
    ? 'Analyze footage to transcribe audio and rank the strongest moments.'
    : "The project's footage is analyzed again and its moments are ranked anew.";
  fillAnalyzeOptions(p);
  if (fresh) $('#analyzeOpts').open = true;
  d.returnValue = '';
  d.showModal();
  $('#againCancel').focus();
  d.addEventListener('close', () => {
    if (back?.isConnected) back.focus();
    if (fresh && d.returnValue === 'ok') keepEmptyFocus($('#momentsEmpty')); // after focus is back, or it ends at once
    // 360 projects have no Analyze options (360 analyze takes none of them)
    if (d.returnValue === 'ok') startJob('analyze', p.kind === '360' ? {} : analyzeOptions());
  }, { once: true });
}

/** Focus control ``act`` of the row now at ``i`` (the last row if fewer remain). */
function focusRow(i, act) {
  const rows = $('#seqList').children;
  const li = rows[Math.min(i, rows.length - 1)];
  (li?.querySelector(`[data-act="${act}"]:not(:disabled)`) ?? li?.querySelector('button') ?? $('#seqTitle')).focus();
}

/** Place the open popover over ``card``: aligned to its top, flipped upward near the viewport bottom. */
function placePop(pop, card) {
  const r = card.getBoundingClientRect();
  const w = Math.min(Math.max(r.width, 260), innerWidth - 16);
  pop.style.width = `${w}px`;
  const ph = pop.offsetHeight;
  const top = r.top + ph > innerHeight - 8 ? r.bottom - ph : r.top;
  pop.style.left = `${Math.min(Math.max(8, r.right - w), innerWidth - w - 8)}px`;
  pop.style.top = `${Math.min(Math.max(8, top), innerHeight - ph - 8)}px`;
}

/**
 * Confirm next to one row with a light-dismiss popover (Escape, clicking outside, scrolling or
 * resizing cancel; opening another closes this one). Resolves true on Remove; focus returns to the
 * trigger. Browsers without the Popover API get the page dialog.
 */
let settlePop = null; // answers the open row popover; only one is ever pending

/** Whether the row popover is showing; ':popover-open' is a syntax error without the Popover API. */
const popOpen = () => !!HTMLElement.prototype.showPopover && $('#seqPop').matches(':popover-open');

function confirmAtRow(li, text) {
  const pop = $('#seqPop');
  const card = li?.querySelector('.seq-card');
  if (!pop.showPopover || !card) return confirmAction('Remove clip from sequence?', text);
  settlePop?.(false); // a popover for another row is still open: that one is cancelled
  const back = document.activeElement;
  $('#seqPopText').textContent = text;
  if (!pop.matches(':popover-open')) pop.showPopover();
  placePop(pop, card);
  $('#seqPopCancel').focus({ preventScroll: true });
  return new Promise((resolve) => {
    const hide = () => pop.matches(':popover-open') && pop.hidePopover();
    const onToggle = (e) => e.newState === 'closed' && cancel(); // Escape or a click outside
    const settle = (ok) => {
      if (settlePop !== settle) return;
      settlePop = null;
      pop.removeEventListener('toggle', onToggle);
      document.removeEventListener('scroll', cancel, { capture: true });
      removeEventListener('resize', cancel);
      resolve(ok);
    };
    function cancel() {
      if (settlePop !== settle) return;
      const inside = pop.contains(document.activeElement) || document.activeElement === document.body;
      settle(false);
      hide();
      // Keyboard and in-popover dismissals go back to the trash button; a click elsewhere keeps its target.
      if (inside && back?.isConnected) back.focus();
    }
    settlePop = settle;
    $('#seqPopOk').onclick = () => {
      settle(true);
      hide();
    };
    $('#seqPopCancel').onclick = cancel;
    pop.addEventListener('toggle', onToggle);
    document.addEventListener('scroll', cancel, { capture: true, passive: true });
    addEventListener('resize', cancel);
  });
}

async function confirmRemove(id, li) {
  const i = S.segments.findIndex((s) => s.id === id);
  if (i < 0) return;
  const name = S.segments[i].label || id;
  if (!(await confirmAtRow(li, `"${name}" is taken out of the sequence. No files are deleted.`))) return;
  removeSeg(id);
  $('#seqLive').textContent = `${name} removed from the sequence`;
  focusRow(i, 'remove');
}

async function removeSelected() {
  const first = S.segments.findIndex((s) => S.selected.has(s.id));
  const n = S.segments.filter((s) => S.selected.has(s.id)).length;
  if (!n) return;
  const what = `${n} clip${n > 1 ? 's' : ''}`;
  if (!(await confirmAction(`Remove ${what} from the sequence?`, 'Only the sequence changes. No files are deleted.'))) return;
  remember();
  S.segments.forEach((s, i) => S.selected.has(s.id) && keepDraft(s, i)); // as removeSeg
  S.segments = S.segments.filter((s) => !S.selected.has(s.id));
  S.selected.clear();
  renderAfterSeqChange();
  $('#seqLive').textContent = `${what} removed from the sequence`;
  focusRow(first, 'select');
}

/** Drop ticks of clips that left the sequence; sync the batch bar, row tints and checkboxes. */
function renderBatch() {
  const ids = new Set(S.segments.map((s) => s.id));
  for (const id of S.selected) if (!ids.has(id)) S.selected.delete(id);
  const n = S.selected.size;
  $('#seqBatch').hidden = !n;
  $('#seqSelCount').textContent = `${n} selected`;
  for (const li of $('#seqList').children) {
    const on = S.selected.has(li.dataset.id);
    li.classList.toggle('selected', on);
    const box = li.querySelector('[data-act="select"]');
    if (box) box.checked = on;
  }
}

/** Accessible names of a row's controls that carry the clip name. */
const ROW_LABELS = {
  select: (name) => `Select ${name}`,
  remove: (name) => `Remove ${name} from sequence`,
  up: (name) => `Move ${name} up`,
  down: (name) => `Move ${name} down`,
};

const seqDur = (s, range) => h('span', { class: 'seq-dur tc', text: `${(s.end - s.start).toFixed(1)} s` },
  h('span', { class: 'vh', text: `, ${range}` }), prepBadge(srcOf(s.source), s));

/** A row's title button (``head``: its rank and thumbnail) and rename pencil, when not renaming. */
function seqMain(s, head, name = s.label || s.id, range = `${fmt(s.start)} to ${fmt(s.end)}`) {
  const main = h('button', { type: 'button', class: 'seq-main', 'data-act': 'focus', title: range, onclick: () => focusMoment(s.id) },
    ...head,
    h('span', { class: 'seq-text' },
      h('span', { class: 'seq-title', text: name, title: name, ondblclick: () => startRename(s.id) }), seqDur(s, range)));
  const pencil = h('button', {
    type: 'button', class: 'seq-icon seq-edit', 'data-act': 'rename', 'aria-label': `Rename ${name}`,
    title: 'Rename', onclick: () => startRename(s.id),
  }, icon('edit'));
  return { main, pencil };
}

function seqRow(s, i, n) {
  const name = s.label || s.id;
  const range = `${fmt(s.start)} to ${fmt(s.end)}`;
  const head = [h('span', { class: 'rank tc', text: String(i + 1) }), thumb(s)];
  if (seqNow() === s.id) head[1].append(nowIcon());
  let main;
  let pencil;
  if (S.renaming === s.id) {
    // same-size stand-in, so the remove button does not jump when the rename closes on mousedown
    pencil = h('span', { class: 'seq-icon seq-edit-slot', 'aria-hidden': 'true' });
    const errId = `seqRenameErr-${i}`;
    const input = h('input', {
      type: 'text', class: 'seq-rename', 'data-act': 'rename', value: S.renameDraft, maxLength: LABEL_MAX,
      autocomplete: 'off', spellcheck: false, 'aria-label': `Name of clip ${i + 1}`, 'aria-describedby': errId,
      'aria-invalid': S.renameErr ? 'true' : null,
      oninput: () => {
        S.renameDraft = input.value;
        if (S.renameErr) {
          S.renameErr = '';
          input.removeAttribute('aria-invalid');
          $(`#${errId}`).textContent = '';
        }
      },
      onkeydown: (e) => {
        if (e.key === 'Enter' || e.key === 'Escape') {
          e.preventDefault();
          e.stopPropagation();
          finishRename(e.key === 'Enter', input);
        }
      },
      onblur: () => {
        if (S.renaming === s.id && input.isConnected) finishRename(true, input, true);
      },
    });
    main = h('div', { class: 'seq-main editing', title: range }, ...head,
      h('span', { class: 'seq-text' }, input,
        h('span', { id: errId, class: 'seq-rename-err', role: 'alert', text: S.renameErr }), seqDur(s, range)));
  } else {
    ({ main, pencil } = seqMain(s, head, name, range));
  }
  const li = h('li', {
    class: `seq${S.selected.has(s.id) ? ' selected' : ''}${seqNow() === s.id ? ' playing' : ''}`, 'data-id': s.id, 'aria-current': s.id === S.focus ? 'true' : null,
    ondragstart: (e) => {
      dragId = s.id;
      li.classList.add('dragging');
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', s.id);
    },
    ondragend: () => {
      dragId = null;
      li.draggable = false;
      li.classList.remove('dragging');
      for (const el of $('#seqList').children) el.classList.remove('drop');
    },
    ondragover: (e) => {
      if (!dragId) return;
      e.preventDefault();
      for (const el of $('#seqList').children) el.classList.toggle('drop', el.dataset.id === s.id && s.id !== dragId);
    },
    ondrop: (e) => {
      e.preventDefault();
      if (dragId) moveSeg(dragId, i);
    },
  },
    h('input', {
      type: 'checkbox', class: 'seq-check', 'data-act': 'select', checked: S.selected.has(s.id), 'aria-label': ROW_LABELS.select(name),
      onchange: (e) => {
        if (e.target.checked) S.selected.add(s.id);
        else S.selected.delete(s.id);
        renderBatch();
      },
    }),
    h('div', { class: 'seq-card', onclick: (e) => !e.target.closest('button,input,a') && focusMoment(s.id) },
      main,
      h('span', { class: 'seq-ctl' },
        h('span', { class: 'seq-acts' },
          pencil,
          h('button', {
            type: 'button', class: 'seq-icon seq-del', 'data-act': 'remove', 'aria-label': ROW_LABELS.remove(name),
            title: 'Remove from sequence', onclick: () => confirmRemove(s.id, li),
          }, icon('trash'))),
        h('span', { class: 'seq-move' },
          h('button', { type: 'button', class: 'seq-chev', 'data-act': 'up', disabled: i === 0, 'aria-label': ROW_LABELS.up(name), onclick: () => moveSeg(s.id, i - 1) }, icon('up')),
          h('button', { type: 'button', class: 'seq-chev', 'data-act': 'down', disabled: i === n - 1, 'aria-label': ROW_LABELS.down(name), onclick: () => moveSeg(s.id, i + 1) }, icon('down')))),
      // The handle arms dragging for the whole row; keyboard users reorder with the chevrons.
      h('span', {
        class: 'seq-handle', 'aria-hidden': 'true', title: 'Drag to reorder',
        onpointerdown: () => { li.draggable = true; },
        onpointerup: () => { if (!dragId) li.draggable = false; },
        onpointercancel: () => { if (!dragId) li.draggable = false; },
      }, icon('grip'))),
  );
  return li;
}

function renderSequence() {
  const segs = S.segments;
  const total = segs.reduce((n, s) => n + (s.end - s.start), 0);
  $('#seqTotal').textContent = segs.length ? `${segs.length} clip${segs.length > 1 ? 's' : ''}, ${fmtLen(total)}` : '';
  if (!S.project) setEmpty($('#seqEmpty'));
  else if (!segs.length) setEmpty($('#seqEmpty'), h('p', {}, 'Nothing in the sequence yet. Add moments from the list, then trim and reorder them here. Clips are exported in this order.'));
  else setEmpty($('#seqEmpty'));
  if (S.renaming && !segs.some((s) => s.id === S.renaming)) S.renaming = null;
  if (S.seqPlay && S.seqPlay.i >= 0 && !seqSeg(S.seqPlay.i)) seqStop('The playing clip left the sequence. Sequence stopped.');
  preserveFocus(() => {
    $('#seqList').replaceChildren(...segs.map((s, i) => seqRow(s, i, segs.length)));
  });
  renderBatch();
  renderRemoved();
  renderSeqPlay();
  renderProposal();
}

// ---------- play sequence ----------

const seqSeg = (j) => S.segments.find((s) => s.id === S.seqPlay?.order[j]);
const seqNow = () => S.seqPlay?.order[S.seqPlay.i];
const nowIcon = () => h('span', { class: 'seq-now', 'aria-hidden': 'true', title: 'Playing' }, icon('play'));

function renderSeqPlay() {
  const b = $('#seqPlayBtn');
  b.disabled = !S.seqPlay && !S.segments.length;
  b.textContent = S.seqPlay ? 'Stop' : 'Play sequence';
  if (S.seqPlay) b.setAttribute('aria-keyshortcuts', 'Escape');
  else b.removeAttribute('aria-keyshortcuts');
  b.title = S.seqPlay ? 'Stop (Escape)' : 'Play the clips in sequence order';
}

/** Mark the playing row (class and icon) without rebuilding the list, and keep it in view. */
function markPlaying() {
  const now = seqNow();
  for (const li of $('#seqList').children) {
    const on = !!now && li.dataset.id === now;
    li.classList.toggle('playing', on);
    const icn = li.querySelector('.seq-now');
    if (on && !icn) li.querySelector('.seq-thumb')?.append(nowIcon());
    if (!on) icn?.remove();
    if (on) revealRow(li);
  }
}

/** Scroll ``li`` into view inside its column only: when the page scrolls as a whole (narrow windows), it stays put, so the video stays on screen. */
function revealRow(li) {
  const col = li.closest('.col');
  if (!col || getComputedStyle(col).overflowY === 'visible') return;
  const r = li.getBoundingClientRect();
  const box = col.getBoundingClientRect();
  if (r.top < box.top) col.scrollTop -= box.top - r.top;
  else if (r.bottom > box.bottom) col.scrollTop += Math.min(r.bottom - box.bottom, r.top - box.top);
}

/** "Clip 2 of 5 · 0:34 / 2:10": the time played in the sequence, of the clips still in it. */
function seqClock(t) {
  const sp = S.seqPlay;
  let done = 0;
  let total = 0;
  sp.order.forEach((id, j) => {
    const s = seqSeg(j);
    if (!s) return;
    total += s.end - s.start;
    if (j < sp.i) done += s.end - s.start;
    else if (j === sp.i && !sp.wait) done += Math.min(Math.max(t - s.start, 0), s.end - s.start);
  });
  return `Clip ${sp.i + 1} of ${sp.order.length} · ${fmtLen(done)} / ${fmtLen(total)}`;
}

function seqStart() {
  if (!S.segments.length) return;
  for (const el of els) {
    if (!el.paused) continue;
    el.play().catch(() => {}); // inside the click: lets later play() calls start it (Safari)
    el.pause();
  }
  S.seqPlay = { order: S.segments.map((s) => s.id), i: -1, wait: false };
  banner('');
  renderSeqPlay();
  seqGo(0);
  if (S.seqPlay) seqPrepare(); // seqGo may have stopped it: nothing could be previewed
}

function seqStop(msg = 'Sequence stopped') {
  if (!S.seqPlay) return;
  S.seqPlay = null;
  cued = null;
  clearTimeout(S.seqTimer);
  v.pause();
  unload(spare()); // the visible player stays the stage player
  for (const el of els) el.preload = 'metadata'; // the stage preview buffers on demand again
  markPlaying();
  renderSeqPlay();
  updatePlayhead();
  if (msg) $('#seqLive').textContent = msg;
}

/** Ask for the proxies the sequence needs, in playback order: the server builds them first in, first out. */
async function seqPrepare() {
  const sp = S.seqPlay;
  if (!sp) return;
  const seen = new Set();
  for (let j = 0; j < sp.order.length; j += 1) {
    if (S.seqPlay !== sp) return; // stopped, or started again, while a request was out
    const src = srcOf(seqSeg(j)?.source);
    if (!src || seen.has(src.path)) continue;
    seen.add(src.path);
    if (src.status === 'needs-proxy' || src.status === 'unknown') await prepareSource(src);
  }
  await seqQueue();
  seqPoll();
}

const SEQ_AHEAD = 3; // clips after the playing one whose window proxies "Play sequence" queues

/**
 * Queue window proxies of the next clips no ready file covers yet, in playback order, behind the
 * playing clip's (``queue``: they do not supersede each other). Asked again on every poll, as the
 * server may drop them (a newer focus, or its queue is full).
 */
async function seqQueue() {
  const sp = S.seqPlay;
  if (!sp) return;
  let n = 0;
  for (let j = Math.max(sp.i, 0); j < sp.order.length && n < SEQ_AHEAD; j += 1) {
    if (S.seqPlay !== sp) return;
    const seg = seqSeg(j);
    const src = srcOf(seg?.source);
    if (!src || !['needs-proxy', 'preparing'].includes(src.status) || readyFor(src, seg)) continue;
    if (j > sp.i) n += 1;
    const dur = sourceDur(src.path);
    if ((src.pending ?? []).some((w) => coversSeg(w, seg, dur))) continue;
    await prepareSource(src, { lo: round2(seg.start), hi: round2(seg.end) }, true);
  }
}

/** While a clip of the sequence has no ready file covering it, refresh the project every 2 s (the stage polls only its own). */
function seqPoll() {
  clearTimeout(S.seqTimer);
  const sp = S.seqPlay;
  if (!sp || S.expired) return;
  const busy = (seg) => {
    const src = srcOf(seg?.source);
    return src && !readyFor(src, seg) && src.status !== 'failed';
  };
  if (!sp.order.some((id, j) => busy(seqSeg(j)))) return;
  S.seqTimer = setTimeout(async () => {
    await refreshProject();
    const now = S.seqPlay;
    if (!now) return;
    if (now.wait && srcOf(seqSeg(now.i)?.source)?.status === 'failed') seqGo(now.i); // skips it
    else cueNext();
    await seqQueue();
    seqPoll();
  }, 2000);
}

/** Load the clip after the playing one into the spare player, paused at its start. */
function cueNext() {
  const sp = S.seqPlay;
  const o = spare();
  let j = sp.i + 1;
  while (j < sp.order.length && !seqSeg(j)) j += 1;
  const cur = seqSeg(sp.i);
  const seg = seqSeg(j);
  const src = srcOf(seg?.source);
  const p = readyFor(src, seg) && fileFor(src, seg); // the whole source, or a window covering the clip
  if (!p || (cur && seg.source === cur.source && Math.abs(seg.start - cur.end) <= GAP && v.dataset.url === p.url)) {
    cued = null; // not playable yet, or it plays on in the visible player
    return;
  }
  if (cued?.id === seg.id && cued.start === seg.start && o.dataset.url === p.url) return;
  const { url, offset } = p;
  cued = { id: seg.id, start: seg.start, vt: seg.start - offset };
  if (o.dataset.url !== url) {
    o.preload = 'auto'; // buffer it, not only its metadata
    o.dataset.url = url;
    o.dataset.offset = offset;
    delete o.dataset.director; // "Play sequence" plays raw previews
    if (url === src.url) delete o.dataset.hi;
    else o.dataset.hi = p.hi; // a window proxy: the stage reads its range after the swap
    o.src = url; // seeks to cued.vt on loadedmetadata
  } else if (o.readyState >= HTMLMediaElement.HAVE_METADATA) o.currentTime = cued.vt;
}

/**
 * Play clip ``j`` of the sequence. Clips that left it are passed over and clips whose preview
 * could not be built are skipped with a note; after the last one the sequence stops. The next
 * clip waits in the spare player, so a step is a swap; otherwise the stage loads it (and waits
 * for its proxy).
 */
function seqGo(j) {
  const sp = S.seqPlay;
  const skipped = [];
  for (; j < sp.order.length; j += 1) {
    const seg = seqSeg(j);
    if (!seg) continue;
    const src = srcOf(seg.source);
    if (src && src.status !== 'failed') break;
    skipped.push(seg.label || seg.id);
  }
  if (skipped.length) banner(`Skipped ${skipped.map((n) => `"${n}"`).join(', ')}: the preview could not be built.`);
  if (j >= sp.order.length) return seqStop(sp.i < 0 ? 'Nothing in the sequence can be previewed.' : 'Sequence finished');
  const cur = seqSeg(sp.i);
  const seg = seqSeg(j);
  const src = srcOf(seg.source);
  const o = spare();
  sp.i = j;
  sp.wait = false;
  const url = readyFor(src, seg) && fileFor(src, seg).url; // the whole source, or a window covering the clip
  if (cur && seg.source === cur.source && Math.abs(playhead(cur) - seg.start) <= FRAME + GAP && v.dataset.url === url && !v.paused) {
    showMoment(seg.id, true); // contiguous: plays on without a cut
  } else if (cued?.id === seg.id && o.dataset.url === url && !o.error) {
    const old = v;
    v = o; // hard cut to the preloaded player
    seekWant = null;
    pendingSeek = null;
    if (o.readyState < HTMLMediaElement.HAVE_METADATA) pendingSeek = seg.start; // applied on loadedmetadata
    else if (Math.abs(o.currentTime - (seg.start - offsetOf())) > GAP) o.currentTime = seg.start - offsetOf();
    o.classList.add('active');
    old.classList.remove('active');
    v.play().catch(() => {});
    old.pause();
    showMoment(seg.id, true);
  } else {
    showMoment(seg.id); // loads the source (or a window proxy covering the clip), or waits for one
    if (url && v.dataset.url === url) v.play().catch(() => {});
    else sp.wait = true; // canplay starts it
  }
  cued = null;
  markPlaying();
  $('#seqLive').textContent = `Playing clip ${j + 1} of ${sp.order.length}: ${seg.label || seg.id}`;
  cueNext();
  updatePlayhead();
}

/** Removed clips that are no ranked moment: listed until reload, so they can be added back. */
function renderRemoved() {
  const ids = new Set(S.segments.map((s) => s.id));
  const gone = Object.entries(S.drafts).filter(([id, d]) => d.source && !ids.has(id));
  $('#removedPanel').hidden = !gone.length;
  preserveFocus(() => {
    $('#removedList').replaceChildren(
      ...gone.map(([id, d]) => {
        const name = d.label || id;
        return h('li', { class: 'cand', 'data-id': id, 'aria-current': id === S.focus ? 'true' : null },
          h('button', { type: 'button', class: 'cand-main', 'data-act': 'focus', onclick: () => focusMoment(id) },
            h('span', { class: 'cand-top' }, h('span', { class: 'cand-title', text: name })),
            h('span', { class: 'cand-time tc' }, `${fmt(d.start)} to ${fmt(d.end)}, ${(d.end - d.start).toFixed(1)} s in ${baseName(d.source)}`)),
          h('button', {
            type: 'button', class: 'btn small', 'data-act': 'add', 'aria-label': `Add ${name} back to sequence`,
            onclick: () => {
              const at = addSeg(id);
              $('#seqLive').textContent = `${name} added back to the sequence at position ${at + 1}`;
              focusRow(at, 'focus'); // its row here is gone; follow the clip into the sequence
            },
          }, 'Add back'),
        );
      }),
    );
  });
}

// ---------- save, export, assemble ----------

function renderSave() {
  const d = dirty();
  const el = $('#saveState');
  el.textContent = !S.project ? '' : S.saving ? 'Saving…' : d ? 'Unsaved changes' : 'Sequence saved';
  el.classList.toggle('unsaved', d);
  $('#saveBtn').disabled = !S.project || !d || S.saving;
}

async function saveReview() {
  if (!dirty()) return;
  const sent = key();
  S.saving = true;
  renderSave();
  try {
    await api(`/api/projects/${encodeURIComponent(S.id)}/review`, { method: 'PUT', body: { segments: JSON.parse(sent) } });
    S.savedKey = sent;
    banner('');
  } finally {
    S.saving = false;
    renderSave();
  }
}

function exportMatches() {
  const clips = S.project?.clips?.clips;
  if (!clips || clips.length !== S.segments.length) return false;
  return clips.every((c, i) => {
    const s = S.segments[i];
    const a = c.pick?.start ?? c.start;
    const b = c.pick?.end ?? c.end;
    return c.id === s.id && Math.abs(a - s.start) < 0.02 && Math.abs(b - s.end) < 0.02;
  });
}

function renderExport() {
  const p = S.project;
  const clips = p?.clips;
  const st = $('#exportState');
  // A 360 clip rendered from the dual-fisheye preview: no SDK, pano.auto_stitch off or the stitch skipped.
  // Color names the clips the grade left as is: a grade every clip skipped is not silent.
  const note = (clips?.clips?.some((c) => c.pano?.preview) ? ' Exported from the 360 preview (not stitched).' : '')
    + colorNote(clips?.clips);
  // The "Latest export ..." facts are advanced (.adv); warnings and guidance always show
  const latest = (rest) => h('span', { class: 'adv', text: `Latest export (${clips.dir ?? 'folder'}) has ${clips.clips.length} clips${rest}` });
  const say = (text) => (text ? h('span', { text }) : null);
  const show = (...parts) => st.replaceChildren(...parts.filter(Boolean)); // no 360 or Color note: say() gives null
  if (!p) show();
  else if (picks() === 'auto') {
    if (!p.highlights?.candidates?.length) show(say('Analyze the footage to export automatic picks.'));
    else if (clips) show(latest('.'), say(`${note} Automatic picks export the best moments of the analysis.`));
    else show(say('Nothing exported yet. Automatic picks export the best moments of the analysis.'));
  } else if (!S.segments.length) show(say('Add moments to the sequence to export them.'));
  else if (!clips) show(say('Nothing exported yet. Exporting saves the sequence first.'));
  else if (exportMatches()) show(latest(' and matches this sequence.'), say(note));
  else show(latest('.'), say(` The latest export differs from this sequence.${note} Export again to update it.`));
  renderBusy();
}

function renderBusy() {
  const p = S.project;
  const busy = !!S.active;
  const why = busy ? 'Another job is running.' : '';
  const set = (id, off, reason) => {
    const b = $(id);
    b.disabled = off;
    b.title = off ? reason : '';
  };
  set('#analyzeBtn', !p || busy, why);
  const big = $('#analyzeBig');
  if (big) {
    const own = S.active?.project_id === S.id && S.active?.action === 'analyze';
    const steps = own && !!stepsJob(); // an older CLI sends no steps: the Jobs log is all there is
    const reason = own ? (steps ? 'Analyzing… each step is listed below.' : 'Analyzing… progress is in Jobs.') : busy ? why
      : p?.source_error ? sourceErrorText(p) : 'Add videos to the source folder first.';
    set('#analyzeBig', !p || busy || !p.sources?.length, reason);
    set('#analyzeOptsBtn', !p || busy || !p.sources?.length, reason);
    const label = own ? 'Analyzing…' : 'Analyze footage';
    if (big.textContent !== label) big.textContent = label;
    const out = $('#analyzeWhy');
    out.classList.toggle('progress', own);
    if (out.textContent !== big.title) out.textContent = big.title; // live region: announce changes only
  }
  if (picks() === 'auto') set('#clipsBtn', !p || busy || !p.highlights?.candidates?.length, busy ? why : 'Analyze the footage first.');
  else set('#clipsBtn', !p || busy || !S.segments.length, busy ? why : 'Add moments to the sequence first.');
  // Without Speech-aware cuts and padding, extend has nothing to do (the CLI refuses it)
  const ext = extendOptions();
  const idle = !ext.speech_aware && !(ext.extend_start > 0) && !(ext.extend_end > 0);
  const extendWhy = busy ? why : !p?.clips ? 'Export clips first.' : 'Set Extend start or Extend end above 0, or turn Speech-aware cuts on.';
  set('#extendBtn', !p || busy || !p.clips || idle, extendWhy);
  const extendSay = $('#extendWhy'); // its aria-describedby hides the title from screen readers
  if (extendSay.textContent !== $('#extendBtn').title) extendSay.textContent = $('#extendBtn').title;
  const preset = p?.preset_problem || (p && !p.sources?.length ? 'Add videos to the source folder first.' : '');
  set('#makeBtn', !p || busy || !!preset, busy ? why : preset);
  const presetWhy = $('#presetWhy');
  presetWhy.hidden = !preset;
  if (presetWhy.textContent !== preset) presetWhy.textContent = preset;
  $('#presetPanel').disabled = !p;
  set('#assembleBtn', !p || busy || !p.clips, busy ? why : 'Export clips first.');
  // Never disabled by a job: showing a folder is harmless.
  const target = exportsFolder(p);
  const reveal = $('#revealExportBtn');
  reveal.hidden = !target || !S.canReveal;
  reveal.dataset.target = target ?? '';
  reveal.title = target === 'outputs' ? 'Shows the folder of the final video' : target ? 'Shows the folder of the exported clips' : '';
  $('#exportPanel').disabled = !p;
  renderSteps();
}

/**
 * The folder "Show Exports Folder" opens: the final video's ('outputs') when the current export
 * has one, else the export's clips; for a project with finals but no export (made on the command
 * line), theirs. Null while nothing has been exported.
 */
function exportsFolder(p) {
  const final = p?.outputs?.some((o) => /\.mp4$/i.test(o.name));
  if (p?.clips) return final && p.final_current ? 'outputs' : 'clips';
  return p?.outputs?.length ? 'outputs' : null;
}

// ---------- analysis steps ----------

const STEP_TEXT = {
  running: 'Running', done: 'Done', skipped: 'Skipped', failed: 'Failed', interrupted: 'Interrupted',
  cancelled: 'Cancelled', waiting: 'Waiting', notrun: 'Not run',
};
const STEP_ICON = { done: '✓', skipped: '–', failed: '✕', interrupted: '!', cancelled: '■', waiting: '○', notrun: '–' }; // running: a spinner
// The CLI's step names, in order (a stable API): analyze sends the first 6, edit all 8. Labels for
// steps not sent yet; a label the CLI sends wins.
const STEP_LABELS = {
  transcribe: 'Transcription', reactions: 'Reactions', keyframes: 'Keyframes', crashes: 'Crashes',
  ranking: 'Ranking', vision: 'Vision rating', clips: 'Clips', assemble: 'Assemble',
};
const STEP_NAMES = Object.keys(STEP_LABELS);

/**
 * The rows to show: the steps sent, plus the ones not sent yet, from the job's ``plan`` (the
 * server's list of the command's steps) or, without one, when ``total`` is a known step count (6
 * or 8): "waiting" while the job runs, "not run" after it ended (a cancel or fatal error).
 * Each row gets a display ``state``: a failed or interrupted step of a cancel reads "cancelled".
 */
function stepRows(j) {
  const { steps, total, plan } = j.progress;
  const cancel = (s) => s.detail === 'cancelled' || (s.state === 'interrupted' && j.status === 'cancelled');
  const rows = steps.map((s) => ({
    ...s,
    state: (s.state === 'failed' || s.state === 'interrupted') && cancel(s) ? 'cancelled' : s.state,
    detail: s.detail === 'cancelled' ? null : s.detail, // the state says it
  }));
  const planned = plan ?? (total === 6 || total === 8 ? STEP_NAMES.slice(0, total).map((step) => ({ step, label: STEP_LABELS[step] })) : null);
  if (planned) {
    const sent = new Set(steps.map((s) => s.step));
    planned.forEach(({ step, label }, i) => {
      if (!sent.has(step))
        rows.push({ step, index: i + 1, label, state: live(j.status) ? 'waiting' : 'notrun', pct: null, detail: null });
    });
    rows.sort((a, b) => a.index - b.index);
  }
  return rows;
}

/**
 * The analysis whose steps the Moments column shows: this project's running one, or, while it is
 * still the latest job, one that failed or was cancelled (a completed one gives way to its moments).
 * Null when it sent no steps and has no plan (a CLI without PROGRESS lines); a plan alone shows
 * its rows, all waiting, before the first step arrives.
 */
function stepsJob() {
  const j = S.active ?? S.jobs[S.jobs.length - 1];
  if (!S.id || j?.project_id !== S.id || j.action !== 'analyze' || j.status === 'completed') return null;
  return j.progress?.steps?.length || j.progress?.plan?.length ? j : null;
}

/** A step's whole pct while it runs, else null. */
const stepPct = (s) => (s.state === 'running' && s.pct != null ? Math.round(s.pct) : null);
// Reduced motion stills the spinner, so a running step shows its elapsed time instead: "Running 42% · 1:23".
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
const stepElapsed = (job, s) => {
  if (s.state !== 'running' || !reduceMotion.matches) return '';
  const since = (S.stepSince[`${job}/${s.step}`] ??= Date.now());
  return fmtLen(Math.floor((Date.now() - since) / 1000));
};
const stepStateText = (s, pct, job) => {
  const secs = stepElapsed(job, s);
  return `${STEP_TEXT[s.state] ?? s.state}${pct == null ? '' : ` ${pct}%`}${secs && (pct == null ? ` ${secs}` : ` · ${secs}`)}`;
};

/** Tick the running steps' elapsed time in place, also while polls bring nothing new; stop when none runs. */
function stepTick(j) {
  if (!reduceMotion.matches || !j?.progress.steps.some((s) => s.state === 'running')) {
    clearInterval(S.stepTimer);
    S.stepTimer = null;
    S.stepSince = {};
  } else S.stepTimer ??= setInterval(() => {
    const job = stepsJob();
    const rows = ($('#freshSteps') ?? $('#analyzeSteps')).querySelectorAll('.step-list > .step');
    const now = job ? stepRows(job) : [];
    if (rows.length !== now.length) return;
    now.forEach((s, i) => {
      if (s.state === 'running') rows[i].querySelector('.step-state').textContent = stepStateText(s, stepPct(s), job.id);
    });
  }, 1000);
}
// The setting changed mid-run: rebuild the rows, so the elapsed time appears or goes at once.
reduceMotion.addEventListener('change', () => {
  for (const box of [$('#freshSteps'), $('#analyzeSteps')]) if (box) delete box.dataset.key;
  renderSteps();
});

/** One step row: an icon (decorative), the label, its state in words, pct (running only) and detail when sent. */
function stepRow(s, job) {
  const running = s.state === 'running';
  const pct = stepPct(s);
  return h('li', { class: 'step', 'data-state': s.state },
    running ? h('span', { class: 'step-icon spin', 'aria-hidden': 'true' })
      : h('span', { class: 'step-icon', 'aria-hidden': 'true', text: STEP_ICON[s.state] ?? '•' }),
    h('span', { class: 'step-label', text: s.label }),
    h('span', { class: 'step-state', text: stepStateText(s, pct, job) }),
    s.detail ? h('span', { class: 'step-detail', text: s.detail }) : null,
    running ? (pct == null
      ? h('div', { class: 'indeterminate step-bar', 'aria-hidden': 'true' })
      : h('div', { class: 'step-bar', 'aria-hidden': 'true' }, h('span', { style: `width:${pct}%` })))
      : null,
  );
}

/**
 * Fill the steps box under "Analyze footage" (before analysis) or under the Moments header
 * ("Analyze again"). Rebuilt only when a step changed; it holds nothing focusable, so polling
 * never moves focus. The status line announces state changes, not every percent tick.
 */
function renderSteps() {
  const fresh = $('#freshSteps');
  const box = fresh ?? $('#analyzeSteps');
  if (fresh) $('#analyzeSteps').hidden = true;
  const j = stepsJob();
  box.hidden = !j;
  stepTick(j);
  if (!j) {
    box.replaceChildren();
    delete box.dataset.key;
    delete box.dataset.shape;
    return;
  }
  const { total } = j.progress;
  const steps = stepRows(j);
  const sig = JSON.stringify([j.id, total, steps]);
  if (box.dataset.key === sig) return;
  box.dataset.key = sig;
  // Only a pct or detail changed: update those rows in place, so a running spinner or
  // indeterminate bar keeps its animation instead of restarting on every tick.
  const shape = JSON.stringify([j.id, total, steps.map((s) => [s.step, s.state, s.label, stepPct(s) == null, !s.detail])]);
  const rows = box.querySelectorAll('.step-list > .step');
  if (box.dataset.shape === shape && rows.length === steps.length) {
    steps.forEach((s, i) => {
      const pct = stepPct(s);
      rows[i].querySelector('.step-state').textContent = stepStateText(s, pct, j.id);
      const detail = rows[i].querySelector('.step-detail');
      if (detail) detail.textContent = s.detail;
      const bar = rows[i].querySelector('.step-bar > span');
      if (bar) bar.style.width = `${pct}%`;
    });
    return;
  }
  box.dataset.shape = shape;
  // Announce the steps whose state changed since the last render of this job; not the
  // placeholders ("waiting", "not run"), which the list itself shows.
  const seen = S.stepStates?.job === j.id ? S.stepStates.states : {};
  const sent = steps.filter((s) => s.state !== 'waiting' && s.state !== 'notrun');
  const changed = sent.filter((s) => seen[s.step] !== s.state);
  S.stepStates = { job: j.id, states: Object.fromEntries(sent.map((s) => [s.step, s.state])) };
  const finished = steps.filter((s) => s.state === 'done' || s.state === 'skipped').length;
  const head = total ? `Analysis steps: ${finished} of ${total} finished` : 'Analysis steps';
  if (changed.length) // a sibling of the box, never hidden or rebuilt, so screen readers hear each change
    $(`#${box.id}Status`).textContent = changed
      .map((s) => `${s.label}: ${(STEP_TEXT[s.state] ?? s.state).toLowerCase()}${s.state !== 'running' && s.detail ? `, ${s.detail}` : ''}`)
      .join('. ');
  box.replaceChildren(
    h('p', { class: 'steps-head', id: `${box.id}Head`, text: head }),
    h('ol', { class: 'step-list', 'aria-labelledby': `${box.id}Head` }, ...steps.map((s) => stepRow(s, j.id))),
  );
}

function renderOutputs() {
  const outs = S.project?.outputs ?? [];
  const final = outs.find((o) => /\.mp4$/i.test(o.name)); // the latest export's, else the newest
  $('#outputs').hidden = !outs.length;
  const current = S.project?.final_current ?? true; // false: the current export is not assembled yet
  $('#outTitle').textContent = !final ? 'Outputs' : current ? 'Final video' : 'Earlier final';
  const fv = $('#finalVideo');
  fv.hidden = !final;
  if (final && fv.dataset.url !== final.url) {
    fv.dataset.url = final.url;
    fv.src = final.url;
  } else if (!final && fv.dataset.url) {
    unload(fv);
  }
  const dl = (u) => `${u}${u.includes('?') ? '&' : '?'}download=1`;
  $('#outList').replaceChildren(
    ...outs.map((o) => h('li', {}, h('a', { href: dl(o.url), download: o.name, text: `Download ${o.name}` }))),
  );
}

async function startJob(action, extra = {}) {
  if (!S.project) return;
  try {
    if (action === 'clips') await saveReview(); // clips use the saved sequence
    const job = await api(`/api/projects/${encodeURIComponent(S.id)}/jobs`, { method: 'POST', body: { action, ...extra } });
    S.tracked.set(job.id, 'running'); // so a job that ends between two polls still refreshes the project
    banner('');
    S.logOpen = true;
    pollJobs();
  } catch (e) {
    fail(e);
  }
}

/** Compress Audio Peaks as sent to the server: 'off', 'light' or 'strong'. */
function peaks() {
  return PEAK_STOPS[Number($('#optPeaks').value)] ?? 'off';
}

/** Color as sent to the server: 'off', 'auto' or 'underwater'. */
function color() {
  return COLOR_STOPS[Number($('#optColor').value)] ?? 'off';
}

/** Color and its Strength for Export clips and One-Click Edit; Strength only below 100% (the CLI's default is 1). */
function colorOptions() {
  const body = { color: color() };
  const pct = Number($('#optColorStrength').value);
  if (body.color !== 'off' && pct < 100) body.color_strength = pct / 100;
  return body;
}

/** 'sequence' (hand-picked) or 'auto' (Clips to Export). */
function picks() {
  return $('#optPicksAuto').checked ? 'auto' : 'sequence';
}

/** The fields among ``keys`` that are filled in and enabled, as numbers (ids: a list). */
function fields(...keys) {
  const out = {};
  for (const key of keys) {
    const el = $(OPT_FIELDS[key]);
    const v = el.value.trim();
    if (el.disabled || !v) continue;
    out[key] = key === 'ids' ? v.split(/[\s,]+/).filter(Boolean) : TEXT_FIELDS.has(key) ? v : Number(v);
  }
  return out;
}

/**
 * False (and the browser shows why) when one of the fields a job sends (``body``) is out of its
 * range. A hidden advanced field is shown first (until reload; the setting stays), so the browser
 * can point at it.
 */
function validOptions(body) {
  const bad = Object.keys(body).filter((key) => key in OPT_FIELDS).map((key) => $(OPT_FIELDS[key])).find((el) => !el.checkValidity());
  if (bad?.classList.contains('adv')) showAdvExport(true);
  bad?.reportValidity();
  return !bad;
}

/**
 * Crop as Export clips sends it (clips --crop): nothing for Default, auto or none, or Custom's box
 * as typed (validOptions checks it, blank included); nothing while Crop is off (360 projects).
 */
function cropOption() {
  const mode = $('#optCrop').value;
  if (!mode || $('#optCrop').disabled) return {};
  return { crop: mode === 'custom' ? $('#optCropBox').value.trim() : mode };
}

/** Extend export's options (Export clips sends them too): speech-aware cuts and the padding. */
function extendOptions() {
  return { speech_aware: $('#optSpeech').checked, ...fields('speech_max_extend', 'speech_max_lead', 'extend_start', 'extend_end') };
}

function exportOptions() {
  const body = {
    picks: picks(), vertical: $('#optVertical').checked, smart_tail: $('#optTail').checked, compress_peaks: peaks(),
    ...colorOptions(), ...extendOptions(), ...fields('tail_min', 'tail_max', 'event_window', 'size', 'audio_fade'),
    ...cropOption(),
  };
  if (body.picks === 'auto') Object.assign(body, fields('top', 'max_total', 'ids'), $('#optOrder').value ? { order: $('#optOrder').value } : {});
  if (!$('#optEventTails').disabled) body.event_tails = $('#optEventTails').value;
  return body;
}

/**
 * One-Click Edit (``edit --kind``): asks first when the preset's clips folder holds an earlier
 * run's clips, and only then sends ``force`` (the CLI replaces them only with --force).
 */
async function makePreset() {
  await refreshProject(); // preset_taken as it is now, not when the page last loaded
  if (!S.project) return;
  const body = { kind: $('#optKind').value, vertical: $('#optVertical').checked, compress_peaks: peaks(), ...colorOptions() };
  const dir = body.vertical ? 'clips_vertical' : 'clips';
  if (S.project?.preset_taken?.includes(dir)) {
    const text = `The ${dir} folder holds the clips of an earlier run. This run replaces the clips it lists and assembles final.mp4 again; other files stay.`;
    if (!(await confirmAction('Replace the clips of the earlier preset run?', text, 'Replace'))) return;
    body.force = true;
  }
  startJob('edit', body);
}

/** Assemble's number fields, by their OPT_FIELDS names (validOptions checks them). */
function assembleFields() {
  return fields('fade_dur', 'final_audio_fade');
}

function assembleOptions() {
  const { final_audio_fade: audioFade, ...rest } = assembleFields();
  const model = $('#optCaptionModel');
  return {
    transition: $('#optTransition').value,
    audio_transition: $('#optAudioTransition').value,
    loudnorm: $('#optLoudnorm').checked,
    compress_peaks: peaks(),
    captions: $('#optCaptions').checked,
    ...rest,
    ...(audioFade === undefined ? {} : { audio_fade: audioFade }),
    ...(model.value && !model.disabled ? { caption_model: model.value } : {}),
  };
}

// ---------- AI Chat (#346) ----------
// A turn is posted with the working sequence and polled like a job. The assistant only proposes: a
// proposal is previewed (S.proposal: the Sequence tab, the moments list and ghost trim handles on
// the rail) until Apply replaces the working sequence, which Save and Export use as before. Text
// from the model or the project is only ever set as text (h()'s text, textContent), never as markup.

const CHAT_POLL_MS = 1000;
const MAX_SUGGESTIONS = 3; // suggestion buttons under a proposal
const MAX_HISTORY_SHOWN = 20; // newest applied changes listed for Revert
const PROVIDER_NAMES = { ollama: 'Ollama', openai: 'OpenAI-compatible', anthropic: 'Anthropic' };
// What a hosted provider gets (context.py sends only these; file names, never paths)
const CONSENT_SENDS = "this project's moment labels, transcript snippets, file names (not full paths) and times, and your messages";
const CHANGE_NAMES = { added: 'added', removed: 'removed', moved: 'moved', trimmed: 'trimmed', relabelled: 'renamed' };
const SMALL_MODEL_NOTE = 'A small local model can miss moments or misread a request. Check each change in the preview before you apply it.';
const aiUrl = (path = '') => `/api/projects/${encodeURIComponent(S.id)}/assistant${path}`;
const momentName = (id) => (cands().find((c) => c.id === id) ?? S.segments.find((s) => s.id === id))?.label || id;
const REVERT = 'revert-';
const isRevert = (p) => p.id.startsWith(REVERT);
/** The sequence a proposal was made on, from its diff's "before" rows, as [id, start, end] in order. */
const baseOf = (diff) => JSON.stringify(diff.filter((r) => r.before).sort((a, b) => a.before.pos - b.before.pos)
  .map((r) => [r.id, r.before.start, r.before.end]));
/** The working sequence in the same shape. */
const seqShape = () => JSON.stringify(S.segments.map((s) => [s.id, s.start, s.end]));
/** A saved turn (chat.json) or a polled one, in one shape. */
const chatTurn = (t) => ({ id: t.id, user: t.user ?? t.text ?? '', status: t.status, error: t.error, events: [...(t.events ?? [])] });
const arrow = () => [h('span', { 'aria-hidden': 'true', text: ' → ' }), h('span', { class: 'vh', text: ' to ' })];

/** "2 added, 1 trimmed" for a proposal's diff. */
function changeCounts(diff) {
  const n = {};
  for (const r of diff ?? []) for (const c of r.changes) n[c] = (n[c] ?? 0) + 1;
  return Object.keys(CHANGE_NAMES).filter((c) => n[c]).map((c) => `${n[c]} ${CHANGE_NAMES[c]}`).join(', ') || 'no change';
}

/** The open project's chat, the turn still running (polled on) and the applied changes. */
async function loadAssistant() {
  const pid = S.id;
  clearTimeout(S.chatTimer);
  S.turn = null;
  const chat = S.chat = pid ? { pid, provider: null, consent: null, turns: [], error: '', loaded: false } : null;
  S.history = [];
  chatError('');
  renderChat();
  if (!chat) return;
  try {
    const [a, hist] = await Promise.all([api(aiUrl()), api(aiUrl('/history'))]);
    if (S.chat !== chat) return; // another project opened meanwhile
    chat.provider = a.provider;
    chat.consent = a.consent ?? null;
    chat.turns = (a.turns ?? []).map(chatTurn);
    S.history = hist.applied ?? [];
    const run = a.running?.[0];
    if (run) {
      chat.turns.push(chatTurn(run));
      S.turn = { id: run.id, next: run.next };
      S.chatTimer = setTimeout(pollTurn, CHAT_POLL_MS);
    }
  } catch (e) {
    if (S.chat !== chat) return;
    chat.error = e.message;
  }
  chat.loaded = true; // Send waits for this: the saved turns would replace one sent meanwhile
  renderChat();
}

function chatError(text) {
  $('#chatError').textContent = text;
}

async function sendChat(e) {
  e.preventDefault();
  const input = $('#chatInput');
  const text = input.value.trim();
  const chat = S.chat;
  if (S.turn) chatError('The assistant is still answering: wait for it, or cancel it.');
  if (!text || !chat?.loaded || S.turn || S.sending) return;
  S.sending = true;
  chatError('');
  renderChatForm();
  try {
    const t = await api(aiUrl('/turns'), { method: 'POST', body: { text, segments: JSON.parse(key()) } });
    if (S.chat !== chat) return;
    input.value = '';
    // its events, even those already in this answer, come in through pollTurn, which acts on them
    chat.turns.push(chatTurn({ ...t, status: 'running', events: [] }));
    if (t.provider) chat.provider = t.provider;
    S.turn = { id: t.id, next: 0 };
    $('#chatLive').textContent = 'Sent. The assistant is answering.';
    S.chatTimer = setTimeout(pollTurn, t.status === 'running' ? CHAT_POLL_MS : 0);
  } catch (err) {
    if (S.chat === chat) chatError(err.message);
  } finally {
    S.sending = false;
    renderChat();
    if (document.activeElement === document.body) input.focus(); // the Send button was disabled under it
  }
}

async function pollTurn() {
  clearTimeout(S.chatTimer);
  const run = S.turn;
  const chat = S.chat;
  if (!run || !chat || S.expired) return;
  const t = chat.turns.find((x) => x.id === run.id);
  try {
    const v = await api(aiUrl(`/turns/${encodeURIComponent(run.id)}?after=${run.next}`));
    if (S.turn !== run) return; // cancelled, cleared or another project meanwhile
    const news = v.events.length > 0 || v.status !== t.status;
    t.events.push(...v.events);
    t.status = v.status;
    t.error = v.error;
    run.next = v.next;
    if (v.provider) chat.provider = v.provider;
    for (const ev of v.events) onChatEvent(ev);
    if (v.status !== 'running') {
      S.turn = null;
      chatError(''); // a "still answering" or a failed poll no longer applies
      $('#chatLive').textContent = turnNews(t);
    }
    if (news) renderChat();
  } catch (e) {
    if (S.turn !== run) return;
    if (e.status === 404) { // the server restarted without it
      S.turn = null;
      t.status = 'failed';
      t.error = 'The server no longer has this reply. Ask again.';
      $('#chatLive').textContent = t.error;
      renderChat();
    } else chatError(e.message); // polling goes on
  }
  if (S.turn === run) S.chatTimer = setTimeout(pollTurn, CHAT_POLL_MS);
}

/** A new event of the polled turn: shown moments are highlighted, a proposal is previewed at once. */
function onChatEvent(ev) {
  if (ev.kind === 'show') {
    S.shown = new Set(ev.ids);
    markMoments();
  } else if (ev.kind === 'proposal') previewProposal(ev);
}

/** What the live region says when a turn ends. */
function turnNews(t) {
  const last = [...t.events].reverse().find((e) => ['reply', 'proposal', 'error'].includes(e.kind));
  if (t.status === 'cancelled') return 'Reply cancelled.';
  if (last?.kind === 'proposal') return `Proposed: ${last.summary}. It is previewed in the Sequence tab.`;
  if (last) return last.text;
  return t.error || 'The assistant gave no answer.';
}

async function cancelTurn() {
  const run = S.turn;
  if (!run) return;
  try {
    const v = await api(aiUrl(`/turns/${encodeURIComponent(run.id)}/cancel`), { method: 'POST' });
    if (S.turn !== run) return;
    S.turn = null;
    clearTimeout(S.chatTimer);
    const t = S.chat.turns.find((x) => x.id === run.id);
    Object.assign(t, chatTurn(v));
    $('#chatLive').textContent = turnNews(t);
    renderChat();
    $('#chatInput').focus(); // the Cancel button is gone
  } catch (e) {
    chatError(e.message);
  }
}

async function clearChat() {
  const chat = S.chat;
  if (!chat) return;
  if (!(await confirmAction('Clear the chat?', 'The conversation is deleted. Changes you applied stay in the sequence, and Applied changes keeps them for reverting.', 'Clear chat'))) return;
  try {
    await api(aiUrl('/chat'), { method: 'DELETE' });
    if (S.chat !== chat) return;
    clearTimeout(S.chatTimer);
    S.turn = null;
    chat.turns = [];
    S.shown = null;
    if (S.proposal && !isRevert(S.proposal)) S.proposal = null; // its turn is gone: it can no longer be applied; no edit is undone
    if (S.pendingProposal && !isRevert(S.pendingProposal)) S.pendingProposal = null;
    chatError('');
    renderProposal();
    renderChat();
    $('#chatLive').textContent = 'Chat cleared. Applied changes stay.';
    $('#chatInput').focus();
  } catch (e) {
    chatError(e.message);
  }
}

/** Preview proposal ``p`` ({id, summary, after, diff}), without changing the working sequence. */
function previewProposal(p) {
  if (S.applying) { // the page is waiting for Apply's answer: this one comes next
    S.pendingProposal = p;
    return;
  }
  S.proposal = { id: p.id, summary: p.summary, after: p.after, diff: p.diff, base_key: p.base_key, seqKey: baseOf(p.diff) };
  $('#propMsg').textContent = '';
  renderProposal();
}

/** A button's preview: in the Sequence tab, which gets the focus. */
function reviewProposal(p) {
  previewProposal(p);
  renderChat();
  showTab('sequence');
  $('#propTitle').focus();
}

/** The Sequence tab's preview of S.proposal, the marks in the moments list and the rail's ghost handles. */
function renderProposal() {
  const p = S.proposal;
  $('#panelSequence').classList.toggle('previewing', !!p);
  $('#propPanel').hidden = !p;
  if (p) {
    $('#propSummary').textContent = p.summary;
    const note = $('#propNote');
    note.hidden = seqShape() === p.seqKey;
    note.textContent = isRevert(p)
      ? 'The sequence changed since this preview. Apply previews the revert again on the sequence as it is now.'
      : 'The sequence changed since this was proposed. Apply makes the same edits to the sequence as it is now, if they still fit.';
    preserveFocus(() => $('#propList').replaceChildren(...p.diff.map(propRow)));
  }
  renderPropBar();
  markMoments();
  drawGhost();
}

function renderPropBar() {
  $('#propApply').disabled = !S.proposal || S.applying;
  $('#propApply').textContent = S.applying ? 'Applying…' : 'Apply';
  $('#propDiscard').disabled = !S.proposal || S.applying;
}

/** One row of the previewed sequence, with its change in words: Added, Removed, Moved 3 → 1, Trimmed 0:10.00–0:18.00 → .... */
function propRow(r) {
  const name = r.label || momentName(r.id);
  const at = r.after ?? r.before;
  const span = (s) => `${fmt(s.start)}–${fmt(s.end)}`;
  const marks = r.changes.map((c) => h('span', { class: `prop-mark ${c}` },
    c === 'moved' ? ['Moved ', String(r.before.pos), ...arrow(), String(r.after.pos)]
      : c === 'trimmed' ? ['Trimmed ', span(r.before), ...arrow(), span(r.after)]
        : c === 'relabelled' ? ['Renamed “', r.before.label, '”', ...arrow(), '“', r.after.label, '”']
          : c === 'added' ? 'Added' : 'Removed'));
  return h('li', { class: ['prop', ...r.changes].join(' '), 'data-id': r.id },
    h('button', { type: 'button', class: 'prop-main', 'data-act': 'focus', onclick: () => focusMoment(r.id) },
      h('span', { class: 'rank tc', text: r.after ? String(r.after.pos) : '–' }),
      h('span', { class: 'prop-text' },
        h('span', { class: 'prop-title', text: name }),
        h('span', { class: 'prop-time tc', text: `${fmt(at.start)} to ${fmt(at.end)}, ${(at.end - at.start).toFixed(1)} s` }),
        marks.length ? h('span', { class: 'prop-marks' }, marks) : null)));
}

/** Mark the moments the assistant showed or its proposal changes, by class only: the list itself is renderMoments'. */
function markMoments() {
  const rows = new Map((S.proposal?.diff ?? []).map((r) => [r.id, r.changes]));
  for (const li of $('#candList').children) {
    const id = li.dataset.id;
    const ch = rows.get(id) ?? [];
    li.classList.toggle('ai-shown', !!id && !!S.shown?.has(id));
    li.classList.toggle('ai-added', ch.includes('added'));
    li.classList.toggle('ai-removed', ch.includes('removed'));
    li.classList.toggle('ai-changed', ch.some((c) => c === 'moved' || c === 'trimmed' || c === 'relabelled'));
  }
}

/** Swap the working sequence for ``after``, keeping drafts of the clips it drops, as removeSeg does. */
function replaceSequence(after) {
  const ids = new Set(after.map((s) => s.id));
  S.segments.forEach((s, i) => !ids.has(s.id) && keepDraft(s, i));
  for (const s of after) delete S.drafts[s.id];
  S.segments = structuredClone(after);
}

async function applyProposal() {
  const p = S.proposal;
  if (!p || S.applying) return;
  const pid = S.id;
  S.applying = true;
  $('#propMsg').textContent = '';
  renderPropBar();
  try {
    // The server checks it against the sequence as it is now; a changed one gets the same edits again.
    const entry = await api(aiUrl(`/proposals/${encodeURIComponent(p.id)}/applied`), { method: 'POST', body: { segments: JSON.parse(key()), base_key: p.base_key } });
    if (S.id !== pid || S.proposal !== p) return;
    remember();
    replaceSequence(entry.after);
    S.proposal = null;
    S.history.push(entry);
    renderAfterSeqChange();
    renderChat();
    $('#sideLive').textContent = `${entry.rebased ? 'Applied to the sequence as it is now' : 'Applied'}: ${p.summary}. Ctrl+Z undoes it.`;
    $('#seqTitle').focus(); // the preview and its buttons are gone
  } catch (e) {
    if (S.proposal !== p) return;
    if (isRevert(p) && e.status === 409 && e.message.includes('preview it again')) {
      S.applying = false; // revertChange previews the new one at once
      const err = await revertChange({ id: p.id.slice(REVERT.length) });
      if (!err) S.pendingProposal = null; // the revert just previewed again stays in view
      if (err && S.proposal === p) $('#propMsg').textContent = `Can't apply: ${err}. Discard it.`;
      else if (!err) $('#sideLive').textContent = 'The sequence changed since that preview: the revert is previewed again on it. Check it, then Apply.';
      return;
    }
    $('#propMsg').textContent = e.status === 409 ? `Can't apply: ${e.message}. Discard it, or ask the assistant again.`
      : e.status === 404 ? 'The assistant no longer has this proposal (the chat was cleared or the server restarted). Discard it and ask again.'
        : e.message;
  } finally {
    S.applying = false;
    const next = S.pendingProposal;
    S.pendingProposal = null;
    if (next && S.id === pid) previewProposal(next);
    else renderPropBar();
  }
}

function discardProposal() {
  if (!S.proposal || S.applying) return;
  S.proposal = null;
  renderProposal();
  renderChat();
  $('#sideLive').textContent = 'Proposed change discarded';
  $('#seqTitle').focus();
}

/** History > Revert this change: a proposal back to the change's "before", previewed like any other.
 * Returns the error's message, or '' once previewed. */
async function revertChange(entry) {
  const chat = S.chat;
  chatError('');
  try {
    const p = await api(aiUrl(`/history/${encodeURIComponent(entry.id)}/revert`), { method: 'POST', body: { segments: JSON.parse(key()) } });
    if (S.chat === chat) reviewProposal(p);
    return '';
  } catch (e) {
    if (S.chat === chat) chatError(e.message);
    return e.message;
  }
}

const TOOL_TEXT = {
  find_moments: (ev) => `Searched for “${ev.query}”: ${ev.found} found`,
  get_moment: (ev) => `Looked at ${momentName(ev.id)}`,
  get_transcript: (ev) => `Read the transcript around ${momentName(ev.id)}`,
};

/** Chat bubbles of turn ``t``: the user's message, then the model's steps, its answer and its proposals. */
function turnItems(t) {
  const items = [h('li', { class: 'msg user', 'data-id': `${t.id}-u` }, h('span', { class: 'vh', text: 'You: ' }), h('p', { text: t.user }))];
  let query = '';
  for (const ev of t.events) {
    const id = `${t.id}-${ev.n}`;
    if (ev.kind === 'tool') {
      if (ev.name === 'find_moments') query = ev.query;
      items.push(h('li', { class: 'msg step', 'data-id': id, text: (TOOL_TEXT[ev.name] ?? (() => ev.name))(ev) }));
    } else if (ev.kind === 'show') items.push(showItem(ev, id, query));
    else if (ev.kind === 'reply') items.push(h('li', { class: 'msg ai', 'data-id': id }, h('span', { class: 'vh', text: 'Assistant: ' }), h('p', { text: ev.text })));
    else if (ev.kind === 'proposal') items.push(proposalItem(ev, id));
    else if (ev.kind === 'error') items.push(h('li', { class: 'msg ai bad', 'data-id': id, text: ev.text }));
  }
  if (t.status === 'running')
    items.push(h('li', { class: 'msg ai pending', 'data-id': `${t.id}-run` },
      h('span', { class: 'spin', 'aria-hidden': 'true' }), h('span', { text: 'Thinking…' }),
      h('button', { type: 'button', class: 'btn small', 'data-act': 'cancel', onclick: cancelTurn, text: 'Cancel' })));
  else if (t.status === 'cancelled') items.push(h('li', { class: 'msg step', 'data-id': `${t.id}-x`, text: 'Cancelled.' }));
  else if (t.error && !t.events.some((e) => e.kind === 'error')) items.push(h('li', { class: 'msg ai bad', 'data-id': `${t.id}-x`, text: t.error }));
  return items;
}

/** show_moments: the moments, highlighted in the list (a toggle) or, by the words the model searched for, filtered through the list's own filter. */
function showItem(ev, id, query) {
  const on = !!S.shown && S.shown.size === ev.ids.length && ev.ids.every((x) => S.shown.has(x));
  const n = ev.ids.length;
  return h('li', { class: 'msg ai', 'data-id': id },
    h('span', { class: 'vh', text: 'Assistant: ' }),
    h('p', { text: ev.note || `${n} moment${n > 1 ? 's' : ''} found` }),
    h('p', { class: 'msg-meta', text: ev.ids.map(momentName).join(', ') }),
    h('span', { class: 'msg-acts' },
      h('button', {
        type: 'button', class: 'btn small', 'data-act': 'show', 'aria-pressed': String(on),
        onclick: () => {
          S.shown = on ? null : new Set(ev.ids);
          markMoments();
          renderChat();
        },
      }, 'Highlight in list'),
      query ? h('button', {
        type: 'button', class: 'btn small', 'data-act': 'filter',
        onclick: () => {
          S.filter = query;
          $('#filter').value = query;
          renderMoments();
          const n = $('#candList').querySelectorAll('.cand').length;
          $('#chatLive').textContent = `Moments filtered by ${query}: ${n} shown${filtersOn(S.filters) ? '; Sort and filter is also on' : ''}`;
        },
      }, `Filter list by “${query}”`) : null));
}

/** A proposal: its summary and changes, Review change (the preview in the Sequence tab) and up to three suggestions, previewed the same way. */
function proposalItem(ev, id) {
  const applied = S.history.some((e) => e.proposal === ev.id);
  const previewing = S.proposal?.id === ev.id;
  const state = applied ? ', applied' : previewing ? ', previewed in the Sequence tab' : '';
  const sugs = (ev.suggestions ?? []).slice(0, MAX_SUGGESTIONS);
  return h('li', { class: `msg ai proposal${previewing ? ' previewing' : ''}`, 'data-id': id },
    h('span', { class: 'vh', text: 'Assistant proposes: ' }),
    h('p', { text: ev.summary }),
    h('p', { class: 'msg-meta', text: `${changeCounts(ev.diff)}${state}` }),
    h('span', { class: 'msg-acts' },
      h('button', { type: 'button', class: 'btn small primary', 'data-act': 'review', onclick: () => reviewProposal(ev) }, applied ? 'Review again' : 'Review change')),
    sugs.length ? h('span', { class: 'msg-acts', role: 'group', 'aria-label': 'Suggestions' },
      sugs.map((s, i) => h('button', {
        type: 'button', class: `btn small outline${S.proposal?.id === s.id ? ' on' : ''}`, 'data-act': `suggest${i}`,
        title: changeCounts(s.diff), onclick: () => reviewProposal(s),
      }, s.summary))) : null);
}

function historyRow(entry) {
  const when = String(entry.applied ?? '').replace('T', ' ').slice(0, 16);
  return h('li', { class: 'history-row', 'data-id': entry.id },
    h('span', { class: 'history-text' },
      h('span', { text: entry.summary || 'Change' }),
      h('span', { class: 'msg-meta tc', text: `${when}, ${changeCounts(entry.diff)}` })),
    h('button', { type: 'button', class: 'btn small', 'data-act': 'revert', onclick: () => revertChange(entry) }, 'Revert this change'));
}

function renderChatForm() {
  const ready = !!S.chat?.loaded && !!S.project?.highlights && !S.chat.consent?.needed && !S.chat.consent?.missing_key;
  const send = $('#chatSend');
  const had = document.activeElement;
  $('#chatInput').disabled = !ready;
  send.disabled = !ready || !!S.turn || S.sending;
  if (had === send && send.disabled) $('#chatInput').focus();
}

function renderChat() {
  const c = S.chat;
  const prov = c?.provider;
  $('#chatProvider').textContent = !c ? ''
    : prov ? prov.label // "Anthropic · claude-sonnet-5-5 · api.anthropic.com"
      : c.error ? `Unavailable: ${c.error}` : 'Connecting…';
  const note = $('#chatNote');
  note.hidden = !prov?.local;
  note.textContent = prov?.local ? SMALL_MODEL_NOTE : '';
  // A missing key comes first: no agreeing to send to a provider that cannot be asked yet.
  const noKey = c?.consent?.missing_key || '';
  const consent = !noKey && c?.consent?.needed ? c.consent : null;
  $('#chatConsent').hidden = !consent && !noKey;
  $('#chatConsentTitle').textContent = noKey ? 'API key needed' : 'Send project data to a hosted model?';
  $('#chatConsentOk').hidden = !consent;
  $('#chatConsentText').textContent = noKey ? `${noKey}.`
    : consent
    ? `Each message sends ${CONSENT_SENDS} to ${PROVIDER_NAMES[prov?.name] ?? prov?.name ?? 'the provider'} at ${consent.host}. Nothing is sent until you agree. You agree once for this provider and host; Cutpoint saves it in config.json. Change the provider in Settings.`
    : '';
  $('#chatClear').disabled = !c?.turns.length;
  const log = $('#chatLog');
  const items = c ? c.turns.flatMap(turnItems) : [];
  const col = log.closest('.col');
  const atEnd = !col || col.scrollHeight - col.scrollTop - col.clientHeight < 24;
  const count = log.children.length;
  preserveFocus(() => log.replaceChildren(...(items.length ? items : [h('li', { class: 'msg step' },
    !S.project ? 'Open a project to chat about its moments.'
      : !S.project.highlights ? 'Analyze the footage first; the assistant works with its moments.'
        : 'Ask for moments to find, add, remove, reorder or trim. Each change is previewed in the Sequence tab before you apply it.')])));
  if (items.length && items.length !== count && atEnd && !$('#panelAssistant').hidden) revealRow(log.lastElementChild);
  const shown = S.history.slice(-MAX_HISTORY_SHOWN).reverse();
  $('#chatHistory').hidden = !shown.length;
  $('#chatHistory summary').textContent = `Applied changes (${S.history.length})`;
  preserveFocus(() => $('#historyList').replaceChildren(...shown.map(historyRow)));
  renderChatForm();
}

/** Agree once to send project data to the hosted provider in use; the server checks it is still the one shown. */
async function giveConsent() {
  const chat = S.chat;
  const key = chat?.consent?.key;
  if (!key) return;
  chatError('');
  try {
    const r = await api('/api/assistant/consent', { method: 'POST', body: { key } });
    if (S.chat !== chat) return;
    chat.consent = r.consent;
  } catch (e) {
    if (S.chat !== chat) return;
    if (e.status === 409) await loadAssistant(); // the provider changed: show the one in use
    return chatError(e.message);
  }
  renderChat();
  if (!chat.consent?.needed) $('#chatInput').focus(); // the button is gone
}

function bindAssistant() {
  $('#chatForm').addEventListener('submit', sendChat);
  $('#chatInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      $('#chatForm').requestSubmit();
    }
  });
  $('#chatClear').addEventListener('click', clearChat);
  $('#chatConsentOk').addEventListener('click', giveConsent);
  $('#propApply').addEventListener('click', applyProposal);
  $('#propDiscard').addEventListener('click', discardProposal);
}

// ---------- jobs ----------

async function pollJobs() {
  clearTimeout(S.jobTimer);
  try {
    const { jobs } = await api('/api/jobs');
    const finished = jobs.filter((j) => live(S.tracked.get(j.id)) && !live(j.status));
    const fresh = jobs.some((j) => live(j.status) && !S.tracked.has(j.id));
    for (const j of jobs) S.tracked.set(j.id, j.status);
    S.jobs = jobs;
    S.active = jobs.find((j) => live(j.status)) ?? null;
    if (fresh) S.logOpen = true;
    if (finished.some((j) => j.status === 'failed')) S.logOpen = true;
    renderJobs();
    renderBusy();
    if (finished.some((j) => j.project_id === S.id)) refreshProject();
  } catch (e) {
    if (S.expired) return; // the banner says why; polling stops
    $('#jobStatus').textContent = e.message;
    $('#jobStatus').dataset.status = 'failed';
  }
  S.jobTimer = setTimeout(pollJobs, S.active ? 1000 : 5000);
}

function jobLine(j) {
  const name = S.projects.find((p) => p.id === j.project_id)?.name ?? j.project_id;
  const n = JOB_NAMES[j.action] ?? j.action;
  if (j.status === 'running') return `${n} running for ${name}`;
  if (j.status === 'cancelling') return `${n} stopping for ${name}`;
  if (j.status === 'completed') return `${n} finished for ${name}`;
  if (j.status === 'cancelled') return `${n} cancelled for ${name}`;
  return `${n} failed for ${name}${j.error ? `: ${j.error}` : ''}`;
}

function renderJobs() {
  const j = S.active ?? S.jobs[S.jobs.length - 1];
  const st = $('#jobStatus');
  st.textContent = j ? jobLine(j) : 'No job has run yet';
  st.dataset.status = j?.status ?? '';
  $('#running').hidden = !S.active;
  const cancel = $('#cancelBtn');
  cancel.hidden = !S.active;
  cancel.disabled = S.active?.status === 'cancelling';
  const log = $('#log');
  const lines = j?.logs ?? [];
  const open = S.logOpen && (lines.length > 0 || !!S.active);
  log.hidden = !open;
  $('#logToggle').textContent = S.logOpen ? 'Hide log' : 'Show log';
  $('#logToggle').setAttribute('aria-expanded', String(S.logOpen));
  $('#logToggle').hidden = !j;
  if (open) {
    const atEnd = log.scrollHeight - log.scrollTop - log.clientHeight < 24;
    const text = lines.join('\n') || 'Waiting for output…';
    if (log.textContent !== text) {
      log.textContent = text;
      if (atEnd) log.scrollTop = log.scrollHeight;
    }
  }
}

// ---------- settings ----------

const PROJECTS_NOTE = 'New projects are created in this folder. Existing projects are not moved; the project list then shows the projects in the new folder.';

/** Fill and open the settings dialog; focus returns to the gear button when it closes. */
async function openSettings() {
  const d = $('#settingsDialog');
  const input = $('#setProjects');
  const box = $('#setVision');
  $('#settingsError').textContent = $('#settingsNote').textContent = '';
  let st;
  try {
    st = await api('/api/settings');
  } catch (e) {
    return fail(e);
  }
  input.value = st.projects_dir;
  input.readOnly = st.projects_dir_locked;
  input.dataset.saved = st.projects_dir;
  $('#setProjectsPick').hidden = !S.canPick || st.projects_dir_locked;
  $('#setProjectsNote').textContent = st.projects_dir_locked
    ? 'Set by the CUTPOINT_PROJECTS_DIR environment variable. Change or unset it, then restart cutpoint ui, to choose the folder here.'
    : `${PROJECTS_NOTE} Saved in ${st.config_file}.`;
  box.checked = st.vision === true;
  box.dataset.saved = String(box.checked);
  $('#setVisionHint').textContent = `Off by default. Slower, and uses several GB of GPU memory while it runs; needs Ollama with ${st.vision_model}. Every analysis, new or "Analyze again", uses this setting; "Analyze again" keeps the Ollama model, URL and vision-top the last analysis recorded. Saved in ${st.config_file}.${S.project?.kind === '360' ? ' Vision rating is not used for 360 footage.' : ''}`;
  fillAiSettings(st.assistant);
  $('#setNerdStats').checked = S.nerdStats;
  S.nerdStatsWas = S.nerdStats; // Cancel or Escape goes back to this
  $('#setAdvExport').checked = S.advExport;
  S.advExportWas = S.advExport;
  S.themeWas = $('#setTheme').value = theme();
  d.showModal();
  // a locked folder cannot be edited: start on the first control that can
  (st.projects_dir_locked ? box : input).focus();
}

async function saveSettings() {
  const input = $('#setProjects');
  const box = $('#setVision');
  const err = $('#settingsError');
  const note = $('#settingsNote'); // progress and remarks; err is for what went wrong
  const value = input.value.trim();
  err.textContent = note.textContent = '';
  const body = {};
  if (!input.readOnly) {
    if (!value) return (err.textContent = 'Enter the full path of a folder.');
    if (value !== input.dataset.saved) body.projects_dir = value;
  }
  if (String(box.checked) !== box.dataset.saved) body.vision = box.checked;
  const ai = aiChanges();
  if (ai && $('#setAiModel').value === AI_OTHER && !aiModel()) {
    err.textContent = 'Enter the model id.';
    return $('#setAiModelOther').focus();
  }
  if (ai && !ai.model && !aiCustom() && aiOffer()?.model_required) {
    err.textContent = `Choose a model for ${aiOffer().title}.`;
    return $('#setAiModel').focus();
  }
  if (ai) body.assistant = ai;
  if (!Object.keys(body).length) return $('#settingsDialog').close();
  if (body.projects_dir && dirty() && !confirm('The project list will show the new folder. Discard the unsaved changes to this sequence?')) return;
  $('#settingsSave').disabled = true;
  if (ai) note.textContent = 'Saving, and checking that the assistant can answer…';
  let st;
  try {
    st = await api('/api/settings', { method: 'PUT', body });
  } catch (e) {
    err.textContent = e.message;
    return;
  } finally {
    $('#settingsSave').disabled = false;
    note.textContent = '';
  }
  err.textContent = '';
  input.value = input.dataset.saved = st.projects_dir;
  box.dataset.saved = String(st.vision === true);
  if (ai) {
    fillAiSettings(st.assistant);
    if (S.id && !body.projects_dir) loadAssistant(); // the chat names the provider in use
    if (st.assistant_check && !body.projects_dir) {
      // saved, but it cannot answer yet: say why here, where it can be fixed
      err.textContent = `Saved. The assistant can't answer yet: ${st.assistant_check}`;
      return;
    }
    if (st.assistant_note && !body.projects_dir) {
      // saved without a full check (consent pending, or a key not proven): neutral, not the error line
      note.textContent = `Saved. ${st.assistant_note}`;
      return;
    }
  }
  $('#settingsDialog').close();
  if (!body.projects_dir) return;
  // A project of the same id in the new folder is another project: start afresh either way.
  try {
    const was = S.id;
    closeProject();
    banner('');
    await loadProjects();
    const next = S.projects.find((p) => p.id === was) ?? S.projects[0];
    if (next) await openProject(next.id);
  } catch (e) {
    fail(e);
  }
}

// The AI assistant's provider (#346): a preset per provider, so no base URL is typed. The page
// never gets an API key: only where each comes from.
const KEY_WHERE = { env: 'set (env)', keyring: 'set (keyring)' };
const AI_CUSTOM = 'custom'; // a base URL from config.json that no preset has: shown and kept (URL read-only)
const AI_HINTS = {
  ollama: 'Runs with Ollama (the default), at OLLAMA_HOST or the Ollama URL in config.json. On this machine nothing leaves it; an Ollama on another machine counts as hosted.',
  anthropic: "Claude through Anthropic's API (api.anthropic.com), with your API key. It asks once before it gets project data.",
  openai: "OpenAI's API (api.openai.com), with your API key. It asks once before it gets project data.",
  xai: "Grok through xAI's API (api.x.ai), with your API key. It asks once before it gets project data.",
  custom: 'The provider and base URL set in config.json, which no preset has. It keeps working, and its model can be changed here; choose a preset and save to use that instead.',
};
const AI_MODEL_HINTS = {
  ollama: 'The name of a model Ollama has pulled. Default: qwen3.5:9b when Ollama has it, else the vision model.',
  anthropic: 'Default: claude-sonnet-5-5. claude-opus-5-5 is stronger, slower and costs more.',
  openai: "Needed: the name of an OpenAI model your key can use. It is asked through OpenAI's Responses API, so gpt-6-astra, gpt-6.1-sol, gpt-6.1-terra, gpt-6-luna and *-codex models all make tool calls.",
  xai: 'Needed: the name of an xAI model your key can use.',
  custom: "The model of the custom provider, saved in config.json. Default: the server's model, when it serves only one.",
};
const AI_TYPE_ID = 'Choose Other model… to enter an id.';
const AI_LIST_STATES = {
  key_needed: `The model list loads once there is an API key. ${AI_TYPE_ID}`,
  consent_needed: AI_TYPE_ID,
  unsupported: AI_TYPE_ID,
};

/** Fill the AI assistant fields from the settings' assistant view. */
function fillAiSettings(a) {
  if (!a) return;
  S.ai = a;
  S.aiModels = {}; // preset -> the models its provider listed
  const sel = $('#setAiProvider');
  const options = a.providers.map((p) => h('option', { value: p.name, text: p.title }));
  if (a.custom) options.push(h('option', { value: AI_CUSTOM, text: 'Custom (from config.json)' }));
  sel.replaceChildren(...options);
  sel.value = sel.dataset.saved = a.custom ? AI_CUSTOM : a.provider;
  $('#setAiModel').dataset.saved = a.model;
  $('#setAiUrl').value = a.url;
  $('#setAiKeyInput').value = '';
  renderAiFields(a.model);
  loadAiModels(false);
}

const aiCustom = () => $('#setAiProvider').value === AI_CUSTOM;
/** The offer of the chosen preset; for Custom, of the configured provider. */
const aiOffer = () => {
  const name = aiCustom() ? S.ai?.provider : $('#setAiProvider').value;
  return S.ai?.providers.find((p) => p.name === name);
};

/** The model chosen: the select's, or the id typed under "Other model…"; '' is the default. */
const aiModel = () =>
  $('#setAiModel').value === AI_OTHER ? $('#setAiModelOther').value.trim() : $('#setAiModel').value;

/** The hints, model field, base URL (Custom only) and key status of the provider chosen;
 * `model` shows that model instead of the one chosen (see renderAiModelList). */
function renderAiFields(model) {
  const p = aiOffer();
  if (!p) return;
  const custom = aiCustom();
  $('#setAiProviderHint').textContent = AI_HINTS[custom ? AI_CUSTOM : p.name] ?? '';
  $('#setAiModel').setAttribute('aria-required', String(!custom && p.model_required));
  renderAiModelList(model);
  $('#setAiModelHint').textContent = AI_MODEL_HINTS[custom ? AI_CUSTOM : p.name] ?? '';
  $('#setAiUrlField').hidden = $('#setAiUrlHint').hidden = !custom;
  $('#setAiUrlHint').textContent = custom ? 'Set in config.json (assistant.url); edit that file to change it.' : '';
  const keyed = p.name in (S.ai.keys ?? {}); // key_env may be empty: a keyring key only (#504)
  $('#setAiKey').hidden = !keyed;
  if (!keyed) return;
  const where = S.ai.keys[p.name];
  const host = S.ai.key_hosts?.[p.name] ?? '';
  // The key is for the saved base URL's host: a preset of the provider a custom URL is saved for
  // has another host, so save that first. Every other choice's key is for its official host.
  const unsaved = !custom && S.ai.custom && p.name === S.ai.provider;
  $('#setAiKeyStatus').textContent = unsaved
    ? 'API key: save the settings first to see or set the key for this provider.'
    : `API key${host ? ` for ${host}` : ''}: ${KEY_WHERE[where] ?? 'not set'}`;
  $('#setAiKeySave').disabled = $('#setAiKeyClear').disabled = unsaved;
  $('#setAiKeyHint').textContent = unsaved
    ? "A key belongs to the base URL's host: the provider's own variable counts only for its official API host, and a key goes only over https:// unless the server is on this machine."
    : `${aiKeyOrder(p.key_env)} Sent only to ${host || 'the base URL'}, over https:// unless the server is on this machine. Never saved in config.json and never shown.`;
}

/** Where a key is read from, in order (#504): the provider's own variable before a saved key;
 *  a saved key before CUTPOINT_ASSISTANT_API_KEY, which is only for custom base URLs. */
function aiKeyOrder(env) {
  if (!env.length) return 'Read from the system keyring.';
  if (env[0] === 'CUTPOINT_ASSISTANT_API_KEY') {
    return `Read from the system keyring, else from ${env[0]} when cutpoint ui starts.`;
  }
  return `Read from ${env.join(' or ')} when cutpoint ui starts, else from the system keyring.`;
}

/** The Model select (#521): the default, the models the provider listed, "Other model…" last.
 * Without `model` the choice made stays, an id typed under Other too; a model the list lacks
 * stays chosen as an extra option, so opening Settings never changes it. */
function renderAiModelList(model) {
  const p = aiOffer();
  if (!p) return;
  const sel = $('#setAiModel');
  const input = $('#setAiModelOther');
  const typing = model === undefined && sel.value === AI_OTHER;
  const current = model ?? aiModel();
  const custom = aiCustom();
  const { options, value } = aiModelChoices(current, S.aiModels?.[$('#setAiProvider').value], {
    typing, required: !custom && p.model_required, placeholder: custom ? '' : p.placeholder,
    fallback: custom ? [] : p.models, // the preset's models when no list came
  });
  sel.replaceChildren(...options.map((o) => h('option', { value: o.value, text: o.text })));
  sel.value = value;
  if (!typing) input.value = '';
  $('#setAiModelOtherField').hidden = value !== AI_OTHER;
}

/** Ask the server for the models the chosen preset's provider offers (cached there; `refresh`
 * asks the provider again) and offer them in the Model list. Other model… takes any other id. */
async function loadAiModels(refresh) {
  const sel = $('#setAiProvider');
  const status = $('#setAiModelsStatus');
  const choice = sel.value;
  const seq = (S.aiModelsSeq = (S.aiModelsSeq ?? 0) + 1);
  status.textContent = '';
  if (!S.ai) return;
  status.textContent = 'Loading the model list…';
  let r;
  try {
    r = await api(`/api/assistant/models?provider=${encodeURIComponent(choice)}${refresh ? '&refresh=1' : ''}`);
  } catch (e) {
    r = { state: 'error', models: [], message: e.message };
  }
  if (seq !== S.aiModelsSeq || sel.value !== choice) return; // another choice since
  S.aiModels[choice] = r.models ?? [];
  renderAiModelList(); // only the list: the key status (a keyring confirmation) stays
  const n = S.aiModels[choice].length;
  if (r.state === 'ok') {
    status.textContent = n
      ? `${n} model${n === 1 ? '' : 's'} listed by the provider. Choose Other model… to enter another.`
      : `The provider listed no models. ${AI_TYPE_ID}`;
  } else if (r.state === 'error') {
    status.textContent = `Could not load the model list: ${String(r.message ?? '').replace(/\.?$/, '.')} ${AI_TYPE_ID}`;
  } else {
    const msg = r.message && String(r.message).replace(/\.?$/, '.');
    status.textContent = [msg, AI_LIST_STATES[r.state]].filter(Boolean).join(' ');
  }
}

/** The preset and model when either differs from what config.json has, else null. A preset
 * always sends an empty base URL (its own); Custom sends only a changed model, so its provider
 * and base URL stay as config.json has them. */
function aiChanges() {
  if (!S.ai) return null;
  const sel = $('#setAiProvider');
  const saved = $('#setAiModel').dataset.saved;
  if (aiCustom()) return aiModel() !== saved ? { model: aiModel() } : null;
  const now = { provider: sel.value, model: aiModel(), url: '' };
  return now.provider !== sel.dataset.saved || now.model !== saved ? now : null;
}

/** Save the typed key in the system keyring, or remove the saved one; the field is emptied either way. */
async function saveAiKey(clear) {
  const p = aiOffer();
  const input = $('#setAiKeyInput');
  const err = $('#settingsError');
  err.textContent = '';
  const key = input.value.trim();
  if (!p) return;
  if (!clear && !key) {
    err.textContent = 'Paste the API key first.';
    return input.focus();
  }
  input.value = '';
  try {
    const r = await api(clear ? '/api/assistant/key/clear' : '/api/assistant/key',
      { method: 'POST', body: clear ? { provider: p.name } : { provider: p.name, key } });
    S.ai.keys = r.keys;
    S.ai.key_hosts = r.key_hosts;
  } catch (e) {
    err.textContent = e.message;
    return;
  }
  renderAiFields();
  $('#setAiKeyStatus').textContent += clear
    ? '. Removed from the keyring.'
    : '. Saved in the system keyring; it is checked when you send the first message.';
  if (!clear) loadAiModels(false); // the list may load now that there is a key
  if (S.id) loadAssistant();
}

/** Turn Stats for Nerds on or off: store it in this browser, show or hide the overlay and tick the box to match. */
function setNerdStats(on) {
  S.nerdStats = on;
  try {
    if (on) localStorage.setItem(NERD_KEY, '1');
    else localStorage.removeItem(NERD_KEY);
  } catch {
    // storage blocked or full: the setting still applies until reload
  }
  $('#setNerdStats').checked = on;
  renderNerdStats();
}

/**
 * Stats for Nerds is cosmetic, so it lives in this browser's storage, not in config.json. The box
 * previews on change; Save keeps it and Cancel or Escape puts back the state the dialog opened
 * with. Without storage it lasts the session.
 */
function initNerdStats() {
  try {
    S.nerdStats = localStorage.getItem(NERD_KEY) === '1';
  } catch {
    // storage blocked: start from the default (off)
  }
  $('#setNerdStats').addEventListener('change', (e) => setNerdStats(e.target.checked));
}

/** Show or hide the Export tab's advanced controls (.adv) and say which hidden ones are in use. */
function showAdvExport(on) {
  $('#panelExport').classList.toggle('show-adv', on);
  renderAdvInUse();
}

/**
 * Hidden advanced fields keep their values and are still sent, so a filled one that a job sends
 * (enabled: the pick mode and boxes use it) is named under the options instead of acting unseen.
 */
function renderAdvInUse() {
  const out = $('#advInUse');
  const shown = $('#panelExport').classList.contains('show-adv');
  const used = shown ? [] : [...Object.values(OPT_FIELDS), ...Object.values(OPT_SELECTS)].map((id) => $(id))
    .filter((el) => el.classList.contains('adv') && !el.disabled && el.value.trim())
    .map((el) => document.querySelector(`label[for=${el.id}]`).textContent.replace(/ \(s\)$/, ''));
  out.hidden = !used.length;
  out.textContent = used.length ? `Advanced options in use: ${used.join(', ')}. Settings > Show advanced export options shows them.` : '';
}

/** Turn Show advanced export options on or off: store it in this browser, show or hide them and tick the box to match. */
function setAdvExport(on) {
  S.advExport = on;
  try {
    if (on) localStorage.setItem(ADV_KEY, '1');
    else localStorage.removeItem(ADV_KEY);
  } catch {
    // storage blocked or full: the setting still applies until reload
  }
  $('#setAdvExport').checked = on;
  showAdvExport(on);
}

/**
 * Show advanced export options (off by default) is a view choice, so like Stats for Nerds it lives
 * in this browser's storage, previews on change, and Cancel or Escape puts it back.
 */
function initAdvExport() {
  try {
    S.advExport = localStorage.getItem(ADV_KEY) === '1';
  } catch {
    // storage blocked: start from the default (off)
  }
  showAdvExport(S.advExport);
  $('#setAdvExport').addEventListener('change', (e) => setAdvExport(e.target.checked));
}

/** The theme in use: index.html set data-theme on <html> before first paint; none is Blue. */
function theme() {
  return document.documentElement.dataset.theme ?? 'blue';
}

/** Switch the theme (#445): recolour the page at once and store it in this browser; Blue is stored as no key. */
function setTheme(name) {
  if (!THEMES.includes(name)) name = 'blue';
  if (name === 'blue') delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = name;
  try {
    if (name === 'blue') localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, name);
  } catch {
    // storage blocked or full: the theme still applies until reload
  }
  $('#setTheme').value = name;
}

function bindSettings() {
  const d = $('#settingsDialog');
  const gear = $('#settingsBtn');
  gear.append(icon('gear'));
  gear.addEventListener('click', openSettings);
  d.addEventListener('close', () => gear.focus());
  $('#setTheme').addEventListener('change', (e) => setTheme(e.target.value));
  const undo = () => {
    setNerdStats(S.nerdStatsWas);
    // a field revealed for an out-of-range value (showAdvExport(true)) stays unless the box changed
    if ($('#setAdvExport').checked !== S.advExportWas) setAdvExport(S.advExportWas);
    setTheme(S.themeWas);
  };
  $('#settingsCancel').addEventListener('click', () => {
    undo();
    d.close();
  });
  d.addEventListener('cancel', undo); // Escape; not on close, which Save fires too
  $('#settingsForm').addEventListener('submit', (e) => {
    e.preventDefault();
    if (!$('#settingsSave').hidden) saveSettings();
  });
  d.addEventListener('close', () => { $('#setAiKeyInput').value = ''; });
  $('#setAiProvider').addEventListener('change', (e) => {
    // model names belong to one provider: back to the saved one, else the default
    const back = e.target.value === e.target.dataset.saved;
    renderAiFields(back ? $('#setAiModel').dataset.saved : '');
    loadAiModels(false);
  });
  $('#setAiModel').addEventListener('change', (e) => {
    $('#setAiModelOtherField').hidden = e.target.value !== AI_OTHER; // the id field only for Other
  });
  $('#setAiModelsRefresh').addEventListener('click', () => loadAiModels(true));
  $('#setAiKeySave').addEventListener('click', () => saveAiKey(false));
  $('#setAiKeyClear').addEventListener('click', () => saveAiKey(true));
  $('#setAiKeyInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault(); // saves the key, not the dialog
      saveAiKey(false);
    }
  });
  $('#setProjectsPick').addEventListener('click', async () => {
    const input = $('#setProjects');
    $('#settingsError').textContent = '';
    try {
      const path = await pickPath('folder', input.value);
      if (path) input.value = path;
      input.focus();
    } catch (e) {
      $('#settingsError').textContent = e.message;
    }
  });
}

// ---------- wiring ----------

function renderAll(seekToStart) {
  renderMoments();
  renderStage(seekToStart);
  renderSequence();
  renderSave();
  renderExport();
  renderOutputs();
  renderUndo();
  renderChat();
}

/**
 * The separators beside the moments column and the sequence column: drag, arrow keys (Home/End for
 * the limits) or double-click to reset. Each width is remembered per browser; without storage it
 * lasts the session. A column never widens past half the window, nor so far that the preview and
 * the other column together get less than STAGE_MIN.
 */
function initSplitters() {
  const ws = $('#workspace');
  const applies = [];
  // Each column's limit depends on the other's width: apply ``first``, then the other (its limit
  // and its CSS default follow), then ``first`` once more, as its limit follows the other's width.
  const refresh = (first) => [first, ...applies.filter((a) => a !== first), first].forEach((a) => a());
  const splitter = ({ bar, col, other, prop, key, min, maxW, dir }) => {
    let pref = null; // dragged width in pixels, or null for the CSS default
    try {
      pref = Number(localStorage.getItem(key)) || null;
    } catch {
      // storage blocked: start from the default
    }
    const store = () => {
      try {
        if (pref == null) localStorage.removeItem(key);
        else localStorage.setItem(key, String(pref));
      } catch {
        // storage blocked or full: the width still applies until reload
      }
    };
    const max = () => Math.max(min, Math.min(maxW, Math.floor(innerWidth / 2),
      ws.clientWidth - other.offsetWidth - STAGE_MIN));
    const clamp = (w, hi = max()) => Math.round(Math.min(Math.max(w, min), hi));
    // pref stays as chosen; only what is applied is clamped, so a wider window gets it back.
    // Each apply reads layout before its own write; refresh() chains three, so a change costs up to
    // two forced relayouts (at most one change per frame while dragging).
    const apply = () => {
      const hi = max();
      const w = pref == null ? null : clamp(pref, hi);
      if (w == null) ws.style.removeProperty(prop);
      else ws.style.setProperty(prop, `${w}px`);
      bar.setAttribute('aria-valuemin', min);
      bar.setAttribute('aria-valuemax', hi);
      bar.setAttribute('aria-valuenow', w ?? col.offsetWidth);
    };
    const set = (w) => {
      pref = w == null ? null : clamp(w);
      refresh(apply);
    };

    bar.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return;
      e.preventDefault();
      bar.setPointerCapture(e.pointerId);
      bar.classList.add('dragging');
      const x0 = e.clientX;
      const w0 = col.offsetWidth;
      // At most one resize per frame, however fast the pointer reports.
      let next = null;
      let frame = 0;
      const flush = () => {
        frame = 0;
        if (next != null) set(next);
        next = null;
      };
      const move = (ev) => {
        next = w0 + dir * (ev.clientX - x0); // dragging away from the column widens it
        if (!frame) frame = requestAnimationFrame(flush);
      };
      const end = () => {
        cancelAnimationFrame(frame);
        flush();
        bar.classList.remove('dragging');
        bar.removeEventListener('pointermove', move);
        bar.removeEventListener('lostpointercapture', end);
        store();
      };
      bar.addEventListener('pointermove', move);
      bar.addEventListener('lostpointercapture', end);
    });
    bar.addEventListener('keydown', (e) => {
      const step = (e.shiftKey ? SPLIT_STEP * 4 : SPLIT_STEP) * dir;
      const w = col.offsetWidth;
      const next = { ArrowLeft: w - step, ArrowRight: w + step, Home: min, End: max() }[e.key];
      if (next == null) return;
      e.preventDefault();
      set(next);
      store();
    });
    bar.addEventListener('dblclick', () => {
      set(null);
      store();
    });
    return apply;
  };
  const side = $('#side');
  const moments = $('.moments');
  // dir: +1 when the column is left of its bar (dragging right widens it), -1 when right of it
  applies.push(
    splitter({ bar: $('#sideSplit'), col: side, other: moments, prop: '--side-w', key: SIDE_KEY,
      min: SIDE_MIN, maxW: SIDE_MAX, dir: -1 }),
    splitter({ bar: $('#momSplit'), col: moments, other: side, prop: '--mom-w', key: MOM_KEY,
      min: MOM_MIN, maxW: MOM_MAX, dir: 1 }),
  );
  addEventListener('resize', () => refresh(applies[0]));
  refresh(applies[0]);
}

/**
 * Mute and volume for every preview video (the stage and the final video), at the right end of the
 * transport bar. The setting is remembered per browser; without storage it lasts the session. The
 * final video's own controls change it too, so each video's volumechange feeds back into the shared
 * setting.
 */
function initVolume() {
  const vids = [...els, $('#finalVideo')];
  const DEFAULT_VOL = { volume: 1, muted: false }; // the browser default when nothing is stored
  let vol = { ...DEFAULT_VOL };
  try {
    const saved = JSON.parse(localStorage.getItem(VOL_KEY));
    const n = Number(saved?.volume);
    if (Number.isFinite(n) && n >= 0 && n <= 1) vol.volume = n;
    if (typeof saved?.muted === 'boolean') vol.muted = saved.muted;
  } catch {
    // storage blocked or unreadable: start from the default
  }
  const store = () => {
    try {
      if (vol.volume === DEFAULT_VOL.volume && vol.muted === DEFAULT_VOL.muted) localStorage.removeItem(VOL_KEY);
      else localStorage.setItem(VOL_KEY, JSON.stringify(vol));
    } catch {
      // storage blocked or full: the setting still applies until reload
    }
  };
  const btn = h('button', { type: 'button', class: 'btn icon-btn', 'aria-label': 'Mute', title: 'Mute', 'aria-keyshortcuts': 'M' });
  const range = h('input', { type: 'range', min: 0, max: 100, step: 1, 'aria-label': 'Volume' });
  const apply = () => {
    for (const el of vids) {
      if (el.volume !== vol.volume) el.volume = vol.volume;
      if (el.muted !== vol.muted) el.muted = vol.muted;
    }
    const pct = Math.round(vol.volume * 100);
    range.value = pct;
    range.setAttribute('aria-valuetext', `${pct}%`);
    btn.setAttribute('aria-pressed', String(vol.muted));
    btn.replaceChildren(icon(vol.muted ? 'muted' : 'volume'));
  };
  const set = (next) => {
    if (next.volume === vol.volume && next.muted === vol.muted) return;
    vol = next;
    apply();
    store();
  };
  // unmuting at zero would stay silent, so it brings the volume back up as native controls do
  const toggleMute = () => set({ volume: vol.muted && vol.volume === 0 ? 1 : vol.volume, muted: !vol.muted });
  btn.addEventListener('click', toggleMute);
  // Space stays play/pause here too: the global keydown handler plays and cancels the keydown, and the
  // button's own Space click (fired on keyup) is cancelled here, so Space never mutes
  btn.addEventListener('keyup', (e) => {
    if (e.key === ' ') e.preventDefault();
  });
  range.addEventListener('input', () => {
    // moving the slider unmutes, as native controls do, unless it is at zero
    const volume = Number(range.value) / 100;
    set({ volume, muted: vol.muted && volume === 0 });
  });
  for (const el of vids) el.addEventListener('volumechange', () => set({ volume: el.volume, muted: el.muted }));
  $('.transport-end').append(h('div', { class: 'vol', role: 'group', 'aria-label': 'Preview sound' }, btn, range));
  apply();
  return toggleMute;
}

/**
 * The sequence column's tabs (Sequence, AI Chat, Export; more can be added as a button and a panel):
 * click, or arrow keys and Home/End on the focused tab, which select as they move (automatic
 * activation). The choice is remembered per browser; without storage it lasts the session. Returns
 * showTab(name, focus).
 */
function initSideTabs() {
  const tabs = [...document.querySelectorAll('#side [role=tab]')];
  const select = (tab, focus = false) => {
    for (const t of tabs) {
      const on = t === tab;
      t.setAttribute('aria-selected', String(on));
      t.tabIndex = on ? 0 : -1;
      const panel = $(`#${t.getAttribute('aria-controls')}`);
      panel.hidden = !on;
      if (!on) for (const vid of panel.querySelectorAll('video')) vid.pause(); // the Export tab's final video: not heard unseen
    }
    if (focus) tab.focus();
    try {
      if (tab === tabs[0]) localStorage.removeItem(TAB_KEY);
      else localStorage.setItem(TAB_KEY, tab.dataset.tab);
    } catch {
      // storage blocked or full: the tab still shows until reload
    }
  };
  let saved = null;
  try {
    saved = localStorage.getItem(TAB_KEY);
  } catch {
    // storage blocked: start on the first tab
  }
  select(tabs.find((t) => t.dataset.tab === saved) ?? tabs[0]);
  for (const t of tabs) t.addEventListener('click', () => select(t));
  $('#side [role=tablist]').addEventListener('keydown', (e) => {
    const i = tabs.indexOf(document.activeElement);
    if (i < 0) return;
    const next = { ArrowLeft: i - 1, ArrowRight: i + 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (next == null) return;
    e.preventDefault();
    select(tabs[(next + tabs.length) % tabs.length], true);
  });
  return (name, focus = false) => {
    const tab = tabs.find((t) => t.dataset.tab === name);
    if (tab && tab.getAttribute('aria-selected') !== 'true') select(tab, focus);
  };
}

/**
 * The moments list's order (#469): Score (rank #1 first, the default) or Position in video, the
 * Sort by radios of Sort and filter. The choice is remembered per browser; without storage it lasts
 * the session.
 */
function initMomentOrder() {
  const radios = [...document.querySelectorAll('#filterPop input[name=momentOrder]')];
  try {
    const v = localStorage.getItem(ORDER_KEY);
    if (Object.hasOwn(ORDER_NAMES, v)) S.order = v;
  } catch {
    // storage blocked: Score
  }
  for (const r of radios) {
    r.checked = r.value === S.order;
    r.addEventListener('change', () => {
      if (!r.checked) return;
      S.order = Object.hasOwn(ORDER_NAMES, r.value) ? r.value : 'score';
      try {
        if (S.order === 'score') localStorage.removeItem(ORDER_KEY);
        else localStorage.setItem(ORDER_KEY, S.order);
      } catch {
        // storage blocked or full: the order still applies until reload
      }
      $('#orderLive').textContent = `Moments sorted by ${ORDER_NAMES[S.order]}`;
      renderMoments();
    });
  }
}

/**
 * Sort and filter: the funnel button beside the search opens the sort order and the quick filters
 * (a disclosure: Escape closes it and returns focus to the button; a click or focus outside closes
 * it). The filters combine with the search text (util.js momentMatches). Sequence and Ready only
 * are remembered per browser like the order; the source file only while its project stays open
 * (resetWorkspace), as paths differ per project. Without storage they last the session.
 */
function initMomentFilters() {
  const btn = $('#filterBtn');
  const pop = $('#filterPop');
  const menu = btn.parentElement;
  btn.prepend(icon('funnel'));
  try {
    const saved = JSON.parse(localStorage.getItem(FILTERS_KEY));
    if (['all', 'in', 'out'].includes(saved?.seq)) S.filters.seq = saved.seq;
    S.filters.ready = saved?.ready === true;
  } catch {
    // storage blocked or not JSON: no filters
  }
  const seqRadios = [...pop.querySelectorAll('input[name=filterSeq]')];
  const showFilters = () => {
    for (const r of seqRadios) r.checked = r.value === S.filters.seq;
    $('#filterReady').checked = S.filters.ready;
  };
  showFilters();
  const open = (on, refocus = false) => {
    pop.hidden = !on;
    btn.setAttribute('aria-expanded', String(on));
    if (refocus) btn.focus();
  };
  btn.addEventListener('click', () => open(pop.hidden));
  menu.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || pop.hidden) return;
    e.preventDefault(); // the page's own Escape (stop Play sequence) waits
    open(false, true);
  });
  document.addEventListener('pointerdown', (e) => !pop.hidden && !menu.contains(e.target) && open(false));
  menu.addEventListener('focusout', (e) => !pop.hidden && e.relatedTarget && !menu.contains(e.relatedTarget) && open(false));
  const changed = () => {
    try {
      const { seq, ready } = S.filters;
      if (!filtersOn({ seq, ready })) localStorage.removeItem(FILTERS_KEY);
      else localStorage.setItem(FILTERS_KEY, JSON.stringify({ seq, ready }));
    } catch {
      // storage blocked or full: the filters still apply until reload
    }
    renderMoments();
    const n = $('#candList').querySelectorAll('.cand').length;
    $('#orderLive').textContent = `${n} of ${cands().length} moments shown`;
  };
  for (const r of seqRadios)
    r.addEventListener('change', () => {
      if (!r.checked) return;
      S.filters.seq = r.value;
      changed();
    });
  $('#filterReady').addEventListener('change', (e) => {
    S.filters.ready = e.target.checked;
    changed();
  });
  $('#filterSource').addEventListener('change', (e) => {
    S.filters.source = e.target.value;
    changed();
  });
  $('#filterClear').addEventListener('click', () => {
    S.filters = { ...NO_FILTERS };
    showFilters();
    changed();
  });
}

/**
 * Bring Sort and filter up to date for the list: the source file choice (only with several
 * ``sources``; ``f`` holds the one in use), and the funnel's dot and name while any filter or a
 * sort other than Score is on. Closed while the list has no moments (``shown`` false).
 */
function syncFilterMenu(sources, f, shown) {
  const sel = $('#filterSource');
  const names = sources.slice().sort((a, b) => naturalCompare(baseName(a), baseName(b)) || naturalCompare(a, b));
  const tag = JSON.stringify(names);
  if (sel.dataset.sources !== tag) {
    sel.dataset.sources = tag;
    sel.replaceChildren(h('option', { value: '', text: 'All files' }), ...names.map((s) => h('option', { value: s, title: s, text: baseName(s) })));
  }
  sel.value = f.source;
  $('#filterSourceWrap').hidden = sources.length < 2;
  const on = [filtersOn(f) && 'filters on', S.order !== 'score' && `sorted by ${ORDER_NAMES[S.order]}`].filter(Boolean);
  const btn = $('#filterBtn');
  btn.classList.toggle('on', !!on.length);
  btn.setAttribute('aria-label', ['Sort and filter moments', ...on].join(', '));
  if (!shown && !$('#filterPop').hidden) {
    $('#filterPop').hidden = true;
    btn.setAttribute('aria-expanded', 'false');
  }
}

/**
 * Export & Assembly Options: each change is remembered per browser (only the options the user
 * changed, so the others follow OPT_DEFAULTS); without storage they last the session. Link
 * Transitions makes the audio transition follow the video one; a crossfade needs a video fade
 * (the CLI refuses it on cuts), so it is unavailable with Cut.
 */
function initExportOptions() {
  let saved = {};
  try {
    const v = JSON.parse(localStorage.getItem(OPTS_KEY));
    if (v && typeof v === 'object') saved = v;
  } catch {
    // storage blocked or unreadable: the defaults
  }
  const store = (key, value) => {
    saved[key] = value;
    try {
      localStorage.setItem(OPTS_KEY, JSON.stringify(saved));
    } catch {
      // storage blocked or full: the choice still applies until reload
    }
  };
  const checks = { vertical: '#optVertical', captions: '#optCaptions', speech_aware: '#optSpeech', smart_tail: '#optTail', loudnorm: '#optLoudnorm', link: '#optLink' };
  const edges = typeof saved.smart_edges === 'boolean' ? saved.smart_edges : undefined; // the old Smart edges box
  for (const [key, id] of Object.entries(checks)) {
    const was = typeof saved[key] === 'boolean' ? saved[key] : key === 'speech_aware' || key === 'smart_tail' ? edges : undefined;
    $(id).checked = was ?? OPT_DEFAULTS[key];
    $(id).addEventListener('change', (e) => {
      store(key, e.target.checked);
      if (key === 'link') syncAudio();
      else syncTiming();
    });
  }
  for (const [key, id] of Object.entries(OPT_FIELDS)) {
    if (typeof saved[key] === 'string') $(id).value = saved[key];
    $(id).addEventListener('change', (e) => {
      store(key, e.target.value);
      syncTiming();
    });
  }
  for (const [key, id] of Object.entries(OPT_SELECTS)) {
    const el = $(id);
    el.value = [...el.options].some((o) => o.value === saved[key]) ? saved[key] : OPT_DEFAULTS[key];
    el.addEventListener('change', () => {
      store(key, el.value);
      syncTiming();
    });
  }
  const events = $('#optEventTails');
  events.value = [...events.options].some((o) => o.value === saved.event_tails) ? saved.event_tails : OPT_DEFAULTS.event_tails;
  const kind = $('#optKind');
  kind.value = [...kind.options].some((o) => o.value === saved.kind) ? saved.kind : OPT_DEFAULTS.kind;
  kind.addEventListener('change', () => store('kind', kind.value));
  events.addEventListener('change', () => {
    store('event_tails', events.value);
    syncTiming();
  });
  $(saved.picks === 'auto' ? '#optPicksAuto' : '#optPicksSeq').checked = true;
  for (const radio of document.querySelectorAll('input[name=optPicks]')) {
    radio.addEventListener('change', () => {
      store('picks', picks());
      syncTiming();
      renderExport();
    });
  }
  syncTiming();
  const video = $('#optTransition');
  const audio = $('#optAudioTransition');
  video.value = saved.transition in LINKED_AUDIO ? saved.transition : OPT_DEFAULTS.transition;
  audio.value = Object.values(LINKED_AUDIO).includes(saved.audio_transition) ? saved.audio_transition : OPT_DEFAULTS.audio_transition;
  video.addEventListener('change', () => {
    store('transition', video.value);
    syncAudio();
  });
  audio.addEventListener('change', () => {
    store('audio_transition', audio.value);
    syncTiming();
  });
  syncAudio();
  const range = $('#optPeaks');
  range.addEventListener('input', () => {
    setPeaks(peaks());
    store('compress_peaks', peaks());
  });
  setPeaks(PEAK_STOPS.includes(saved.compress_peaks) ? saved.compress_peaks : OPT_DEFAULTS.compress_peaks);
  const tint = $('#optColor');
  tint.addEventListener('input', () => {
    setColor(color());
    if (!S.id) return;
    // per project: a grade for one footage (Underwater on a dive) does not carry over to the next
    const mine = saved.color_by_project && typeof saved.color_by_project === 'object' ? saved.color_by_project : {};
    store('color_by_project', { ...mine, [S.id]: color() });
  });
  const strength = $('#optColorStrength');
  strength.addEventListener('input', () => {
    setColorStrength(Number(strength.value));
    store('color_strength', Number(strength.value));
  });
  const pct = saved.color_strength;
  setColorStrength(Number.isFinite(pct) && pct >= 0 && pct <= 100 ? pct : 100);
  setColor('off'); // startColor sets a project's
  return saved;
}

/**
 * Enable the Export tab's fields that apply: the automatic-pick fields with Automatic picks, Top
 * moments without candidate ids (the CLI takes one or the other), the speech caps with Speech-aware
 * cuts, the tail lengths with Smart tails. The sequence's moments keep their exact ends with Smart
 * tails off, which turns event tails off too, so Event tails is not sent then. The fields the CLI
 * would not use are off too: Output size with Vertical 9:16, Fade length under a video Cut,
 * Assembly audio fade with the Crossfade audio transition, Captions model without captions. The
 * crop box shows with Crop Custom.
 */
function syncTiming() {
  const auto = picks() === 'auto';
  const able = (id, on) => {
    const el = $(id);
    el.disabled = !on;
    document.querySelector(`label[for=${el.id}]`)?.classList.toggle('off', !on);
  };
  $('#autoOpts').hidden = !auto;
  able('#optTop', auto && !$('#optIds').value.trim());
  for (const id of ['#optMaxTotal', '#optIds', '#optOrder']) able(id, auto);
  const speech = $('#optSpeech').checked;
  const tail = $('#optTail').checked;
  able('#optSpeechExtend', speech);
  able('#optSpeechLead', speech);
  able('#optTailMin', tail);
  able('#optTailMax', tail);
  able('#optEventTails', auto || tail);
  able('#optEventWindow', (auto || tail) && $('#optEventTails').value !== 'off');
  const pano = S.project?.kind === '360'; // the 360 exporter renders its own frame: no crop or size
  able('#optCrop', !pano);
  const custom = $('#optCrop').value === 'custom';
  // hidden and disabled: a required box Crop does not use must not block Export clips
  $('#optCropBox').hidden = !custom;
  $('#optCropBox').disabled = !custom || pano;
  document.querySelector('label[for=optCropBox]').hidden = !custom;
  able('#optSize', !$('#optVertical').checked && !pano);
  able('#optFadeDur', $('#optTransition').value === 'fade');
  able('#optFinalAudioFade', $('#optAudioTransition').value !== 'crossfade');
  able('#optCaptionModel', $('#optCaptions').checked);
  renderAdvInUse();
}

/** Put the audio transition in line with Link Transitions and the video transition. */
function syncAudio() {
  const video = $('#optTransition').value;
  const audio = $('#optAudioTransition');
  const linked = $('#optLink').checked;
  const fade = audio.querySelector('option[value=crossfade]');
  fade.disabled = !linked && video === 'cut';
  fade.title = fade.disabled ? 'A crossfade needs the Fade video transition.' : '';
  if (linked) audio.value = LINKED_AUDIO[video];
  else if (fade.disabled && audio.value === 'crossfade') audio.value = 'cut';
  audio.disabled = linked;
  syncTiming(); // Fade length and Assembly audio fade follow the transitions
}

/** Set the Compress Audio Peaks slider to ``value`` ('off', 'light' or 'strong'). */
function setPeaks(value) {
  const range = $('#optPeaks');
  range.value = String(Math.max(0, PEAK_STOPS.indexOf(value)));
  range.setAttribute('aria-valuetext', PEAK_NAMES[peaks()]);
}

/** Set the Color slider to ``value`` ('off', 'auto' or 'underwater'); Strength shows only with a grade. */
function setColor(value) {
  const range = $('#optColor');
  range.value = String(Math.max(0, COLOR_STOPS.indexOf(value)));
  range.setAttribute('aria-valuetext', COLOR_NAMES[color()]);
  $('#colorStrengthRow').hidden = color() === 'off';
}

/** Set Color's Strength slider to ``pct`` (0-100). */
function setColorStrength(pct) {
  const range = $('#optColorStrength');
  range.value = String(pct);
  range.setAttribute('aria-valuetext', `Strength ${range.value}%`);
  $('#colorStrengthOut').textContent = `${range.value}%`;
}

/**
 * The Color a project opens with: the choice this browser saved for it (color_by_project), else
 * color in config.json, else Off. Never a grade for gaming footage (config.json's color does not
 * apply to it either); a grade the user then picks is still sent, and the CLI warns. Strength is
 * one choice for every project.
 */
function startColor() {
  const mine = S.savedOpts.color_by_project?.[S.id];
  const value = COLOR_STOPS.includes(mine) ? mine : S.colorDefault;
  setColor(S.project?.gaming ? 'off' : value);
}

function bind() {
  initSplitters();
  const toggleMute = initVolume();
  bindSettings();
  initNerdStats();
  showTab = initSideTabs();
  initMomentOrder();
  initMomentFilters();
  bindAssistant();
  S.savedOpts = initExportOptions();
  initAdvExport();
  showPlaying(false);
  trimInputs.start = trimField('start', 'Clip Start');
  trimInputs.end = trimField('end', 'Clip End');
  $('#trimMsg').after(
    h('button', { type: 'button', class: 'btn small', id: 'resetBtn', hidden: true, onclick: () => {
      const c = current();
      if (!c?.cand) return;
      remember();
      if (c.seg) {
        c.seg.start = c.cand.start;
        c.seg.end = c.cand.end;
        changed();
      } else S.drafts[c.id] = { at: S.drafts[c.id]?.at }; // keep where a removed moment goes back
      setTrimMsg('');
      renderTrim(true);
      seekTo(c.cand.start);
    }, text: 'Reset to analyzed times' }),
  );

  $('#projectSelect').addEventListener('change', (e) => e.target.value && openProject(e.target.value));
  $('#newBtn').addEventListener('click', () => openDialog(S.styles));
  $('#saveBtn').addEventListener('click', () => saveReview().catch(fail));
  $('#filter').addEventListener('input', (e) => {
    S.filter = e.target.value;
    renderMoments();
  });
  $('#toggleBtn').addEventListener('click', () => {
    // The stage stays on this clip either way, so the button flips and a second click undoes.
    const c = current();
    if (!c) return;
    (c.seg ? removeSeg : addSeg)(c.id);
    $('#seqLive').textContent = `${c.label} ${c.seg ? 'removed from' : 'added to'} the sequence`;
  });
  $('#seqPlayBtn').addEventListener('click', () => (S.seqPlay ? seqStop() : seqStart()));
  $('#undoBtn').addEventListener('click', undo);
  $('#redoBtn').addEventListener('click', redo);
  $('#seqRemoveSel').addEventListener('click', removeSelected);
  $('#seqClearSel').addEventListener('click', () => {
    S.selected.clear();
    renderBatch();
    focusRow(0, 'select');
  });
  $('#analyzeBtn').addEventListener('click', () => analyzeAgain());
  // an out-of-range Analyze option blocks Analyze: open the options so the browser can point at it
  $('#againOk').addEventListener('click', () => {
    if (!$('#analyzeDialog form').checkValidity()) $('#analyzeOpts').open = true;
  });
  $('#revealHeadBtn').prepend(icon('folder'));
  $('#revealHeadBtn').addEventListener('click', (e) => revealFolder(e.currentTarget));
  $('#revealExportBtn').prepend(icon('folder'));
  $('#revealExportBtn').addEventListener('click', (e) => revealFolder(e.currentTarget, e.currentTarget.dataset.target));
  $('#clipsBtn').addEventListener('click', () => {
    const body = exportOptions();
    if (validOptions(body)) startJob('clips', body);
  });
  $('#assembleBtn').addEventListener('click', () => {
    if (validOptions(assembleFields())) startJob('assemble', assembleOptions());
  });
  $('#extendBtn').addEventListener('click', () => {
    const body = extendOptions();
    if (validOptions(body)) startJob('extend', body);
  });
  for (const id of ['#optExtendStart', '#optExtendEnd', '#optSpeech']) $(id).addEventListener('input', renderBusy); // Extend export's reason
  $('#optIds').addEventListener('input', syncTiming);
  $('#makeBtn').addEventListener('click', makePreset);
  $('#cancelBtn').addEventListener('click', async () => {
    if (!S.active) return;
    $('#cancelBtn').disabled = true;
    try {
      await api(`/api/jobs/${encodeURIComponent(S.active.id)}/cancel`, { method: 'POST' });
      pollJobs();
    } catch (e) {
      fail(e);
    }
  });
  $('#logToggle').addEventListener('click', () => {
    S.logOpen = !S.logOpen;
    renderJobs();
  });

  // Escape hides the camera path tooltip of a moment card (#517) until focus or the pointer moves on
  let tipsOff = null; // the card whose tooltip Escape hid
  const tipsOn = () => {
    tipsOff = null;
    document.body.classList.remove('tips-off');
  };
  addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    tipsOff = document.querySelector('.cand:hover') ?? document.activeElement?.closest?.('.cand') ?? null;
    if (tipsOff) document.body.classList.add('tips-off');
  }, true);
  addEventListener('focusin', (e) => tipsOff && e.target.closest?.('.cand') !== tipsOff && tipsOn());
  addEventListener('pointerover', (e) => tipsOff && e.target.closest?.('.cand') !== tipsOff && tipsOn());
  addEventListener('keydown', (e) => {
    if (e.defaultPrevented || $('#newDialog').open || $('#confirmDialog').open || $('#analyzeDialog').open || $('#settingsDialog').open || popOpen()) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
      e.preventDefault();
      saveReview().catch(fail);
      return;
    }
    // Ctrl+Z / Ctrl+Y (Shift+Ctrl+Z): the sequence's undo stack, manual and AI edits alike; a text
    // field keeps its own undo
    const k = e.key.toLowerCase();
    if ((e.ctrlKey || e.metaKey) && !e.altKey && (k === 'z' || k === 'y')
      && !e.target.closest('input:not([type=range]):not([type=checkbox]):not([type=radio]),textarea,select,[contenteditable]')) {
      e.preventDefault();
      if (k === 'y' || e.shiftKey) redo();
      else undo();
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    // text fields, the column splitters (their own arrow keys) and any dialog or popover keep their keys
    if (e.target.closest('input:not([type=range]),textarea,select,[role=separator],dialog,[popover]')) return;
    if (e.key === '[' || e.key === ']') {
      e.preventDefault();
      setRange(e.key === '[' ? 'start' : 'end', playhead());
    } else if (e.key === ' ' && (!e.target.closest('button,a,summary') || e.target.closest('.vol'))) {
      e.preventDefault();
      togglePlay();
    } else if (e.key === 'Escape' && S.seqPlay) {
      e.preventDefault();
      seqStop();
    } else if (e.key.toLowerCase() === 'm' && !e.repeat) {
      e.preventDefault();
      toggleMute();
    }
  });
  addEventListener('beforeunload', (e) => {
    if (dirty()) e.preventDefault();
  });
}

async function main() {
  bind();
  try {
    const session = await startSession();
    S.styles = session.styles ?? [];
    S.canReveal = session.can_reveal === true;
    S.canPick = session.can_pick === true;
    openDialog = initNewProject(async (created) => {
      try {
        await loadProjects();
        await openProject(created.id);
      } catch (e) {
        fail(e); // the dialog has closed: report on the banner
      }
    }, S.canPick);
  } catch (e) {
    fail(e);
    $('#newBtn').disabled = true;
    $('#projectSelect').disabled = true;
    $('#settingsBtn').disabled = true;
    setEmpty($('#stageEmpty'), h('p', {}, e.message));
    return;
  }
  try {
    await loadProjects();
  } catch (e) {
    // e.g. a config.json the server rejects (409 names the file): the banner says what to fix,
    // and Settings stays usable.
    fail(e);
  }
  try {
    // A compress_peaks set in config.json is the default, unless this browser saved a choice; the
    // slider keeps Light if it cannot be read
    const st = await api('/api/settings');
    if (!PEAK_STOPS.includes(S.savedOpts.compress_peaks) && st.compress_peaks_set) setPeaks(st.compress_peaks);
    // likewise config.json's color, applied as each project opens (startColor: Off for gaming)
    if (COLOR_STOPS.includes(st.color)) S.colorDefault = st.color;
  } catch {
    // the project list has reported a config.json the server rejects
  }
  const wanted = decodeURIComponent(location.hash.slice(1));
  const first = S.projects.find((p) => p.id === wanted) ?? S.projects[0];
  if (first) await openProject(first.id);
  else renderAll(true);
  pollJobs();
}

main();
