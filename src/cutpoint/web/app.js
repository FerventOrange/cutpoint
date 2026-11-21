// cutpoint editing workspace: ranked moments, trim preview, sequence, export and assemble.
import { api, startSession, whenExpired } from './api.js';
import { initNewProject, pickPath } from './picker.js';
import { $, baseName, colorNote, crashOf, crashText, filtersOn, fmt, fmtBytes, fmtLen, h, KINDS, kindOf, momentMatches, NO_FILTERS, parseTime, positionCompare, readStore, round2, rowInPlace, scoreText, spanText, svg, writeStore } from './util.js';
import { bindAssistant, loadAssistant, markMoments, refreshFrameIndex, renderChat, renderProposal, renderRankUndo, syncRankingBusy, undoRanking } from './chat.js';
import { bindModelsTab } from './models.js';
import { bindPlayer, cancelResume, clearStall, drop, els, ensureVideo, pendingSeek, prepareSource, seek, seekTo, seekWant, setPendingSeek, setSeekWant, setV, setWasIn, showPlaying, spare, togglePlay, unload, updatePlayhead, v } from './player.js';
import { ADV, initKindFilter, initMomentFilters, initMomentOrder, initSideTabs, initSplitters, initTheme, initVolume, KEYS, NERD, previewVolume, renderAdvInUse, setTheme, showAdvExport, syncFilterMenu, theme, VERBOSE } from './prefs.js';
// what chat.js, models.js, prefs.js and player.js use from here
export { addSeg, analyzeAgain, askDirector, banner, blend, BUSY_TEXT, cands, chooseColor, COLOR_NAMES, colorChoice, confirmAction, cued, current, cutPlay, dip, directed, directorBad, drawGhost, drawRail, drawSbHead, durOf, exportedLen, fail, filterLabel, focusMoment, follow, FRAME, holdsClip, icon, inLoaded, keepDraft, key, loaded, offsetOf, ofSource, OPT_FIELDS, OPT_SELECTS, planned, playable, playhead, pollSources, popOpen, precueCut, preserveFocus, readyFor, refreshProject, remember, renderAfterSeqChange, renderMoments, renderNerdStats, renderPanoNote, revealFolder, revealRow, S, sentence, seqCheck, seqClock, seqGo, seqHome, seqNow, seqSeg, seqStop, shownName, showTab, srcOf, tenths };

const MIN_CLIP = 0.1; // seconds
const MARGIN = 10; // seconds: the least rail context around a moment; playback and trims use the whole source
const MIN_VIEW = 2; // seconds: the rail's deepest zoom
const LONG_CLIP = 600; // seconds: longer clips get a note under the cut line
const LABEL_MAX = 200; // characters; ui/projects.py validate_segments rejects longer labels
const THUMB_W = 160; // pixels requested for sequence thumbnails
const FILM_TILE = 78; // pixels: the rail's filmstrip holds about one keyframe still per this width (#261)
const FILM_STEP = 0.25; // seconds: filmstrip stills sit on multiples of this times a power of 2, so pans and 2x zooms reuse them
const FILM_EXACT = 1; // seconds: a finer grid asks for exact stills; keyframe ones would repeat the same keyframe
const FILM_WAIT = 150; // ms the view stays still before new filmstrip stills are asked for
const FILM_FAILS = 3; // failed stills after which a source gets no filmstrip this session
const JOB_NAMES = { analyze: 'Analysis', edit: 'One-click edit', clips: 'Clip export', extend: 'Export extension', assemble: 'Assembly', resume: 'Resume', crashes: 'Crash detection', events: 'Closer look with vision' };
const live = (status) => status === 'running' || status === 'cancelling'; // the job still holds the server
// localStorage: the Export & Assembly Options the user changed, by option ({link: false, ...}); the rest keep their defaults.
// Candidate ids (ids) are never stored: they belong to one project (A284)
const OPTS_KEY = 'cutpoint.exportOptions';
// Export & Assembly Options for a viewer who changed none; Compress peaks starts from config.json's
// compress_peaks when it sets one, else Light.
const OPT_DEFAULTS = {
  vertical: false, captions: false, speech_aware: true, smart_tail: true, loudnorm: true, link: true,
  transition: 'cut', audio_transition: 'cut', compress_peaks: 'light',
  picks: 'sequence', order: '', event_tails: 'auto', kind: 'highlights', crop_mode: '', caption_model: '',
  format: 'mp4',
};
// The Export panel's number and text fields, by the option they send; blank sends nothing (the CLI's default)
const OPT_FIELDS = {
  top: '#optTop', max_total: '#optMaxTotal', ids: '#optIds', speech_max_extend: '#optSpeechExtend',
  speech_max_lead: '#optSpeechLead', tail_min: '#optTailMin', tail_max: '#optTailMax',
  event_window: '#optEventWindow', extend_start: '#optExtendStart', extend_end: '#optExtendEnd',
  crop: '#optCropBox', size: '#optSize', audio_fade: '#optAudioFade', fade_dur: '#optFadeDur',
  final_audio_fade: '#optFinalAudioFade', // sent as Assemble's audio_fade (assembleOptions)
};
const TEXT_FIELDS = new Set(['crop', 'size']); // OPT_FIELDS sent as typed, not as numbers
// The Export panel's selects that keep their choice, by their storage key; '' (Default) sends nothing
const OPT_SELECTS = { order: '#optOrder', crop_mode: '#optCrop', caption_model: '#optCaptionModel', format: '#optFormat' };
const PEAK_STOPS = ['off', 'light', 'strong']; // the Compress Audio Peaks slider's stops, left to right
const PEAK_NAMES = { off: 'None', light: 'Light', strong: 'Max' };
const COLOR_STOPS = ['off', 'auto', 'underwater']; // the Color slider's stops, left to right (--color)
const COLOR_NAMES = { off: 'Off', auto: 'Auto', underwater: 'Underwater' };
const LINKED_AUDIO = { cut: 'cut', fade: 'crossfade' }; // Link Transitions: the audio transition of each video one
const FRAME = 0.04; // seconds: "Play sequence" moves on this close to a clip's end
const GAP = 0.05; // seconds: a clip starting this close to the previous one's end plays on without a cut
const PRECUE_WAIT = 500; // ms a Fade cut rests in Cut mode, its stage player idle, before the spare player opens clip 2 (precueCut)
const CUT_PLAY = 3; // seconds: "Play across the cut" plays this much of each clip; the roll track shows as much on each side
const FADE_DUR = 0.25; // seconds: Fade length left blank (assemble --fade-dur's default)
const JOINS_NOTE = 'Assemble again to apply the joins.';
const FADE_ROOM = 0.45; // a Fade takes at most this share of the shortest clip it joins (assemble.crossfade_duration)
const JOIN_NAMES = { cut: 'Cut', fade: 'Fade' };
const JOIN_NEXT = { default: 'cut', cut: 'fade', fade: null }; // a join's button steps Default, Cut, Fade; null: Default
const CUT_STILL_W = 640; // pixels requested for Cut mode's two frames (the most the server cuts)
const CUT_NEEDS = 'Cut mode needs a sequence of at least two clips.';
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
  savedKey: '{"segments":[],"joins":[]}',
  drafts: {}, // trims of moments not (yet) in the sequence; whole segments for removed non-moments; ``at``: index a removed clip had
  focus: null,
  filter: '',
  order: 'score', // the moments list's order: 'score' (rank #1 first) or 'position' (initMomentOrder)
  filters: { ...NO_FILTERS }, // the moments list's quick filters (initMomentFilters); util.js momentMatches applies them
  kind: '', // the kind filter above the grid (initKindFilter): '' for all, else a key of util.js KINDS; not kept
  jobs: [],
  active: null, // the running job, if any
  tracked: new Map(), // job id -> last seen status
  logOpen: false,
  saving: false,
  selected: new Set(), // sequence ids ticked for batch actions
  nerdStats: false, // Settings > Stats for Nerds: the scoring overlay on the preview
  nerdStatsWas: false, // nerdStats when Settings opened, restored by Cancel or Escape
  target: null, // the sequence's target length in seconds (seqTarget), or null
  advExport: false, // Settings > Show advanced export options: the Export panel's .adv controls
  advExportWas: false, // advExport when Settings opened, restored by Cancel or Escape
  verboseMoments: false, // Settings > Verbose moment labels: each tile's file, id and start-end
  verboseMomentsWas: false, // verboseMoments when Settings opened, restored by Cancel or Escape
  singleKeys: true, // Settings > Single-key shortcuts: M, [, ], I and O act (off: only modifier shortcuts and keys inside controls)
  singleKeysWas: true, // singleKeys when Settings opened, restored by Cancel or Escape
  themeWas: 'graphite', // Settings > Theme when Settings opened, restored by Cancel or Escape
  renaming: null, // sequence id whose label is being edited
  renameDraft: '',
  renameErr: '',
  expired: false, // the server rejected the session token: nothing polls or retries any more
  stepStates: null, // {job, states: {step: state}}: the analysis steps last announced (renderSteps)
  stepSince: {}, // "job/step" -> ms the UI first saw the step running (elapsed time under reduced motion)
  stepTimer: null, // ticks that elapsed time once a second
  view: null, // {id, lo, hi}: the part of the source the trim rail shows, separate from the trim
  // {order: [ids], i, wait, across}: "Play sequence" runs; order is fixed at its start, wait: the clip's video is not ready.
  // across: {back}: it plays over one cut instead ("Play across the cut"); back: the clip selected before it
  seqPlay: null,
  cutMode: false, // the stage shows the cut after the selected clip (setCut); "Play sequence" ends it
  cutAt: 0, // the cut last shown: its first clip's index, where Cut mode stays when that clip leaves the sequence
  // the joins with a transition of their own, [{a, b, transition}]: clip a's end into clip b's start. A choice
  // stays while its clips are apart (a reorder, a removal) and counts again once b follows a; no entry is Default
  joins: [],
  undo: [], // {segments, drafts, joins} before each edit, manual or applied from AI Chat, newest last (at most UNDO_MAX)
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
  dismissed: new Set(), // ids of proposals discarded from their chat card (and of ranking cards dismissed)
  // ranking cards (#902), by card id (`${turn}-${n}`): {payload} a newer preview from Preview again,
  // {previewing} while it is asked, {stale} the base the apply route called stale
  rankings: new Map(),
  rankApplying: null, // the card id whose ranking the apply route is saving
  rankUndoing: false, // the undo route is swapping the ranking (from a card or the Moments link)
  shown: null, // Set of the moment ids the assistant showed (show_moments), highlighted in the list
  history: [], // the applied assistant changes (GET history), oldest first
};

let openDialog;
let showTab = () => {}; // showTab('sequence', focus): select a tab of the sequence column (initSideTabs)
let cued = null; // {id, start, vt}: the clip the spare player holds, at video time vt
let blend = null; // {old, end, lap}: a Fade playing across a cut: the outgoing player, its out point in video time and the fade's seconds
let dip = null; // {lap}: a Fade across a cut the spare player was not ready for, played as a dip to black in the visible player

// ---------- state helpers ----------

// exact: a word splice of AI Chat, exported at exactly its edges and joined with hard cuts
const pick = (s) => ({ id: s.id, source: s.source, start: s.start, end: s.end, label: s.label, ...(s.exact ? { exact: true } : {}) });
const key = () => JSON.stringify(S.segments.map(pick));
const joinKey = (j) => `${j.a}\n${j.b}`;
/** What Save sends, as a string: the sequence and the joins, in a fixed order, so that undoing a choice is no change. */
const reviewKey = () => JSON.stringify({
  segments: S.segments.map(pick),
  joins: S.joins.map(({ a, b, transition }) => ({ a, b, transition })).sort((x, y) => (joinKey(x) < joinKey(y) ? -1 : 1)),
});

// Export clips cuts a clip where a sentence or reaction ends (Smart tails, Speech-aware cuts), not at
// the trim handle. The server plans that for the sequence (POST /edges); lengths, totals and
// "Play sequence" use it, so they are those of the export. Until it answers, and for a row trimmed
// since, the row's own range counts.
const EDGES_DELAY_MS = 300;
const exportPlan = { pid: null, map: {}, asked: '', timer: 0 };
/** The range a row is exported at: ``{start, end}``. */
function planned(s) {
  const e = exportPlan.pid === S.id ? exportPlan.map[s.id] : null;
  return e && e.from[0] === s.start && e.from[1] === s.end ? e : s;
}
const exportedLen = (s) => { const e = planned(s); return e.end - e.start; };

/** Ask for the planned ranges of the working sequence, a moment after the last change. */
function askEdges() {
  clearTimeout(exportPlan.timer);
  exportPlan.timer = setTimeout(async () => {
    // read when the timer fires: an input event comes before the change event that enables or
    // disables the dependent fields (syncTiming)
    const pid = S.id;
    // every export setting that moves a clip's ends, as Export clips sends it (a field the browser
    // rejects is left out: the route would answer 400)
    const settings = fields('speech_max_extend', 'speech_max_lead', 'tail_min', 'tail_max', 'event_window', 'extend_start', 'extend_end');
    for (const key of Object.keys(settings)) if (!$(OPT_FIELDS[key]).checkValidity()) delete settings[key];
    const body = {
      segments: S.segments.map(pick), smart_tail: $('#optTail').checked, speech_aware: $('#optSpeech').checked, ...settings,
      ...($('#optEventTails').disabled ? {} : { event_tails: $('#optEventTails').value }),
    };
    const asked = `${pid}|${JSON.stringify(body)}`;
    if (!pid || !body.segments.length || exportPlan.asked === asked) return;
    exportPlan.asked = asked;
    try {
      const r = await api(`/api/projects/${encodeURIComponent(pid)}/edges`, { method: 'POST', body });
      if (exportPlan.asked !== asked || S.id !== pid) return; // a newer question is out, or another project opened
      const from = new Map(body.segments.map((s) => [s.id, [s.start, s.end]]));
      exportPlan.pid = pid;
      exportPlan.map = {};
      for (const [id, e] of Object.entries(r.edges ?? {})) if (from.has(id)) exportPlan.map[id] = { start: e.start, end: e.end, reason: e.reason, from: from.get(id) };
      renderSequence();
      // the rail and the length text show the export too; a trim error or Cut mode's reason just shown stays
      const m = $('#trimMsg');
      const cutNote = m.textContent === CUT_NEEDS && !m.classList.contains('vh');
      if (m.classList.contains('error')) drawRail();
      else renderTrim();
      if (cutNote) setTrimMsg(CUT_NEEDS, false, true);
    } catch {
      exportPlan.asked = ''; // the raw ranges stay; the next change asks again
    }
  }, EDGES_DELAY_MS);
}
const dirty = () => reviewKey() !== S.savedKey;
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

/** "7:13.6": a time to the tenth, rounded before it is split, so 59.96 is "1:00.0" and not "0:60.0". */
function tenths(t) {
  const n = Math.round(Math.abs(t) * 10);
  const h = Math.floor(n / 36000);
  const m = Math.floor(n / 600) % 60;
  const sec = ((n % 600) / 10).toFixed(1).padStart(4, '0');
  return `${t < 0 && n ? '-' : ''}${h ? `${h}:${String(m).padStart(2, '0')}` : m}:${sec}`;
}

function validate(s, e, dur) {
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
const sentence = (text) => (/[.!?]$/.test(text) ? text : `${text}.`); // one final period, never two

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
  const row = a?.closest?.('[data-id]');
  const id = row?.dataset.id;
  const act = a?.dataset?.act;
  const listId = a?.closest?.('ol')?.id;
  const at = row ? [...row.parentElement.children].indexOf(row) : -1;
  fn();
  if (!id || !act || !listId) return;
  // a row that left the list (Add with Not in sequence, say) hands focus to the row now in its place,
  // not to the list's first: focusing that scrolled the column back to the top
  const rows = $(`#${listId}`).children;
  const li = $(`#${listId} [data-id="${CSS.escape(id)}"]`) ?? rows[rowInPlace(at, rows.length)];
  const btn = li?.querySelector(`[data-act="${act}"]`);
  (btn && !btn.disabled ? btn : (li?.querySelector('button:not(:disabled)') ?? $(`#${listId} button:not(:disabled)`)))?.focus();
}

const setEmpty = (el, ...kids) => {
  kids = kids.filter((k) => k != null && k !== false); // replaceChildren would print "null"
  el.replaceChildren(...kids);
  el.hidden = !kids.length;
};

// ---------- projects ----------

async function loadProjects() {
  try {
    S.projects = (await api('/api/projects')).projects;
  } catch (e) {
    // e.g. an invalid config.json (409): creating a project would fail the same way until a load
    // succeeds. Settings stays usable to fix it.
    S.projects = [];
    S.projectsFailed = true;
    renderProjectMenu();
    throw e;
  }
  S.projectsFailed = false;
  renderProjectMenu();
}

/**
 * The header's "cutpoint / <project>": the trigger's name, the menu's list (the open project marked,
 * a divider, New project...), then for an analyzed project Analyze again (waits while a job runs),
 * Show folder (where the server can open folders) and a line of facts (count, style, analysis time),
 * and the tab title. Called whenever the list or the open project changes, and by ``renderBusy``
 * when a job starts or ends; it rebuilds the menu only when what it shows changed.
 */
function renderProjectMenu() {
  renderStyleMenu();
  const name = S.projects.find((p) => p.id === S.id)?.name ?? S.project?.name;
  const shown = S.id && name;
  const label = S.projectsFailed ? 'Projects unavailable' : shown ? name : S.projects.length ? 'Choose project' : 'No projects yet';
  $('#projectName').textContent = label;
  $('#projectBtn').title = label;
  $('#projectBtn').disabled = S.projectsFailed || S.sessionFailed === true;
  document.title = shown ? `cutpoint / ${name}` : 'cutpoint';
  const pop = $('#projectMenu');
  const had = pop.contains(document.activeElement) ? document.activeElement : null; // an open menu keeps its focus
  // the open project's actions and facts; Analyze again waits while a job runs or another run holds the project (aria-disabled keeps it focusable)
  const hl = S.id && S.project?.id === S.id ? S.project.highlights : null;
  const held = !!hl && heldElsewhere(S.project); // another run (the command line, another server) holds the project
  const busy = !!S.active || held;
  const why = S.active ? 'Another job is running.' : BUSY_TEXT;
  const facts = hl ? projectFacts(S.project) : [];
  const sig = JSON.stringify([label, S.id, S.projects.map((p) => [p.id, p.name, !!p.interrupted]), !!hl, busy, why, S.canReveal, facts]);
  if (pop.dataset.sig === sig) return; // nothing to change: leave the menu (and a click under way) alone
  pop.dataset.sig = sig;
  pop.replaceChildren(
    ...S.projects.map((p) => h('button', {
      type: 'button', role: 'menuitemradio', class: 'proj-item', tabindex: '-1', 'aria-checked': String(p.id === S.id), 'data-id': p.id,
      onclick: () => pickProject(p.id),
    }, icon('check'), h('span', { text: p.interrupted ? `${p.name} (interrupted)` : p.name }))),
    ...(S.projects.length ? [h('div', { class: 'proj-sep', role: 'separator' })] : []),
    h('button', { type: 'button', role: 'menuitem', class: 'proj-item', id: 'newProjectItem', tabindex: '-1', onclick: pickNewProject },
      h('span', { text: 'New project…' })),
    ...(hl ? [h('div', { class: 'proj-sep', role: 'separator' }),
      h('button', {
        type: 'button', role: 'menuitem', class: 'proj-item', id: 'analyzeAgainItem', tabindex: '-1', 'aria-haspopup': 'dialog',
        'aria-disabled': busy ? 'true' : null, 'aria-describedby': busy ? 'projBusy projStats' : 'projStats', title: busy ? why : null,
        onclick: pickAnalyzeAgain,
      }, h('span', { text: 'Analyze again' })),
      ...(S.canReveal ? [h('button', { type: 'button', role: 'menuitem', class: 'proj-item', id: 'revealItem', tabindex: '-1', onclick: pickReveal },
        h('span', { text: 'Show folder' }))] : []),
      ...(busy ? [h('div', { class: 'vh', id: 'projBusy', role: 'none', text: why })] : []), // the title no longer describes the item
      h('div', { class: 'proj-stats', id: 'projStats', role: 'none' }, ...facts.map((f) => h('span', { text: f })))] : []),
  );
  if (had) {
    const back = had.dataset.id ? pop.querySelector(`[data-id="${CSS.escape(had.dataset.id)}"]`) : had.id && pop.querySelector(`#${CSS.escape(had.id)}`);
    (back ?? $('#newProjectItem')).focus();
  }
}

/** The lines under the project menu's actions: the moment count and style, and the analysis' length and time when known. */
function projectFacts(p) {
  const hl = p.highlights;
  const stats = p.analysis; // null when the length or the time is unknown
  return [
    `${cands().length} found, ${hl.style ?? p.style} style`,
    ...(stats ? [`Analyzed ${spanText(stats.video_s)} of video in ${spanText(stats.took_s)}`] : []),
  ];
}

// ---------- the style button (#897) ----------

/** The name of ``style`` in the style menu: its category's (Talk for commentary), else the style's own. */
const styleName = (d, style) => d.choices.find((c) => c.style === style)?.label ?? style;

/** The style button's text for the project's ``detected`` view. */
function styleChipText(d) {
  if (!d.auto) return `Style: ${styleName(d, d.chosen)} (your choice)`;
  if (d.category && !d.stale) return d.ambiguous ? `Not sure: ${d.fallback_label} style` : `Detected: ${d.label}`;
  if (d.category) return `Detected earlier: ${d.label}`;
  return 'Auto: analyze again to detect';
}

/** The text lines of the style menu: why (the reason as recorded), the sources that disagree, an earlier detection, a suggestion. */
function styleLines(d) {
  if (!d.category) return [];
  const why = d.reason ? ` (${d.reason})` : '';
  const lines = [];
  if (d.stale) lines.push(d.ambiguous ? `Detected earlier: not sure, leans ${d.label}${why}` : `Detected earlier: ${d.label}${why}`);
  else if (d.ambiguous) lines.push(`Leans ${d.label}${why}; used the ${d.fallback_label} style${d.used_style ? ` (${d.used_style})` : ''}`);
  else lines.push(`Detected: ${d.label}${why}`);
  for (const s of d.sources) lines.push(`Disagree: ${s.name} looks like ${s.label ?? s.category}${s.reason ? ` (${s.reason})` : ''}`);
  if (d.suggest && d.suggest !== d.chosen) lines.push(`Suggested: ${styleName(d, d.suggest)} (${d.suggest})`);
  return lines;
}

/**
 * The header's style button: hidden before the first analysis and for 360 projects (``detected`` null).
 * Its menu: the lines of ``styleLines``, then radio items Auto, the categories with their style (Gameplay, Action, Action cam, Talk; General is only a detection label)
 * (Action cam only with camera telemetry or when chosen, else under Other styles) and Other styles
 * (the rest of the session's styles). Choosing one asks first in the Analyze dialog; while a job runs
 * or another run holds the project the items are aria-disabled (still focusable), as Analyze again.
 */
function renderStyleMenu() {
  const wrap = $('#styleMenuWrap');
  const p = S.id && S.project?.id === S.id ? S.project : null;
  const d = p?.detected ?? null;
  wrap.hidden = !d;
  const pop = $('#styleMenu');
  if (!d) {
    if (!pop.hidden) closeStyleMenu();
    pop.dataset.sig = '';
    return;
  }
  const text = styleChipText(d);
  $('#styleName').textContent = text;
  $('#styleBtn').title = text;
  const busy = !!S.active || heldElsewhere(p);
  const why = S.active ? 'Another job is running.' : BUSY_TEXT;
  const main = d.choices.filter((c) => c.main);
  const others = S.styles.filter((s) => s !== 'auto' && !main.some((c) => c.style === s));
  const lines = styleLines(d);
  const sig = JSON.stringify([p.id, d, others, busy, why]);
  if (pop.dataset.sig === sig) return;
  pop.dataset.sig = sig;
  const had = pop.contains(document.activeElement) ? document.activeElement.dataset.style : null;
  const item = (style, label, name = label) => h('button', {
    type: 'button', role: 'menuitemradio', class: 'proj-item style-item', tabindex: '-1', 'data-style': style,
    'aria-checked': String(style === d.chosen), 'aria-disabled': busy ? 'true' : null,
    'aria-describedby': busy ? 'styleBusy' : null, title: busy ? why : null,
    onclick: () => pickStyle(style, name),
  }, icon('check'), h('span', { text: label }));
  pop.replaceChildren(
    ...(lines.length ? [h('div', { class: 'proj-stats', id: 'styleLines', role: 'none' }, ...lines.map((l) => h('span', { text: l }))),
      h('div', { class: 'proj-sep', role: 'separator' })] : []),
    item('auto', 'Auto (detect from footage)', 'Auto'),
    ...main.map((c) => item(c.style, `${c.label}: ${c.style}`, c.label)),
    ...(others.length ? [h('div', { class: 'proj-sep', role: 'separator' }),
      h('div', { class: 'style-group-label', id: 'styleOtherLabel', role: 'none', text: 'Other styles' }),
      h('div', { role: 'group', 'aria-labelledby': 'styleOtherLabel', class: 'style-group' },
        ...others.map((s) => item(s, styleName(d, s))))] : []),
    ...(busy ? [h('div', { class: 'vh', id: 'styleBusy', role: 'none', text: why })] : []),
  );
  // the lines sit outside the menu's items, so the menu itself carries them to a screen reader
  if (lines.length) pop.setAttribute('aria-describedby', 'styleLines'); else pop.removeAttribute('aria-describedby');
  if (had != null) (pop.querySelector(`[data-style="${CSS.escape(had)}"]`) ?? pop.querySelector('.style-item'))?.focus();
}

/** Close the style menu; ``refocus`` returns focus to its button. */
function closeStyleMenu(refocus = false) {
  $('#styleMenu').hidden = true;
  $('#styleBtn').setAttribute('aria-expanded', 'false');
  if (refocus) $('#styleBtn').focus();
}

/** A style item (``name``: what the dialog calls it): the current one only closes the menu; another asks in the Analyze dialog, then analyzes in it. */
function pickStyle(style, name) {
  const item = $('#styleMenu').querySelector(`[data-style="${CSS.escape(style)}"]`);
  if (item?.getAttribute('aria-disabled') === 'true') return; // a job runs: the menu stays open
  closeStyleMenu(true);
  if (style === S.project?.detected?.chosen) return;
  analyzeAgain(false, { style, name });
}

/** The style menu: a menu button that works like the project menu (arrows, Home, End, Escape, Tab, a click outside). */
function initStyleMenu() {
  const btn = $('#styleBtn');
  const pop = $('#styleMenu');
  const wrap = $('#styleMenuWrap');
  const items = () => [...pop.querySelectorAll('.style-item')];
  const show = (last = false) => {
    pop.hidden = false;
    btn.setAttribute('aria-expanded', 'true');
    (last ? items().at(-1) : pop.querySelector('[aria-checked=true]') ?? items()[0])?.focus();
  };
  btn.addEventListener('click', () => (pop.hidden ? show() : closeStyleMenu()));
  btn.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    e.preventDefault();
    if (pop.hidden) show(e.key === 'ArrowUp');
  });
  pop.addEventListener('mousedown', (e) => !e.target.closest('.style-item') && e.preventDefault());
  pop.addEventListener('keydown', (e) => {
    const all = items();
    const at = all.indexOf(document.activeElement);
    const to = { ArrowDown: (at + 1) % all.length, ArrowUp: (at - 1 + all.length) % all.length, Home: 0, End: all.length - 1 }[e.key];
    if (to != null) {
      e.preventDefault();
      all[to].focus();
    } else if (e.key === 'Escape') {
      e.preventDefault();
      closeStyleMenu(true);
    } else if (e.key === 'Tab') closeStyleMenu();
  });
  document.addEventListener('pointerdown', (e) => !pop.hidden && !wrap.contains(e.target) && closeStyleMenu());
}

/** Close the project menu; ``refocus`` returns focus to the trigger. */
function closeProjectMenu(refocus = false) {
  $('#projectMenu').hidden = true;
  $('#projectBtn').setAttribute('aria-expanded', 'false');
  if (refocus) $('#projectBtn').focus();
}

function pickProject(id) {
  closeProjectMenu(true);
  if (id !== S.id) openProject(id);
}

function pickAnalyzeAgain() {
  if ($('#analyzeAgainItem').getAttribute('aria-disabled') === 'true') return; // a job runs or another run holds the project: the menu stays open
  closeProjectMenu(true); // focus is on the name first, so the dialog returns it there
  analyzeAgain();
}

function pickReveal() {
  closeProjectMenu(true);
  revealFolder($('#projectBtn'));
}

function pickNewProject() {
  closeProjectMenu(true);
  openDialog(S.styles);
}

/**
 * The project menu (a menu button): a click, or the arrow keys on the name, open it with focus on the
 * open project; the arrow keys, Home and End only move focus (nothing happens until Enter, Space or a
 * click on an item); Escape and choosing an item close it and return focus to the name; Tab and a
 * click outside only close it.
 */
function initProjectMenu() {
  const btn = $('#projectBtn');
  const pop = $('#projectMenu');
  const wrap = $('#projectMenuWrap');
  const items = () => [...pop.querySelectorAll('.proj-item')];
  const show = (last = false) => {
    pop.hidden = false;
    btn.setAttribute('aria-expanded', 'true');
    (last ? items().at(-1) : pop.querySelector('[aria-checked=true]') ?? items()[0]).focus();
  };
  btn.addEventListener('click', () => (pop.hidden ? show() : closeProjectMenu()));
  btn.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    e.preventDefault();
    if (pop.hidden) show(e.key === 'ArrowUp');
  });
  // a press on the facts or a divider must not take focus out of the menu (the keys would stop working)
  pop.addEventListener('mousedown', (e) => !e.target.closest('.proj-item') && e.preventDefault());
  pop.addEventListener('keydown', (e) => {
    const all = items();
    const at = all.indexOf(document.activeElement);
    const to = { ArrowDown: (at + 1) % all.length, ArrowUp: (at - 1 + all.length) % all.length, Home: 0, End: all.length - 1 }[e.key];
    if (to != null) {
      e.preventDefault();
      all[to].focus();
    } else if (e.key === 'Escape') {
      e.preventDefault(); // the page's own Escape (stop Play sequence) waits
      closeProjectMenu(true);
    } else if (e.key === 'Tab') closeProjectMenu();
  });
  document.addEventListener('pointerdown', (e) => !pop.hidden && !wrap.contains(e.target) && closeProjectMenu());
}

async function openProject(id) {
  if (id !== S.id && dirty() && !confirm('Discard the unsaved changes to this sequence?')) return;
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

/**
 * The proxy polls (A754): only the sources, not every moment and its transcript. A source whose
 * status changed (a proxy finished or failed) or a changed list of sources refreshes the whole
 * project once; otherwise the page takes the new sources as a whole refresh would, and nothing
 * else: the page's review (saved since the project was read) stays. Other changes made outside
 * the page meanwhile (busy or run from a CLI run, a highlights.json rewritten) arrive through the
 * job poll or the next whole refresh, not this poll.
 */
async function pollSources() {
  const id = S.id;
  if (!id) return;
  try {
    const { sources } = await api(`/api/projects/${encodeURIComponent(id)}/sources`);
    if (id !== S.id || S.project?.id !== id) return;
    const old = S.project.sources ?? [];
    const same = sources.length === old.length
      && sources.every((s, i) => s.path === old[i].path && s.status === old[i].status);
    if (!same) return refreshProject();
    adopt({ ...S.project, sources }, false, true);
  } catch (e) {
    fail(e);
  }
}

/** Forget the per-project edit state; ``segments`` and ``joins`` become the saved sequence. */
function resetWorkspace(segments, joins = []) {
  seqStop('', false);
  if (S.cutMode) setCut(false);
  S.drafts = {};
  S.focus = null;
  S.selected.clear();
  S.renaming = null;
  S.segments = segments;
  S.joins = joins;
  S.savedKey = reviewKey();
  S.undo = [];
  S.redo = [];
  S.undoTag = null;
  S.proposal = null;
  S.pendingProposal = null;
  S.shown = null;
  S.filters.source = ''; // a path of the project left
  S.kind = '';
  $('#optIds').value = ''; // candidate ids belong to the project left (A284)
}

/** Leave the open project (its folder is no longer listed): back to the empty workspace. */
function closeProject() {
  S.id = null;
  S.project = null;
  railEvents.id = null;
  envelopes.clear();
  resetWorkspace([]);
  clearTimeout(S.prepTimer);
  drop('');
  history.replaceState(null, '', location.pathname + location.search);
  renderProjectMenu();
  renderAll(true);
  loadAssistant();
}

/** Show project ``p``; ``sourcesOnly``: only its ``sources`` are new (pollSources), the rest is the
 * page's own, so the saved sequence is not taken from it again. */
function adopt(p, switched, sourcesOnly = false) {
  const sig = S.project?.events_sig;
  S.project = p;
  if (switched) {
    S.id = p.id;
    delete $('#momentsEmpty').dataset.refocus;
    resetWorkspace(structuredClone(p.review?.segments ?? []), structuredClone(p.review?.joins ?? []));
    renderProjectMenu();
    history.replaceState(null, '', `#${encodeURIComponent(p.id)}`);
  } else if (!sourcesOnly && !dirty()) {
    // Polling must never clobber unsaved edits; saved ones follow the server.
    S.segments = structuredClone(p.review?.segments ?? []);
    S.joins = structuredClone(p.review?.joins ?? []);
    S.savedKey = reviewKey();
  }
  if (!current()) S.focus = S.segments[0]?.id ?? cands()[0]?.id ?? null;
  clearTimeout(S.prepTimer);
  if (switched) startColor();
  if (switched) startTarget();
  if (switched) envelopes.clear();
  loadRailEvents(p);
  if (switched) syncTiming(); // Crop and Output size are off for 360 projects
  renderAll(switched);
  if (switched) loadAssistant();
  else if (sig !== p.events_sig) refreshFrameIndex(); // an analysis finished: its frame index
}

// ---------- rendering: moments ----------

// What the Moments tiles on the page were drawn from (tilesKey): a project poll (every 1-2 s while
// a proxy builds or a 360 preview renders) or a render that changes none of it keeps the tiles, as
// a gaming project can list thousands (A753).
let tilesDrawn = { list: null, rest: null };

function renderMoments() {
  const p = S.project;
  const hl = p?.highlights;
  const list = cands();
  const resume = $('#resumeBtn'); // a stopped run of progress.json (#109); before analysis the empty state has its own
  resume.hidden = !hl || !p.run?.resumable;
  if (!resume.hidden) resume.dataset.label = resumeLabel(p.run);
  renderRankUndo(); // the last ranking change's Undo/Redo, above the list
  $('.moments').classList.toggle('bare', !hl);
  $('#filterWrap').hidden = !list.length;
  $('#kindSeg').hidden = !list.length;
  // a kind shows only when a moment of this project has it (All always); an emptied one falls back to All
  const present = new Set(list.map(kindOf));
  if (S.kind && !present.has(S.kind)) S.kind = '';
  for (const b of $('#kindSeg').querySelectorAll('button')) {
    const gone = !!b.dataset.kind && !present.has(b.dataset.kind);
    if (gone && b === document.activeElement) $('#kindSeg [data-kind=""]').focus();
    b.hidden = gone;
    b.setAttribute('aria-pressed', String(b.dataset.kind === S.kind));
  }
  $('#momentsCount').textContent = hl ? String(list.length) : ''; // the project's total, whatever the filter shows
  const empty = $('#momentsEmpty');
  // the big button had focus (disabling it while analysis runs moves focus to body): keep it in this column
  const hadFocus = empty.contains(document.activeElement) || (!!empty.dataset.refocus && document.activeElement === document.body);
  empty.classList.toggle('fresh', !!p && !hl);
  if (!p) setEmpty(empty);
  else if (!hl) renderUnanalyzed(empty, p.sources?.length ?? 0, p.source_error ? sourceErrorText(p) : '', p.kind === '360', p.run, !p.source && !p.sources?.length);
  else if (!list.length)
    setEmpty(empty, h('p', {}, 'Analysis found no moments. Try another style, or add footage, then choose Analyze again in the project menu.'));
  else setEmpty(empty);
  if (!p || hl) {
    delete empty.dataset.n;
    delete empty.dataset.refocus;
  }

  const sources = [...new Set(list.map((c) => c.source))];
  const multi = sources.length > 1;
  // Sort and filter: the source file choice only with several sources, and only one of this project's
  const f = { ...S.filters, source: multi && sources.includes(S.filters.source) ? S.filters.source : '' };
  syncFilterMenu(sources, f, !!list.length, list.some((c) => c.long_bit));
  const segOf = (c) => S.segments.find((s) => s.id === c.id);
  const items = list
    .map((c, i) => ({ c, rank: i + 1 }))
    .filter(({ c }) => {
      const seg = segOf(c);
      // the filter finds a moment by the label it shows, its analysis label and vision label
      const label = filterLabel(c);
      return momentMatches({ ...c, label, picked: !!seg, ready: readyFor(srcOf(c.source), seg ?? { ...c, ...S.drafts[c.id] }) }, S.filter, f)
        && (!S.kind || kindOf(c) === S.kind);
    });
  if (S.order === 'position') items.sort((a, b) => positionCompare(a.c, b.c)); // the badge keeps the score rank
  const listHadFocus = $('#candList').contains(document.activeElement);
  const none = list.length && !items.length ? `No moments match ${filtersOn(f) || S.kind ? 'these filters' : 'this filter'}.` : '';
  const drawn = tilesKey(list, items, sources, multi, none);
  // A tile: the still with the kind dot and length (Add/Remove sits over its corner: no nested
  // buttons), then rank and title, kind and score with the preview readiness as an icon, the id and
  // times, the file with several sources, and the strip. The transcript is the tile's tooltip.
  const redraw = drawn.list !== tilesDrawn.list || drawn.rest !== tilesDrawn.rest;
  tilesDrawn = drawn;
  if (redraw && lazySeen) for (const b of $('#candList').querySelectorAll('.cand-thumb.loading')) { lazySeen.unobserve(b); lazyAsks.delete(b); } // tiles about to be replaced
  if (redraw) preserveFocus(() => {
    $('#candList').replaceChildren(
      ...items.map(({ c, rank }) => {
        const seg = segOf(c);
        const picked = !!seg;
        const badge = prepBadge(srcOf(c.source), seg ?? { ...c, ...S.drafts[c.id] }, true);
        const tip = c.pano?.flags?.length ? `tip-${c.id}` : null;
        const title = c.display_label || c.label || c.cli_id || c.id;
        const kind = kindOf(c);
        const still = thumb(c, 'cand-thumb', true);
        // the length shows once, on the still; it is also the screen reader's only length
        still.append(h('span', { class: 'cand-dot', 'aria-hidden': 'true' }), h('span', { class: 'cand-len tc', text: `${(c.end - c.start).toFixed(1)} s` }));
        return h('li', { class: `cand${picked ? ' picked' : ''}`, 'data-id': c.id, 'data-kind': kind, 'aria-current': c.id === S.focus ? 'true' : null },
          // the hover tooltip: the full title and the transcript (the tile shows neither in full)
          h('button', { type: 'button', class: 'cand-main', 'data-act': 'focus', 'aria-describedby': tip, title: [title, c.bit_description, c.transcript].filter(Boolean).join('\n'), onclick: () => focusMoment(c.id) },
            still,
            h('span', { class: 'cand-top' },
              h('span', { class: 'rank tc', text: `#${rank}` }),
              h('span', { class: 'cand-title', text: title }),
              c.must_include && !crashOf(c) ? h('span', { class: 'tag', text: 'must include' }) : null,
              crashBadge(c),
              bitBadge(c),
              poolBadge(c)),
            h('span', { class: 'cand-meta' }, KINDS[kind], h('span', { 'aria-hidden': 'true', text: ' · ' }), h('span', { class: 'vh', text: ', score ' }), scoreText(c.score, c.pool === 'visual'), badge),
            // what the dialogue reader says happens (a dialogue moment): two lines at most, in the button's name
            c.bit_description ? h('span', { class: 'cand-what', text: c.bit_description }) : null,
            h('span', { class: 'cand-time tc' },
              // the CLI id (h001) as text, for Candidate ids and clips --ids (#487)
              c.cli_id ? [h('span', { class: 'cand-id' }, h('span', { class: 'vh', text: 'Candidate ' }), c.cli_id), ' '] : null,
              // the start-end range: hidden with the id and the file unless Verbose moment labels is on
              h('span', { class: 'cand-range' }, `${fmt(c.start)} to ${fmt(c.end)}`)),
            multi ? h('span', { class: 'cand-src' }, h('span', { class: 'vh', text: 'in ' }), baseName(c.source)) : null,
            flagBadges(c.pano),
            sourceStrip(c),
            // read after the rest: aria-describedby (camera path tiles) overrides the title, which holds the transcript
            tip && c.transcript ? h('span', { class: 'vh', text: `, ${c.transcript}` }) : null),
          // an icon button over the still's top right corner: a dark + on amber, a white - on red
          h('button', {
            type: 'button', class: 'cand-toggle', 'data-act': 'toggle',
            'aria-label': `${picked ? 'Remove' : 'Add'} ${title} ${picked ? 'from' : 'to'} sequence`,
            title: `${picked ? 'Remove from' : 'Add to'} sequence`,
            onclick: () => (picked ? removeSeg(c.id) : addSeg(c.id)),
          }, icon(picked ? 'minus' : 'plus')),
          // the reasons of the camera path badges (#517): shown on hover and keyboard focus, read as the card's description
          tip ? h('span', { class: 'flag-tip', role: 'tooltip', id: tip, text: `Camera path looks poor: ${c.pano.reasons.map((r) => r.text).join('; ')}.` }) : null,
        );
      }),
    );
  });
  if (redraw && none) $('#candList').replaceChildren(h('li', { class: 'empty', text: none }));
  // the focused card left the list (Add with Not in sequence, say) and no card is left to take focus
  if (listHadFocus && !$('#candList').contains(document.activeElement) && !$('#filterWrap').hidden) $('#filterBtn').focus();
  markMoments();
  renderBusy(); // first: the buttons are as disabled as they will be when focus looks for one
  if (hadFocus && !empty.contains(document.activeElement))
    ([$('#resumeBig'), $('#analyzeBig'), $('#pauseBtn'), $('#candList button'), $('#resumeBtn')].find((b) => b && !b.hidden && !b.disabled) ?? $('#momentsTitle')).focus();
}

/**
 * Everything the Moments tiles show, as two strings: ``list``, the project's moments, and
 * ``rest``, each shown tile (its rank, whether it is in the sequence, its preview readiness), the
 * focused moment, the sources' urls (stills), lengths and unplayable parts (strips), whether the
 * file names show and the no-match text. Equal strings draw the same tiles.
 */
function tilesKey(list, items, sources, multi, none) {
  const srcs = sources.map((path) => {
    const src = srcOf(path);
    const dur = sourceDur(path);
    return [path, src?.url ?? null, dur ?? null, dur ? sourceGaps(src, dur) : []];
  });
  const tiles = items.map(({ c, rank }) => {
    const seg = S.segments.find((s) => s.id === c.id);
    return [rank, !!seg, prepState(srcOf(c.source), seg ?? { ...c, ...S.drafts[c.id] })];
  });
  return { list: JSON.stringify(list), rest: JSON.stringify([tiles, srcs, S.focus, multi, none]) };
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

/**
 * The readiness of a row's preview as a small icon and word (never colour alone); null when there is
 * nothing to say. ``iconOnly`` (a Moments tile): the icon alone (a spinner ring while encoding), the
 * word as its hover title and as screen reader text.
 */
function prepBadge(src, seg, iconOnly = false) {
  const state = prepState(src, seg);
  if (!state) return null;
  const [name, word] = PREP[state];
  const title = `Preview ${word.toLowerCase()}`;
  if (iconOnly)
    return h('span', { class: 'prep icon-only', 'data-state': state, title },
      state === 'encoding' ? [h('span', { class: 'prep-spin', 'aria-hidden': 'true' }), icon(name)] : icon(name),
      h('span', { class: 'vh', text: `, ${title.toLowerCase()}` }));
  return h('span', { class: 'prep', 'data-state': state, title },
    icon(name), h('span', { class: 'vh', text: 'Preview ' }), word);
}

/** The name of a moment whose row is called ``label``: the Moments list's title (display_label,
 * what the vision model saw for a generic label; #634) unless the row was renamed. */
function shownName(cand, label) {
  return cand?.display_label && (label == null || label === cand.label) ? cand.display_label : label;
}

/** The labels the list filter searches: the title, the analysis label and the vision label. */
const filterLabel = (c) => [c.display_label, c.label, c.vision_label].filter(Boolean).join(' ');

/** A long dialogue bit's badge: "Bit · 2:05" (the length in m:ss; words for a screen reader, never colour alone), or null. */
function bitBadge(c) {
  if (!c.long_bit) return null;
  return h('span', { class: 'tag bit' }, 'Bit', h('span', { 'aria-hidden': 'true', text: ' · ' }), h('span', { class: 'vh', text: ', ' }), fmtLen(c.end - c.start));
}

/**
 * "CRASH 0.91": a moment with a crash of crashes.json (util.js crashOf), with its confidence; a solid
 * outline when it is always kept (forced), dashed when it is ranked like any moment. Words say which
 * (the title and the screen reader), never the outline alone. Null for no crash.
 */
function crashBadge(c) {
  const crash = crashOf(c);
  if (!crash) return null;
  const how = crash.forced ? 'always kept' : 'ranked like any moment';
  return h('span', { class: `tag crash${crash.forced ? '' : ' weak'}`, title: `Crash found, ${how}` }, crashText(crash), h('span', { class: 'vh', text: `, ${how}` }));
}

/** "visual · view": a moment of the visual pool (#635: beauty shots, friends, cruising runs that
 * analysis adds without a sound or speech cue), with its kind when it has one. */
function poolBadge(c) {
  if (c.pool !== 'visual') return null;
  const kind = typeof c.kind === 'string' && c.kind ? c.kind : null;
  return h('span', { class: 'tag pool' }, 'visual',
    kind ? [h('span', { 'aria-hidden': 'true', text: ' · ' }), h('span', { class: 'vh', text: ', ' }), kind] : null);
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

/**
 * Project open but not analyzed: source count (or why the folder cannot be read) and the one button that starts analysis; ``pano``: a 360 project (n counts its recordings).
 * ``run``: the project's recorded run (#109); one that stopped puts Resume first. ``lone``: the project has only that record (a first
 * command-line analysis stopped before its moments), so there is nothing to analyze but Resume.
 */
function renderUnanalyzed(el, n, error, pano, run = null, lone = false) {
  const resume = !!run?.resumable;
  const state = `${n}|${error}|${pano}|${resume && run.command}|${lone}`;
  if (el.dataset.n === state) return; // polling re-renders; keep the button (and its focus)
  el.dataset.n = state;
  setEmpty(el,
    folderArt(),
    lone ? null : error
      ? h('p', { class: 'fresh-error', text: error })
      : h('p', { class: `fresh-count fresh-lead${n ? '' : ' none'}` },
        pano
          ? (n ? `360 footage: ${n} recording${n === 1 ? '' : 's'} found` : 'No 360 recordings found')
          : (n ? `${n} video${n === 1 ? '' : 's'} found in the sources` : 'No videos found in the sources')),
    h('h3', { class: 'fresh-lead', text: resume ? `The ${RUN_NAMES[run.command] ?? 'analysis'} stopped before it finished` : 'No moments analyzed yet' }),
    h('p', {
      class: 'fresh-lead', text: resume ? 'Resume continues it with the same options and keeps the work it finished.' : pano
        ? 'Analyze footage to read the 360 recordings, transcribe audio and rank the strongest moments. Clips are reframed when you export them. Vision rating is not used for 360 footage.'
        : 'Analyze footage to transcribe audio and rank the strongest moments.',
    }),
    resume ? h('button', {
      type: 'button', id: 'resumeBig', class: 'btn primary', 'aria-haspopup': 'dialog', 'data-label': resumeLabel(run),
      onclick: (e) => resumeRun(e.currentTarget),
    }, resumeLabel(run)) : null,
    h('button', {
      type: 'button', id: 'analyzeBig', class: `btn ${resume ? 'outline' : 'primary'}`, 'aria-describedby': 'analyzeWhy analyzeNote', hidden: lone, 'data-gone': lone ? '1' : null,
      onclick: async (e) => {
        e.currentTarget.disabled = true; // no second start while the request is in flight
        keepEmptyFocus(el);
        await startJob('analyze'); // vision: the server applies the Settings value
        if (![...S.tracked.values()].some(live)) renderBusy(); // it failed: enable again
      },
    }, 'Analyze footage'),
    // while the first analysis runs (renderBusy): the way to stop it, to continue with Resume
    h('button', {
      type: 'button', id: 'pauseBtn', class: 'btn primary', hidden: true, 'aria-describedby': 'pauseWhy',
      onclick: (e) => pauseAnalysis(e.currentTarget),
    }, 'Pause analysis'),
    h('span', { id: 'pauseWhy', class: 'vh', text: 'Stops the analysis. Resume continues it and keeps the videos it finished; before the first is transcribed, it starts over.' }),
    // the Analyze options (#483) for the first analysis; Analyze footage alone uses none
    h('button', {
      type: 'button', id: 'analyzeOptsBtn', class: 'btn outline', 'aria-haspopup': 'dialog', hidden: pano || lone, 'data-gone': pano || lone ? '1' : null,
      onclick: () => analyzeAgain(true),
    }, 'Analysis options…'),
    h('button', {
      type: 'button', id: 'revealBtn', class: 'btn outline', hidden: !S.canReveal || lone,
      onclick: (e) => revealFolder(e.currentTarget),
    }, icon('folder'), 'Show folder'),
    h('p', { id: 'analyzeWhy', class: 'fresh-why', 'aria-live': 'polite' }),
    h('div', { id: 'freshSteps', class: 'steps', hidden: true }), // the running analysis's steps (renderSteps)
    h('p', { id: 'freshStepsStatus', class: 'vh', role: 'status' }),
    h('p', { id: 'analyzeNote', class: 'fresh-note', text: 'This process can take a long time on large footage.' }),
  );
}

/**
 * Pause the first analysis (#811): cancel the job with its normal cleanup; the project then has a
 * stopped run and the empty state offers Resume. aria-disabled, not disabled, so the button keeps focus.
 */
async function pauseAnalysis(btn) {
  if (!S.active || btn.getAttribute('aria-disabled') === 'true') return;
  // progress.json (what Resume continues) exists once the first source is transcribed; the page
  // does not refresh the project during its own job, so read the server's record first
  const transcribed = stepsJob()?.progress.steps.some((s) => s.step === 'transcribe' && s.state === 'done');
  let rec = S.project?.run;
  if (!rec && !transcribed) {
    try {
      rec = (await api(`/api/projects/${encodeURIComponent(S.id)}`)).run;
    } catch { /* cannot tell: ask */ }
  }
  if (!rec && !transcribed) {
    const ok = await confirmAction('Stop the analysis?', 'No video is transcribed yet, so the analysis starts over.', 'Stop', false);
    if (!S.active || btn.hidden || !btn.isConnected) { // the run ended while the dialog was open: focus would drop to the body
      ([$('#resumeBig'), $('#analyzeBig'), $('#revealBtn')].find((b) => b && !b.hidden && !b.disabled) ?? $('#momentsTitle')).focus();
      return;
    }
    if (!ok) return;
  }
  if (!S.active) return;
  btn.setAttribute('aria-disabled', 'true');
  btn.textContent = 'Pausing…';
  try {
    await api(`/api/jobs/${encodeURIComponent(S.active.id)}/cancel`, { method: 'POST' });
    pollJobs();
  } catch (e) {
    btn.removeAttribute('aria-disabled');
    btn.textContent = 'Pause analysis';
    fail(e);
  }
}

// What a recorded run (progress.json's command, #109) is called on the page
const RUN_NAMES = {
  analyze: 'analysis', '360 analyze': '360 analysis', edit: 'one-click edit', clips: 'clip export',
  events: 'event detection', crashes: 'crash detection', profile: 'activity profile',
};
const RUN_ENDS = { interrupted: 'was stopped', low_memory: 'stopped because memory ran short', failed: 'failed' };
// The page's words for a project another run holds (the server's BUSY_TEXT, #109)
const BUSY_TEXT = 'Analysis is running on this project; wait for it to finish';
/** Another run holds the open project: one of the command line, or of another server (not this page's own job). */
const heldElsewhere = (p) => !!p?.busy && S.active?.project_id !== p.id;
const resumeLabel = (run) => (run.command === 'clips' ? 'Resume export' : run.command === 'edit' ? 'Resume one-click edit' : run.command === 'crashes' ? 'Resume crash detection' : run.command === 'events' ? 'Resume closer look' : 'Resume analysis');

/**
 * Resume the project's stopped run (#109): the server's ``cutpoint resume --dry-run`` says what it continues, a
 * dialog asks, then ``cutpoint resume`` runs as a job.
 */
async function resumeRun(btn) {
  if (!S.project || S.resumeCheck) return; // one check at a time
  const id = S.id;
  // aria-disabled, not disabled: the button keeps focus, so the dialog gives it back
  S.resumeCheck = true;
  btn.setAttribute('aria-disabled', 'true');
  let info;
  try {
    info = await api(`/api/projects/${encodeURIComponent(id)}/resume`);
    banner('');
  } catch (e) {
    fail(e);
    return;
  } finally {
    S.resumeCheck = false;
    btn.removeAttribute('aria-disabled');
    renderBusy();
  }
  if (id !== S.id) return; // another project was opened meanwhile
  if (info.status === 'done') { // it finished after all (another run, say)
    refreshProject();
    return;
  }
  const what = RUN_NAMES[info.command] ?? info.command;
  const steps = Object.entries(info.steps ?? {}).map(([name, s]) => {
    const state = s.total != null ? `${s.done ?? 0} of ${s.total} done` : (STEP_TEXT[s.state] ?? s.state ?? 'not started').toLowerCase();
    return `${STEP_LABELS[name] ?? name}: ${state}`;
  });
  const text = [
    `The last ${what} ${RUN_ENDS[info.status] ?? 'stopped'}.`,
    steps.length ? `So far: ${steps.join('; ')}.` : '',
    'Resume runs it again with the same options and keeps the work it finished.',
    ...(info.warnings ?? []).map(sentence),
    // the recorded command line as it is replayed: a --force or an --out elsewhere shows here
    info.argv?.length ? `Command: cutpoint ${commandLine(info.argv)}` : '',
  ].filter(Boolean).join(' ');
  if (!(await confirmAction(`${resumeLabel({ command: info.command })}?`, text, 'Resume', false))) return;
  if (btn.id === 'resumeBig') keepEmptyFocus($('#momentsEmpty')); // the empty state goes when the job starts
  await startJob('resume');
}

/** A command line as typed: arguments with spaces or quotes in double quotes. */
const commandLine = (argv) => argv.map((a) => (/[\s"]/.test(a) || a === '' ? `"${a.replace(/"/g, '\\"')}"` : a)).join(' ');

/** Ask the server to show a project folder in the file manager: 'sources', 'clips', 'outputs' or 'project'. */
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
  const j = S.seqPlay?.across ? -1 : S.seqPlay?.order.indexOf(id) ?? -1;
  if (j >= 0 && seqSeg(j)) return seqGo(j);
  seqStop(undefined, false);
  showMoment(id);
}

/** ``keepVideo``: the player is already at the moment (a sequence step), so it is not sought. */
function showMoment(id, keepVideo = false) {
  S.focus = id;
  for (const li of [...$('#candList').children, ...$('#seqList').children, ...$('#removedList').children, ...sbBlocks()]) {
    if (li.dataset.id === id) li.setAttribute('aria-current', 'true');
    else li.removeAttribute('aria-current');
  }
  // the storyboard's tab stop follows the selection; a moment outside the sequence leaves it on the first block
  const blocks = sbBlocks();
  const stop = blocks.find((b) => b.dataset.id === id) ?? blocks[0];
  for (const b of blocks) b.tabIndex = b === stop ? 0 : -1;
  renderStage(true, keepVideo);
}

// ---------- rendering: stage ----------

/** Where a moved trim point seeks: its point; during "Play sequence" a new end gets 2 s of pre-roll, so the cut is seen. */
function trimSeek(which) {
  const c = current();
  return S.seqPlay && which === 'end' ? Math.max(c.start, c.end - 2) : c[which];
}

/** ``shown``: on screen too; else only screen readers get it, as the cut line says the same. */
function setTrimMsg(text, bad = false, shown = bad) {
  const m = $('#trimMsg');
  if (m.textContent !== text) m.textContent = text;
  m.classList.toggle('error', bad);
  m.classList.toggle('vh', !shown);
}

/** ``widen``: a point that left the rail widens it to show both; the trim handles keep the zoom and pan instead. */
function setRange(which, t, widen = true) {
  const c = current();
  if (!c) return false;
  t = Math.max(0, Math.min(round2(t), handleRange('end', c).max)); // the source's end can round up past itself
  const s = which === 'start' ? t : c.start;
  const e = which === 'end' ? t : c.end;
  const err = validate(s, e, durOf(c));
  if (err) {
    renderTrim();
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
  renderTrim();
  return true;
}

function renderStage(seekToStart, keepVideo = false) {
  if (trimDrag && dragStale(trimDrag)) dropDrag(); // the reused handle loses its drag state at once
  if (S.cutMode && cutIndex() < 0) { // too few clips left, or a moment outside the sequence opened
    $('#sideLive').textContent = S.segments.length < 2 ? `${CUT_NEEDS} Back to Clip mode.` : 'Back to Clip mode.';
    if ($('#cutView').contains(document.activeElement)) $('#stageMode [data-mode=clip]').focus(); // hidden next: it would fall to the body
    setCut(false);
  }
  const c = current();
  $('#stageBody').hidden = !c;
  $('#toggleBtn').hidden = !c;
  renderClipHead(c);
  renderFullPrep(c);
  renderNerdStats(c);
  renderPanoNote(c);
  if (!S.project)
    setEmpty($('#stageEmpty'),
      h('p', {}, 'No project is open.'),
      h('p', { class: 'muted' }, 'Create one from a video file or a folder of footage on this computer. Source files are never changed.'),
      h('button', { type: 'button', class: 'btn primary', onclick: () => openDialog(S.styles), text: 'New project' }));
  else if (!c)
    setEmpty($('#stageEmpty'), ...(S.project.highlights ? [h('p', {}, 'Pick a moment to preview it.')] : [
      svg('icon stage-empty-art', ICONS.film),
      h('h3', { text: 'No moments yet' }),
      // a stopped run waits for Resume in the Moments column: no start-over button here
      S.project.run?.resumable
        ? h('p', {}, 'Resume the analysis in the Moments column to find its best moments.')
        : h('p', {}, 'Analyze the footage to find its best moments. Preview and trim them here.'),
      // the same action as the Analyze footage button of the Moments column
      S.project.run?.resumable ? null : h('button', { type: 'button', id: 'stageAnalyze', class: 'btn primary', onclick: () => $('#analyzeBig')?.click(), text: 'Analyze footage' }),
    ]));
  else setEmpty($('#stageEmpty'));
  $('#stageEmpty').classList.toggle('stage-empty', !c && !S.project?.highlights); // no project, or not analyzed: one centred layout
  syncStageAnalyze();
  if (!c) {
    v.pause();
    return;
  }
  if (seekToStart) S.view = null; // a new moment starts on its fit view
  ensureVideo(seekToStart && !keepVideo ? c : null);
  renderTrim();
}

/**
 * The stage's title row: kind, the clip's place in the sequence (else the moment's name), score and
 * source, and the buttons that step through the sequence. A step button that just reached the end
 * hands focus to its partner instead of dropping it.
 */
function renderClipHead(c = current()) {
  if (!c) return;
  if (S.cutMode && cutIndex() >= 0) return renderCutHead();
  $('#stageTitle').removeAttribute('aria-label'); // Cut mode's
  const i = c.seg ? S.segments.indexOf(c.seg) : -1;
  const name = shownName(c.cand, c.seg ? c.label : null) || c.label;
  const kind = kindOf(c.cand);
  $('#stageDot').dataset.kind = kind ?? '';
  $('#stageTitle').textContent = i >= 0 ? `Clip ${i + 1} · ${name}` : name;
  $('#stageTitle').title = name;
  const meta = $('#stageMeta');
  meta.textContent = [KINDS[kind], c.cand && `score ${scoreText(c.cand.score, c.cand.pool === 'visual')}`, srcOf(c.source)?.name ?? baseName(c.source)].filter(Boolean).join(' · ');
  meta.title = c.source;
  $('#toggleBtn').textContent = c.seg ? 'Remove from sequence' : 'Add to sequence';
  const nav = $('#clipNav');
  const prev = $('#prevClip');
  const next = $('#nextClip');
  const had = nav.contains(document.activeElement) ? document.activeElement : null; // read first: disabling drops focus
  nav.hidden = i < 0;
  prev.disabled = i <= 0;
  next.disabled = i < 0 || i >= S.segments.length - 1;
  prev.textContent = prev.disabled ? '‹ Previous' : `‹ Clip ${i}`;
  next.textContent = next.disabled ? 'Next ›' : `Clip ${i + 2} ›`;
  prev.setAttribute('aria-label', prev.disabled ? 'Previous clip' : `Previous clip, Clip ${i}`);
  next.setAttribute('aria-label', next.disabled ? 'Next clip' : `Next clip, Clip ${i + 2}`);
  if (had && (had.disabled || nav.hidden)) {
    const other = had === prev ? next : prev;
    (nav.hidden || other.disabled ? $('#toggleBtn') : other).focus();
  }
}

/** Open the clip ``d`` places on in the sequence from the open one. */
function stepClip(d) {
  const c = current();
  const to = c?.seg && S.segments[S.segments.indexOf(c.seg) + d];
  if (to) focusMoment(to.id);
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
  // laughter has two sources, shown apart: said in the transcript (signals.laughter) and heard in the audio (signals.events.laughter)
  const bits = sig
    ? Object.entries(sig).filter(([k, n]) => k !== 'laughter' && typeof n === 'number' && n > 0).map(([k, n]) => `${k} ${n.toFixed(2)}`)
    : [];
  const laugh = (n) => (typeof n === 'number' && n > 0 ? n.toFixed(2) : '');
  const written = laugh(sig?.laughter);
  const heard = laugh(sig?.events?.laughter);
  const all = cands();
  el.textContent = [
    `Score ${cand.score.toFixed(2)}, rank ${all.indexOf(cand) + 1} of ${all.length}`,
    `Analyzed ${fmt(cand.start)} to ${fmt(cand.end)}`,
    bits.length ? `Signals: ${bits.join(', ')}` : '',
    written ? `Laughter in the transcript: ${written}` : '',
    heard ? `Laughter heard in the audio: ${heard}` : '',
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

function renderTrim() {
  const c = current();
  if (!c) return;
  if (S.cutMode) return renderCut();
  drawRail(c);
  const len = c.end - c.start;
  const out = exportedLen(c);
  const differs = Math.abs(out - len) >= 0.005;
  const why = planned(c).reason;
  const long = Math.max(len, out) > LONG_CLIP;
  setTrimMsg(`Clip length ${len.toFixed(2)} s${differs ? ` · exports as ${out.toFixed(2)} s${why ? ` (${why})` : ''}` : ''}${
    c.cand ? `, analyzed as ${fmt(c.cand.start)} to ${fmt(c.cand.end)}` : ''}.${
    long ? ' This clip is longer than 10 minutes, so exporting it takes a while.' : ''}`, false, long);
}

/** The line under the player: the trim, what the export adds to it, and Reset. */
function drawCut(c) {
  const len = c.end - c.start;
  const out = exportedLen(c);
  $('#cutRange').textContent = `${tenths(c.start)} → ${tenths(c.end)}`;
  $('#cutLen').textContent = `| ${len.toFixed(2)} s`;
  $('#cutOut').textContent = `${out.toFixed(2)} s`;
  const added = round2(out) - round2(len); // of the rounded lengths, so the numbers on the line add up
  const tail = $('#cutTail');
  tail.hidden = added < 0.005;
  tail.textContent = tail.hidden ? '' : `+${added.toFixed(2)} s ${planned(c).reason ?? 'tail'}`;
  const reset = $('#resetBtn');
  const had = document.activeElement === reset; // read first: disabling a focused button drops its focus
  reset.disabled = !c.cand || (c.start === c.cand.start && c.end === c.cand.end);
  reset.title = c.cand ? `Back to the analyzed ${fmt(c.cand.start)} to ${fmt(c.cand.end)}` : '';
  if (reset.disabled && had) $('#startHandle').focus();
}

/**
 * What the export adds to the trim, dashed outside it: from the end to the planned end and, when the
 * planned start is earlier, from there to the start. The handles keep editing the raw range.
 */
function drawTails(box, c, from, len, place) {
  const e = planned(c);
  const strips = [];
  if (e.end > c.end + 1e-6) strips.push([c.end, e.end]);
  if (e.start < c.start - 1e-6) strips.push([e.start, c.start]);
  box.replaceChildren(...strips.map(([a, b]) => {
    const el = h('div', { class: 'rail-tail' });
    place(el, a, b, from, len);
    return el;
  }));
}

/** The rail (the view, with the clip and analyzed range on it) and the overview strip of the whole source. */
function drawRail(c = current()) {
  if (!c) return;
  if (S.cutMode) return renderCut(); // the rail is hidden: no stills are asked for it
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
  drawTails($('#railTail'), c, lo, span, place);
  drawGaps(c, dur, lo, hi, place);
  drawFilm(c);
  drawWave(c);
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
    drawTails($('#ovTail'), c, 0, dur, place);
  }
  const t = playhead(c);
  setWasIn(t >= lo && t <= hi);
  // a marker the view cuts off loses its edge border on that side
  const sel = $('#railSel');
  sel.classList.toggle('cut-l', c.start < lo);
  sel.classList.toggle('cut-r', c.end > hi);
  drawHandles(c, lo, hi);
  drawGhost(c);
  drawRuler(lo, hi);
  drawEvents(c, lo, hi);
  drawCut(c);
  const hadPreset = document.activeElement; // read first: disabling a focused button drops its focus
  for (const b of $('#zoomPresets').children) {
    const want = Number(b.dataset.span);
    b.disabled = !!dur && want > dur + 1e-6; // a source shorter than the level cannot show it
    b.setAttribute('aria-pressed', String(!b.disabled && Math.abs(span - want) < 0.05));
    if (b.disabled && hadPreset === b) seek.focus();
  }
  updatePlayhead();
}

const RULER_STEPS = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
const RULER_GAP = 64; // pixels: the least room between two ruler labels
const RULER_ENDS = [72, 112]; // pixels: kept clear at the left and right for the view's end labels, in the ruler's row
const EVENT_GAP = 80; // pixels: the least room between two moment labels under the rail
const EVENTS_MAX = 8;

/** Time ruler under the rail: a tick and a label at multiples of a round step that leaves RULER_GAP pixels. */
function drawRuler(lo, hi) {
  const el = $('#railRuler');
  const span = hi - lo;
  const width = $('#rail').clientWidth || 600;
  const step = RULER_STEPS.find((x) => (x / span) * width >= RULER_GAP) ?? RULER_STEPS.at(-1);
  const sig = `${lo}|${hi}|${width}`;
  if (el.dataset.sig === sig) return;
  el.dataset.sig = sig;
  const ticks = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi; t += step) {
    const at = ((t - lo) / span) * 100;
    if ((at / 100) * width < RULER_ENDS[0] || (1 - at / 100) * width < RULER_ENDS[1]) continue;
    ticks.push(h('span', { class: 'rail-tick', style: `left: ${at.toFixed(2)}%`, text: hms(t).replace(/^0:0?/, '') }));
  }
  el.replaceChildren(...ticks);
}

/** The detected events (``GET .../events``) of the open project by source path; ``sig`` is the project's ``events_sig``: it tells when ``events.json`` or ``crashes.json`` changed. */
const railEvents = { id: null, sig: '', by: new Map() };
const EVENT_NAMES = { audio: 'Sound', ocr: 'On-screen text', telemetry: 'Camera', text: 'Talk' };

/** Fetch the open project's events once (again when its events.json changed); the markers fall back to the other moments until they arrive or when there are none. */
async function loadRailEvents(p) {
  const sig = String(p.events_sig ?? '');
  if (railEvents.id === p.id && railEvents.sig === sig) return;
  Object.assign(railEvents, { id: p.id, sig, by: new Map() });
  try {
    const { events } = await api(`/api/projects/${encodeURIComponent(p.id)}/events`);
    if (railEvents.id !== p.id || railEvents.sig !== sig) return;
    const by = new Map();
    for (const e of Array.isArray(events) ? events : []) by.set(e.source, [...(by.get(e.source) ?? []), e]);
    railEvents.by = by;
    redrawEvents();
  } catch {
    // decorative: the markers stay the other moments
  }
}

function redrawEvents() {
  const c = current();
  if (c && S.view) drawEvents(c, S.view.lo, S.view.hi);
}

/**
 * Markers under the rail, at most EVENTS_MAX, EVENT_GAP apart: the detected events of this source in the view, the most confident first
 * (a title gives kind, time and confidence), else, for a source with none, the other moments that start in the view, coloured by kind.
 */
function drawEvents(c, lo, hi) {
  const el = $('#railEvents');
  const span = hi - lo;
  const width = $('#rail').clientWidth || 600;
  const found = railEvents.by.get(c.source) ?? [];
  const marks = [];
  const room = (at) => marks.every((m) => (Math.abs(at - m.at) / 100) * width >= EVENT_GAP);
  if (found.length) {
    const seen = found.filter((e) => e.end >= lo && e.start <= hi)
      .sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0) || a.start - b.start);
    for (const e of seen) {
      const at = ((Math.max(e.start, lo) - lo) / span) * 100;
      if (marks.length >= EVENTS_MAX || !room(at)) continue;
      const sure = e.confidence == null ? '' : `, confidence ${e.confidence}`;
      const kind = EVENT_NAMES[e.kind] ?? e.kind;
      // a stretch (talk) shows its length: to its end or the view's
      const wide = e.kind === 'text' ? ((Math.min(e.end, hi) - Math.max(e.start, lo)) / span) * 100 : 0;
      marks.push({
        at, name: e.label, ev: e.kind, ...(wide > 0 ? { wide } : {}),
        title: `${kind}: ${e.label}, ${fmt(e.start)}${wide > 0 ? ` to ${fmt(e.end)}` : ''}${sure}`,
        said: `${kind}: ${e.label} at ${fmt(e.start)}${wide > 0 ? ` to ${fmt(e.end)}` : ''}`,
      });
    }
  } else {
    const near = cands().filter((x) => x.id !== c.id && x.source === c.source && x.start >= lo && x.start <= hi)
      .sort((a, b) => a.start - b.start);
    let last = -Infinity;
    for (const x of near) {
      const at = ((x.start - lo) / span) * 100;
      if (marks.length >= EVENTS_MAX || ((at - last) / 100) * width < EVENT_GAP) continue;
      last = at;
      const name = shownName(x, x.label) || x.id;
      marks.push({ at, name, kind: kindOf(x), title: `${name}, ${fmt(x.start)} to ${fmt(x.end)}` });
    }
  }
  marks.sort((a, b) => a.at - b.at);
  const sig = marks.map((m) => `${m.at.toFixed(2)}${m.name}${m.ev ?? ''}${m.wide?.toFixed(2) ?? ''}`).join('|');
  if (el.dataset.sig === sig) return;
  el.dataset.sig = sig;
  // the markers are aria-hidden: the events they show are read from this visually hidden line
  $('#railEventsText').textContent = marks.filter((m) => m.said).map((m) => m.said).join('; ');
  el.replaceChildren(...marks.map((m) => {
    const props = {
      class: m.wide ? 'rail-event range' : 'rail-event', ...(m.ev ? { 'data-ev': m.ev } : { 'data-kind': m.kind }),
      style: `left: ${m.at.toFixed(2)}%${m.wide ? `; width: ${m.wide.toFixed(2)}%` : ''}`, title: m.title,
    };
    // a range's label sits in its own box: as wide as the range it would vanish when narrow, and run under later labels when long
    return m.wide ? h('span', props, h('span', { class: 'rail-event-name', text: m.name })) : h('span', { ...props, text: m.name });
  }));
}

// The audio envelope behind the rail's markers (``GET .../envelope``: ``step`` seconds per value in ``peaks``, 0-1): fetched once
// per source, drawn as one SVG path of vertical bars for the view. Purely decorative, like the filmstrip.
const WAVE_COL = 3; // pixels: the least width of a bar
const WAVE_RETRY_MS = 5000; // between asks for an envelope the server is still decoding
const envelopes = new Map(); // `${project id}|${source path}` -> {step, peaks} | a Symbol (loading) | 'none'

/** The envelope of ``c``'s source if it is loaded, else null (and ask for it once). */
function envelopeOf(c) {
  const src = srcOf(c.source);
  if (!src?.url || filmOff()) return null;
  const key = `${S.id}|${c.source}`;
  const have = envelopes.get(key);
  if (have) return typeof have === 'object' ? have : null;
  const token = Symbol('loading'); // one per request chain: a chain of an earlier visit stops when this one replaces it
  envelopes.set(key, token);
  fetchEnvelope(key, token, c.source, `${src.url.split('?')[0]}/envelope`);
  return null;
}

/** Whether ``source``'s waveform is on screen: the focused clip's, or in Cut mode either side of the current cut. */
function onScreen(source) {
  if (current()?.source === source) return true;
  const k = S.cutMode ? cutIndex() : -1;
  return k >= 0 && (S.segments[k].source === source || S.segments[k + 1].source === source);
}

/** Ask for an envelope; while the server is still decoding it (``pending``) ask again after WAVE_RETRY_MS, until it is answered, the project changes or the user leaves the source. */
function fetchEnvelope(key, token, source, url) {
  api(url).then((d) => {
    if (envelopes.get(key) !== token) return; // another project was opened meanwhile
    if (d?.pending) {
      setTimeout(() => {
        if (envelopes.get(key) !== token) return;
        if (!onScreen(source)) envelopes.delete(key); // left: the next draw of it starts a new chain
        else fetchEnvelope(key, token, source, url);
      }, WAVE_RETRY_MS);
      return;
    }
    envelopes.set(key, Array.isArray(d?.peaks) && d.peaks.length && d.step > 0 ? d : 'none');
    if (S.cutMode) renderCut();
    else if (current()?.source === source) drawWave();
  }, () => envelopes.get(key) === token && envelopes.set(key, 'none'));
}

/** Draw the envelope of the current source over the view. */
function drawWave(c = current()) {
  const env = c && S.view ? envelopeOf(c) : null;
  paintWave($('#railWave'), env, c?.source, S.view?.lo, S.view?.hi, $('#rail').clientWidth || 600);
}

/** Draw ``env`` of ``source`` over [lo, hi] in ``svg``: a bar per at least WAVE_COL pixels, the loudest value in it, tall by its square root. */
function paintWave(svg, env, source, lo, hi, width) {
  const sig = env ? `${source}|${lo}|${hi}|${width}|${env.peaks.length}` : '';
  if (svg.dataset.sig === sig) return;
  svg.dataset.sig = sig;
  const path = svg.firstElementChild;
  if (!env) {
    path.removeAttribute('d');
    return;
  }
  const cols = Math.max(1, Math.min(Math.floor(width / WAVE_COL), Math.ceil((hi - lo) / env.step)));
  const w = (hi - lo) / cols;
  const bars = [];
  for (let j = 0; j < cols; j += 1) {
    if (lo + (j + 1) * w <= 0) continue; // Cut mode's track can reach before the source
    const from = Math.floor(Math.max(0, lo + j * w) / env.step);
    const to = Math.max(from, Math.ceil((lo + (j + 1) * w) / env.step) - 1);
    let top = 0;
    for (let i = from; i <= Math.min(to, env.peaks.length - 1); i += 1) top = Math.max(top, env.peaks[i]);
    const half = Math.sqrt(top) * 46;
    if (half > 0.5) bars.push(`M${j + 0.5} ${(50 - half).toFixed(1)}V${(50 + half).toFixed(1)}`);
  }
  svg.setAttribute('viewBox', `0 0 ${cols} 100`);
  if (bars.length) path.setAttribute('d', bars.join(''));
  else path.removeAttribute('d');
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

/** Move a trim point, clamped to its handle's range, through the same path as the I and O keys. */
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
new ResizeObserver(() => {
  drawFilm();
  if (S.view) { // the ruler, the waveform and the event labels depend on the width too
    drawRuler(S.view.lo, S.view.hi);
    drawWave();
    redrawEvents();
  }
}).observe($('#rail')); // a wider rail holds more stills

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

// The stage players and the source proxies they play are in player.js.
bindPlayer();

$('#zoomPresets').addEventListener('click', (e) => {
  const b = e.target.closest('button[data-span]');
  const c = current();
  if (!b || !c || !S.view) return;
  if (b.getAttribute('aria-pressed') === 'true') { // pressed again: back to the clip's fit view
    const f = fitView(c);
    setView(f.lo, f.hi);
  } else zoomBy(Number(b.dataset.span) / (S.view.hi - S.view.lo));
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
  const exact = d ? d.exact : c.exact; // a word splice stays one when it is added back
  S.segments.splice(at, 0, { id, source: c.source, start: d?.start ?? c.start, end: d?.end ?? c.end, label: d?.label ?? c.label ?? id, ...(exact ? { exact: true } : {}) });
  delete S.drafts[id];
  renderAfterSeqChange();
  return at;
}

/** Keep a removed clip's trim and index ``at`` for re-adding; a clip that is no moment is kept
 * whole, so the stage and the Removed list can still offer it after it leaves the sequence. */
function keepDraft(seg, at) {
  const exact = seg.exact ? { exact: true } : {};
  S.drafts[seg.id] = cands().some((c) => c.id === seg.id)
    ? { start: seg.start, end: seg.end, label: seg.label, at, ...exact }
    : { source: seg.source, start: seg.start, end: seg.end, label: seg.label, at, ...exact };
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
  $('#cutRollMsg').textContent = ''; // about the clips as they were
  // the selected clip left the sequence: Cut mode stays where it was, on the cut that took its place
  if (S.cutMode && !S.segments.some((s) => s.id === S.focus)) S.focus = S.segments[Math.min(S.cutAt, S.segments.length - 1)]?.id ?? S.focus;
  renderMoments();
  renderSequence();
  renderSave();
  renderExport();
  if (!current()) S.focus = cands()[0]?.id ?? S.segments[0]?.id ?? null;
  renderStage(false);
  // look_for's possible windows show whether they are in the sequence
  if (S.chat?.turns.some((t) => t.events.some((e) => e.possible?.length))) renderChat();
}

function changed() {
  renderSequence();
  renderSave();
  renderExport();
}

// ---------- undo ----------
// One stack for every edit of the working sequence, by hand or applied from AI Chat (#346): a
// snapshot of the segments, drafts and joins before each one. Ctrl+Z undoes, Ctrl+Y or Shift+Ctrl+Z redoes.
// An applied AI Chat change that sets Color keeps the Color it replaced too ({mode, strength}, as
// colorChoice gives it); other edits leave the Color alone, so undoing them never changes it.

const UNDO_MAX = 50;
const UNDO_JOIN_MS = 1000; // edits of one kind this close together (a held key, a drag) are one step
const snapshot = (withColor = false) => ({ segments: structuredClone(S.segments), drafts: structuredClone(S.drafts), joins: structuredClone(S.joins), ...(withColor ? { color: colorChoice() } : {}) });

/** Keep the sequence as it is before an edit (and the Color, with ``withColor``); ``tag`` joins a quick run of the same edit (one trim point of one clip) into one step. */
function remember(tag = null, withColor = false) {
  const now = performance.now();
  const last = S.undoTag;
  S.undoTag = tag ? { tag, at: now } : null;
  if (tag && last?.tag === tag && now - last.at < UNDO_JOIN_MS) return;
  S.undo.push(snapshot(withColor));
  if (S.undo.length > UNDO_MAX) S.undo.shift();
  S.redo = [];
  renderUndo();
}

/** Move one snapshot from ``from`` to the sequence, keeping the current one on ``to``. */
function restore(from, to, done) {
  if (!from.length) return;
  const s = from.pop();
  const wasCut = S.cutMode;
  to.push(snapshot(!!s.color));
  S.segments = s.segments;
  S.drafts = s.drafts;
  S.joins = s.joins;
  if (s.color) chooseColor(s.color);
  S.undoTag = null;
  renderAfterSeqChange();
  renderUndo();
  // renderStage said why Cut mode ended
  $('#sideLive').textContent = wasCut && !S.cutMode ? `${done}. ${$('#sideLive').textContent}` : done;
}
const undo = () => restore(S.undo, S.redo, 'Undone');
const redo = () => restore(S.redo, S.undo, 'Redone');

function renderUndo() {
  const had = document.activeElement; // read first: disabling a focused button drops its focus
  $('#undoBtn').disabled = !S.undo.length;
  $('#redoBtn').disabled = !S.redo.length;
  if (had?.disabled && (had.id === 'undoBtn' || had.id === 'redoBtn'))
    ([$('#undoBtn'), $('#redoBtn')].find((b) => !b.disabled) ?? seqHome()).focus();
}

/** Where focus goes when the control that had it is gone: Select all, or the Sequence tab (the heading is visually hidden, so it cannot show where focus is). */
const seqHome = () => ($('#seqSelAllRow').hidden ? $('#tabSequence') : $('#seqSelAll'));

// ---- dragging a clip (the Sequence list and the storyboard share this state) ----
let dragId = null;
let dragIn = null; // 'seq' or 'sb': the list the drag started in
const DRAG_TYPE = 'application/x-cutpoint-clip'; // what a drag of ours carries; a file or text dragged in does not
const DRAG_EDGE = 48; // px from the visible edge of the list where a held drag starts to scroll it
const DRAG_MAX = 10; // px per frame at the edge itself
let dragPush = null; // {el, horiz, v}: how the list scrolls now; v is signed px per frame
let dragRaf = 0;
let dragDirty = false; // a render was asked for during the drag and waits for its end
let dragOrder = ''; // the sequence's order when the drag began

/** A drag of one of our clips (``dragId`` set and our type on it), not a stale id or something dragged in. */
const ownDrag = (e) => !!dragId && Array.from(e.dataTransfer?.types ?? []).includes(DRAG_TYPE);

/** Forget the drag: the id, the scrolling and every marker; a render that waited for it runs now. Safe to call at any time. */
function endDrag() {
  const dirty = dragDirty;
  dragId = null;
  dragIn = null;
  dragPush = null;
  dragDirty = false;
  pressArmed = false; // a drag's end ends its press (not every browser sends the pointer's cancel)
  if (dragRaf) cancelAnimationFrame(dragRaf);
  dragRaf = 0;
  for (const el of document.querySelectorAll('.seq.dragging, .seq.drop, .sb-cell.dragging, .sb-cell.drop')) el.classList.remove('dragging', 'drop', 'after');
  if (dirty) renderSequence();
}

const dragOrderNow = () => S.segments.map((s) => s.id).join();

/**
 * A press on a row arms it (``draggable``) and the drag starts a moment later; a render in between would replace the row and
 * the drag would never begin. So an armed press holds renders like a drag does, until the pointer is released or cancelled
 * (a document-level listener: the release may land outside the row; a window blur ends it too, for a release that never
 * comes). The hold ends at once, so a render that follows the release (a double click's rename) runs unheld; only the
 * render that waited is deferred a task, as a drag that started holds renders itself (Edge sends dragstart before the cancel).
 */
let pressArmed = false;
function armPress() {
  if (pressArmed) return;
  pressArmed = true;
  dragOrder = dragOrderNow();
  const release = () => {
    document.removeEventListener('pointerup', release, true);
    document.removeEventListener('pointercancel', release, true);
    window.removeEventListener('blur', release);
    pressArmed = false;
    setTimeout(() => { // a render that waited; one that ran meanwhile has cleared it
      if (!dragId && dragDirty) {
        dragDirty = false;
        renderSequence();
      }
    }, 0);
  };
  document.addEventListener('pointerup', release, true);
  document.addEventListener('pointercancel', release, true);
  window.addEventListener('blur', release);
}
const seqDragStart = (id, kind, e, el) => {
  dragId = id;
  dragIn = kind;
  dragOrder = dragOrderNow();
  el.classList.add('dragging');
  e.dataTransfer.effectAllowed = 'move';
  e.dataTransfer.setData(DRAG_TYPE, id);
  e.dataTransfer.setData('text/plain', id);
};

/** Mark child ``i`` of ``els`` (a list's rows or blocks) as the drop place, on the side where the clip lands: a clip moved down lands after the target, one moved up before it. Markers of the other list go too. */
function markDrop(els, i) {
  const over = S.segments[i]?.id !== dragId;
  const after = S.segments.findIndex((x) => x.id === dragId) < i;
  for (const x of document.querySelectorAll('.seq.drop, .sb-cell.drop')) x.classList.remove('drop', 'after');
  const el = [...els][i];
  if (el && over) el.classList.add('drop', ...(after ? ['after'] : []));
}

/** A drag over (or entering) child ``i`` of ``els``: accept it if it is ours. Edge fires only dragenter when the element under the pointer changes, so both events come here. */
function dragHover(e, els, i) {
  if (!ownDrag(e)) return;
  e.preventDefault();
  markDrop(els, i);
}

/** A drop on child ``i``: move the held clip there, if it is ours. */
function dragDrop(e, i) {
  e.preventDefault();
  const id = ownDrag(e) ? dragId : null;
  endDrag();
  if (id) moveSeg(id, i);
}

/** The gap between a list's children belongs to the list: route its drag events to the child nearest the pointer. */
function gapDrag(list, kind, horiz) {
  const near = (e) => {
    if (e.target !== list || dragIn !== kind) return -1;
    const at = horiz ? e.clientX : e.clientY;
    let best = -1;
    let dist = Infinity;
    [...list.children].forEach((k, i) => {
      const r = k.getBoundingClientRect();
      const d = Math.abs(at - (horiz ? r.left + r.right : r.top + r.bottom) / 2);
      if (d < dist) [best, dist] = [i, d];
    });
    return best;
  };
  const hover = (e) => { const i = near(e); if (i >= 0) dragHover(e, list.children, i); };
  list.addEventListener('dragover', hover);
  list.addEventListener('dragenter', hover);
  list.addEventListener('drop', (e) => { const i = near(e); if (i >= 0) dragDrop(e, i); });
}

/** The list a held drag scrolls and the visible band of it: the column below its sticky bars and above the footer, or the page when that scrolls as a whole; the storyboard sideways. */
function dragArea() {
  if (dragIn === 'sb') {
    const el = $('#sbList');
    const r = el.getBoundingClientRect();
    return el.scrollWidth > el.clientWidth ? { el, horiz: true, lo: r.left, hi: r.right, a: r.top - 24, b: r.bottom + 24 } : null;
  }
  const col = $('#seqList').closest('.col');
  const cs = col && getComputedStyle(col);
  if (!col || cs.overflowY === 'visible') {
    const jobs = $('.jobs');
    return { el: null, lo: $('#panelSequence .col-head').offsetHeight, hi: innerHeight - (jobs?.offsetHeight || 0), a: 0, b: innerWidth };
  }
  const box = col.getBoundingClientRect();
  return {
    el: col, lo: box.top + (parseFloat(cs.scrollPaddingTop) || 0), a: box.left, b: box.right,
    hi: box.bottom - (parseFloat(cs.scrollPaddingBottom) || 0) - (parseFloat(cs.getPropertyValue('--seq-foot-h')) || 0), // above the sticky footer
  };
}

/** While a clip is held near the top or bottom of the list (left or right of the storyboard) scroll it, faster the nearer the edge. The browser's own scroll no longer reaches: the sticky bars and footer sit on the column's edges. */
function dragScroll(e) {
  if (!dragId) return;
  const area = dragArea();
  const p = area?.horiz ? e.clientX : e.clientY;
  const cross = area?.horiz ? e.clientY : e.clientX;
  const edge = area ? Math.min(DRAG_EDGE, (area.hi - area.lo) / 3) : 0;
  let v = 0;
  if (area && cross >= area.a && cross <= area.b) {
    const f = (d) => 1 + (DRAG_MAX - 1) * Math.min(1, d / edge) ** 2;
    if (p < area.lo + edge) v = -f(area.lo + edge - p);
    else if (p > area.hi - edge) v = f(p - (area.hi - edge));
  }
  dragPush = v ? { el: area.el, horiz: area.horiz, v } : null;
  if (dragPush && !dragRaf) dragRaf = requestAnimationFrame(dragStep);
}

function dragStep() {
  dragRaf = 0;
  if (!dragPush) return;
  const { el, horiz, v } = dragPush;
  if (!el) scrollBy(0, v);
  else if (horiz) el.scrollLeft += v;
  else el.scrollTop += v;
  dragRaf = requestAnimationFrame(dragStep);
}

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
  undo: [['path', { d: 'M12.5 8c-2.65 0-5.05.99-6.9 2.6L2 7v9h9l-3.62-3.62c1.39-1.16 3.16-1.88 5.12-1.88 3.54 0 6.55 2.31 7.6 5.5l2.37-.78C21.08 11.03 17.15 8 12.5 8z' }]],
  redo: [['path', { d: 'M18.4 10.6C16.55 8.99 14.15 8 11.5 8c-4.65 0-8.58 3.03-9.96 7.22L3.9 16c1.05-3.19 4.05-5.5 7.6-5.5 1.95 0 3.73.72 5.12 1.88L13 16h9V7l-3.6 3.6z' }]],
  play: [['path', { d: 'M8 5v14l11-7z' }]],
  pause: [['path', { d: 'M6 19h4V5H6v14zm8-14v14h4V5h-4z' }]],
  stop: [['path', { d: 'M6 6h12v12H6z' }]],
  check: [['path', { d: 'M9 16.17 4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z' }]],
  hourglass: [['path', { d: 'M6 2v6h.01L6 8.01 10 12l-4 4 .01.01H6V22h12v-5.99h-.01L18 16l-4-4 4-3.99-.01-.01H18V2H6zm10 14.5V20H8v-3.5l4-4 4 4zm-4-5-4-4V4h8v3.5l-4 4z' }]],
  funnel: [['path', { d: 'M4.25 5.61C6.27 8.2 10 13 10 13v6c0 .55.45 1 1 1h2c.55 0 1-.45 1-1v-6s3.72-4.8 5.74-7.39A1 1 0 0 0 18.95 4H5.04c-.83 0-1.3.95-.79 1.61z' }]],
  plus: [['path', { d: 'M19 13h-6v6h-2v-6H5v-2h6V5h2v6h6v2z' }]],
  minus: [['path', { d: 'M19 13H5v-2h14v2z' }]],
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

// One observer for every lazy still; each box maps to the function that asks for its still.
const lazyAsks = new WeakMap();
const lazySeen = typeof IntersectionObserver === 'function'
  ? new IntersectionObserver((entries) => {
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      lazySeen.unobserve(e.target);
      lazyAsks.get(e.target)?.();
      lazyAsks.delete(e.target);
    }
  })
  : null;

/**
 * A still of ``s`` in a box of class ``cls``. ``lazy`` (the Moments grid, which can hold hundreds):
 * a still not seen before is asked for once its box scrolls into view. The URL is the sequence
 * row's, so a moment and its row share one cached still: the server cuts it once, the first time
 * the tile scrolls into view, and serves it from its cache after that.
 */
function thumb(s, cls = 'seq-thumb', lazy = false) {
  const url = thumbUrl(s);
  const state = url ? thumbs.get(url) : 'failed';
  const box = h('span', { class: `${cls}${state === 'failed' ? ' failed' : state ? '' : ' loading'}` }, icon('film'));
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
      const ask = () => {
        thumbQueue.push(img);
        pumpThumbs();
      };
      if (lazy && lazySeen) {
        lazyAsks.set(box, () => {
          if (img.isConnected) ask();
        });
        lazySeen.observe(box);
      } else queueMicrotask(ask); // after the render has put the row into the page
    }
    box.append(img);
  }
  return box;
}

/** Why a label cannot be saved (mirrors ui/projects.py validate_segments: at most 200 printable characters), or ''. */
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

/** Ask with the shared dialog; resolves true when the user confirms. Focus returns to the trigger. ``danger``: the confirm button's warning look. */
function confirmAction(title, text, ok = 'Remove', danger = true) {
  const d = $('#confirmDialog');
  const back = document.activeElement;
  $('#confirmTitle').textContent = title;
  $('#confirmText').textContent = text;
  $('#confirmOk').textContent = ok;
  $('#confirmOk').classList.toggle('danger', danger);
  $('#confirmOk').classList.toggle('primary', !danger);
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
 * for 360 projects; "Crashes become moments at" (--must-include-crash) when the style has a default
 * (``must_include_crash_default``: every style but 360 projects). "Show every video to the vision
 * model" (--vision-detect-all) is on only while the analysis runs in ``style`` (the style button's,
 * else the project's) ``auto``.
 */
function fillAnalyzeOptions(p, style = p.style) {
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
  $('#anVisionDetectAll').checked = !!o.vision_detect_all;
  $('#anVisionDetectAll').disabled = style !== 'auto';
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
    // only Auto detects the footage: another style sends none (the analysis's own is kept)
    vision_detect_all: $('#anVisionDetectAll').disabled ? null : $('#anVisionDetectAll').checked,
    skip: [...document.querySelectorAll('input[name=anSkip]:checked')].map((box) => box.value),
  };
}

const AFTER_ANALYSIS = 'Available after the first analysis';

/**
 * "Describe the footage" (``crashes --context``) and "Look closer with vision" (``events``) in the Analyze
 * dialog, from the project's ``refine``. Both wait for an analysis. Describe shows only where the analysis
 * runs the crash step (``crash_step``, from the style before the first analysis; 360 projects get a note), is filled with the description ``activity.json`` has, and
 * Applies only with text (an empty one keeps the profile). Each is off, with its reason, where the events
 * extra is missing (Cutpoint lite).
 */
function fillRefine(p) {
  const r = p.refine ?? {};
  const analyzed = !!p.highlights;
  const pano = p.kind === '360';
  const text = $('#anDescribe');
  const apply = $('#anDescribeBtn');
  const why = (id, msg) => { $(id).textContent = msg; };
  $('#describeBox').hidden = !pano && !r.crash_step;
  apply.onclick = () => $('#analyzeDialog').close('describe');
  $('#anVisionBtn').onclick = () => $('#analyzeDialog').close('vision');
  for (const el of [text, apply, $('#dAnDescribe'), $('#describeBox label')]) el.hidden = pano; // only the note stays
  text.value = r.context ?? '';
  const off = pano ? "Describe the footage doesn't apply to 360 footage." : !analyzed ? AFTER_ANALYSIS : r.unavailable ?? '';
  why('#anDescribeWhy', off);
  text.disabled = !!off;
  apply.disabled = !!off || !text.value.trim();
  text.oninput = () => { apply.disabled = !!off || !text.value.trim(); };
  const vision = $('#anVisionBtn');
  const noVision = pano ? 'Vision rating is not used for 360 footage.' : !analyzed ? AFTER_ANALYSIS : r.unavailable ?? '';
  why('#anVisionWhy', noVision);
  vision.disabled = !!noVision;
}

/**
 * "Analyze again" (``fresh``: the first analysis, from the empty state's Analysis options):
 * confirm in a dialog with the Analyze options; vision follows Settings, as for a first analysis.
 * ``to`` (the style button): ``{style, name}``, the style to analyze in from now on ('auto': detected),
 * sent with the options; Describe the footage and Look closer with vision are not offered then.
 */
function analyzeAgain(fresh = false, to = null) {
  const p = S.project;
  if (!p) return;
  const d = $('#analyzeDialog');
  const back = document.activeElement;
  $('#analyzeTitle').textContent = fresh ? 'Analyze footage' : to ? `Analyze again as ${to.name}?` : 'Analyze again?';
  $('#analyzeText').textContent = fresh
    ? 'Analyze footage to transcribe audio and rank the strongest moments.'
    : to
      ? `The project's footage is analyzed again ${to.style === 'auto' ? 'in the style detected from it' : `in the ${to.style} style`} and its moments are ranked anew. Transcripts are reused; the sequence, clips and videos stay.`
      : "The project's footage is analyzed again and its moments are ranked anew.";
  fillAnalyzeOptions(p, to?.style ?? p.style);
  fillRefine(p);
  $('#refine').hidden = !!to;
  if (fresh) $('#analyzeOpts').open = true;
  d.returnValue = '';
  d.showModal();
  $('#againCancel').focus();
  d.addEventListener('close', () => {
    const answer = d.returnValue;
    if (back?.isConnected) back.focus();
    if (fresh && answer === 'ok') keepEmptyFocus($('#momentsEmpty')); // after focus is back, or it ends at once
    // 360 projects have no Analyze options (360 analyze takes none of them)
    if (answer === 'ok') startJob('analyze', p.kind === '360' ? {} : { ...analyzeOptions(), ...(to && { style: to.style }) });
    // each of these is its own CLI command, not an analysis: crashes --context, events
    else if (answer === 'describe') startJob('crashes', { context: $('#anDescribe').value.trim() });
    else if (answer === 'vision') startJob('events');
  }, { once: true });
}

/** Focus control ``act`` of the row now at ``i`` (the last row if fewer remain). */
function focusRow(i, act) {
  const rows = $('#seqList').children;
  const li = rows[Math.min(i, rows.length - 1)];
  (li?.querySelector(`[data-act="${act}"]:not(:disabled)`) ?? li?.querySelector('button') ?? seqHome()).focus();
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
  const hidePlay = !!n && !S.seqPlay;
  if (hidePlay && document.activeElement === $('#seqPlayBtn')) $('#seqClearSel').focus(); // a hidden button must not keep the focus
  $('#seqPlayBtn').hidden = hidePlay; // Play sequence, or the selection's actions, in the same place; Stop stays
  $('#seqSelCount').replaceChildren(String(n), h('span', { class: 'seq-sel-word', text: ' selected' }));
  const all = $('#seqSelAll');
  $('#seqSelAllRow').hidden = !S.segments.length;
  all.checked = n > 0 && n === S.segments.length;
  all.indeterminate = n > 0 && n < S.segments.length;
  for (const li of $('#seqList').children) {
    const on = S.selected.has(li.dataset.id);
    li.classList.toggle('selected', on);
    const box = li.querySelector('[data-act="select"]');
    if (box) box.checked = on;
  }
}

/** Tick every clip of the sequence, or clear all ticks when every clip is already ticked. */
function selectAll() {
  const n = S.segments.length;
  if (!n) return;
  const on = S.selected.size < n;
  S.selected.clear();
  if (on) for (const s of S.segments) S.selected.add(s.id);
  renderBatch();
  $('#seqLive').textContent = on ? `All ${n} clip${n > 1 ? 's' : ''} selected` : 'Selection cleared';
}

/** Accessible names of a row's controls that carry the clip name. */
const ROW_LABELS = {
  select: (name) => `Select ${name}`,
  remove: (name) => `Remove ${name} from sequence`,
  up: (name) => `Move ${name} up`,
  down: (name) => `Move ${name} down`,
};

const seqDur = (s, range) => {
  // a tail (or a speech-aware end) runs the export past the cut: say so under the exported length
  const tail = planned(s).end - s.end > 0.05;
  return h('span', { class: 'seq-dur tc', text: `${exportedLen(s).toFixed(1)} s` },
    h('span', { class: 'vh', text: `, ${range}` }),
    tail ? h('span', { class: 'seq-tail', text: `cut ${(s.end - s.start).toFixed(1)} s + tail` }) : null,
    // a clip that is ready says nothing; one that is not gets the icon-only badge of the Moments tiles
    prepState(srcOf(s.source), s) === 'ready' ? null : prepBadge(srcOf(s.source), s, true));
};

/** The kind line under a row's title: the kind of its ranked moment; none for a clip that is not one. */
const seqKind = (s) => {
  const kind = kindOf(cands().find((c) => c.id === s.id));
  return kind ? h('span', { class: 'seq-kind', text: KINDS[kind] }) : null;
};

/** A row's title button (``head``: its rank and thumbnail) and rename pencil, when not renaming. */
function seqMain(s, head, name = s.label || s.id, range = `${fmt(s.start)} to ${fmt(s.end)}`) {
  const main = h('button', { type: 'button', class: 'seq-main', 'data-act': 'focus', title: range, onclick: () => focusMoment(s.id) },
    ...head,
    h('span', { class: 'seq-text' },
      h('span', { class: 'seq-title', text: name, title: name, ondblclick: () => startRename(s.id) }), seqKind(s), seqDur(s, range)));
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
    class: `seq${S.selected.has(s.id) ? ' selected' : ''}${seqNow() === s.id ? ' playing' : ''}`, 'data-id': s.id,
    'data-kind': kindOf(cands().find((c) => c.id === s.id)), 'aria-current': s.id === S.focus ? 'true' : null,
    // the whole row drags; a press on the checkbox or the row's buttons does not arm it, nor does renaming
    onpointerdown: (e) => {
      li.draggable = S.renaming !== s.id && !e.target.closest('input, .seq-ctl');
      if (li.draggable && e.button === 0) armPress(); // the primary button only: another one may never send its release
    },
    onpointerup: () => { if (!dragId) li.draggable = false; },
    onpointercancel: () => { if (!dragId) li.draggable = false; },
    ondragstart: (e) => seqDragStart(s.id, 'seq', e, li),
    ondragend: () => {
      endDrag();
      li.draggable = false;
    },
    ondragenter: (e) => dragHover(e, $('#seqList').children, i),
    ondragover: (e) => dragHover(e, $('#seqList').children, i),
    ondrop: (e) => dragDrop(e, i),
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
          h('button', { type: 'button', class: 'seq-chev', 'data-act': 'down', disabled: i === n - 1, 'aria-label': ROW_LABELS.down(name), onclick: () => moveSeg(s.id, i + 1) }, icon('down'))))),
  );
  return li;
}

/**
 * The line under the list: the sequence's exported length in mono and, when a target length is
 * set, "of ~4:00" with a thin bar of how much of it the sequence fills (full and capped at
 * the bar's end when over). Without a target: the total alone.
 */
function renderSeqFoot() {
  const segs = S.segments;
  const total = segs.reduce((n, s) => n + exportedLen(s), 0);
  const target = seqTarget();
  const has = target != null;
  const editing = !$('#seqTargetForm').hidden;
  $('#seqFoot').hidden = !segs.length;
  const btn = $('#seqTotal');
  btn.textContent = fmtLen(total);
  const over = has && Math.round(total) > Math.round(target);
  btn.setAttribute('aria-label', has
    ? `${fmtLen(total)} of about ${fmtLen(target)}${over ? `, ${fmtLen(total - target)} over` : ''}. Change the target length`
    : `${fmtLen(total)}. Set a target length`);
  btn.title = has ? 'Change the target length: a guide for the bar, it cuts nothing' : 'Set a target length: a guide, it cuts nothing';
  $('#seqTarget').textContent = seqTargetText(total);
  $('#seqTarget').hidden = editing;
  $('#seqBar').hidden = !has || editing;
  $('#seqBarFill').style.width = `${has ? Math.min(100, (total / target) * 100) : 0}%`; // over the target: a full bar, never wider
  if (segs.length) $('#sbTotal').textContent = sbTotalText(total); // the target may have changed alone
}

/** "of ~4:00" (", 0:12 over" when ``total`` passes it) for the footer and the storyboard's total; empty without a target. */
function seqTargetText(total) {
  const target = seqTarget();
  if (target == null) return '';
  return `of ~${fmtLen(target)}${Math.round(total) > Math.round(target) ? `, ${fmtLen(total - target)} over` : ''}`;
}

/** The sequence's target length in seconds (a guide for the footer's bar, saved with the project in ui/review.json), or null. */
const seqTarget = () => S.target;

/** The project's saved target (``review.target``); the editor closes. Called when a project opens. */
function startTarget() {
  const v = S.project?.review?.target;
  S.target = Number.isFinite(v) && v > 0 ? v : null;
  $('#seqTargetForm').hidden = true;
  $('#seqTotal').setAttribute('aria-expanded', 'false');
  renderSeqFoot();
}

/** Open (``on``) or close the footer's target field; focus goes to the field, or back to the total. */
function editSeqTarget(on) {
  $('#seqTargetForm').hidden = !on;
  $('#seqTotal').setAttribute('aria-expanded', String(on));
  $('#seqTargetErr').textContent = '';
  $('#seqTargetInput').removeAttribute('aria-invalid');
  renderSeqFoot();
  if (on) {
    const input = $('#seqTargetInput');
    input.value = S.target == null ? '' : fmtLen(S.target);
    input.focus();
    input.select();
  } else {
    $('#seqTotal').focus();
  }
}

/** Set the target from the field: M:SS or seconds; empty clears it; anything else is refused with a message. */
async function commitSeqTarget() {
  const text = $('#seqTargetInput').value.trim();
  const secs = text ? parseTime(text) : null;
  if (text && !(secs > 0 && secs <= 36000)) {
    $('#seqTargetErr').textContent = 'Enter a length such as 4:00 or 240 seconds, up to 10 hours.';
    $('#seqTargetInput').setAttribute('aria-invalid', 'true');
    $('#seqTargetInput').focus();
    return;
  }
  $('#seqTargetInput').removeAttribute('aria-invalid');
  // saved with the project (ui/review.json) by its own request: not part of the sequence, so it never marks it unsaved
  const id = S.id;
  try {
    await api(`/api/projects/${encodeURIComponent(id)}/target`, { method: 'PUT', body: { seconds: secs } });
  } catch (e) {
    // not saved: the old target stays and the field stays open with the reason
    if (S.id !== id) return;
    $('#seqTargetErr').textContent = `Could not save the target length: ${e.message || e}`;
    $('#seqTargetInput').focus();
    return;
  }
  if (S.id !== id) return; // another project opened meanwhile
  S.target = secs;
  if ($('#seqTargetForm').hidden) renderSeqFoot(); // Cancel came first: the form is already closed
  else editSeqTarget(false);
  $('#seqLive').textContent = secs == null ? 'Target length removed' : `Target length ${fmtLen(secs)}`;
}

function renderSequence() {
  // A held clip's row must survive, or its dragend never comes (the background polls and the edges reply render too):
  // wait for the drag's end. If the order changed underneath, the drag is over: render now.
  if (dragId || pressArmed) {
    if (dragOrderNow() === dragOrder) {
      dragDirty = true;
      return;
    }
    dragDirty = false; // this render is the one the drag waited for
    endDrag();
  }
  dragDirty = false; // this render runs: nothing waits
  const segs = S.segments;
  renderSeqFoot();
  if (!S.project) setEmpty($('#seqEmpty'));
  else if (!segs.length) setEmpty($('#seqEmpty'), h('p', {}, 'Nothing in the sequence yet. Add moments from the list, then trim and reorder them here. Clips are exported in this order.'));
  else setEmpty($('#seqEmpty'));
  if (S.renaming && !segs.some((s) => s.id === S.renaming)) S.renaming = null;
  if (S.seqPlay && S.seqPlay.i >= 0 && !seqSeg(S.seqPlay.i)) seqStop('The playing clip left the sequence. Sequence stopped.');
  if (S.seqPlay?.across) {
    const k = S.segments.findIndex((s) => s.id === S.seqPlay.order[0]);
    if (k < 0 || S.segments[k + 1]?.id !== S.seqPlay.order[1]) seqStop('These clips are no longer neighbours. Stopped.');
  }
  preserveFocus(() => {
    $('#seqList').replaceChildren(...segs.map((s, i) => seqRow(s, i, segs.length)));
  });
  renderBatch();
  renderRemoved();
  renderSeqPlay();
  renderStoryboard();
  renderProposal();
  askEdges();
}

// ---------- storyboard ----------

/**
 * The sequence as one block per clip under the trim controls: width in proportion to the exported
 * length, colour by kind. Click selects the clip for trimming, drag (or Alt+Left/Right) reorders
 * through moveSeg, double-click (or Shift+Enter) plays the sequence from that clip. Arrow keys
 * move between the blocks (one tab stop).
 */
function renderStoryboard() {
  const segs = S.segments;
  $('#storyboard').hidden = !segs.length;
  const total = sbLength();
  $('#sbTotal').textContent = segs.length ? sbTotalText(total) : '';
  const stop = segs.some((s) => s.id === S.focus) ? S.focus : segs[0]?.id; // the block that holds the tab stop
  const had = $('#sbList').contains(document.activeElement) ? document.activeElement.dataset.id : null;
  $('#sbList').replaceChildren(...segs.map((s, i) => sbBlock(s, i, segs.length, stop)));
  if (had) $(`#sbList .sb-block[data-id="${CSS.escape(had)}"]`)?.focus();
  markPlaying();
  drawSbRuler();
  drawSbHead();
  renderClipHead(); // a clip's place in the sequence may have changed
}

/** "26 clips · 3:28", and "of ~4:00" after it when a target is set (the same words as the footer). */
const sbTotalText = (total) => `${S.segments.length} clip${S.segments.length > 1 ? 's' : ''} · ${fmtLen(total)}${seqTarget() == null ? '' : ` ${seqTargetText(total)}`}`;

/** The storyboard's buttons, in order. */
const sbBlocks = () => [...$('#sbList').querySelectorAll('.sb-block')];
const sbLength = () => S.segments.reduce((n, s) => n + exportedLen(s), 0);
const SB_RULER_GAP = 120; // pixels: the least room between two labels of the storyboard's ruler
const SB_RULER_END = 84; // pixels: a label this close to the total at the right end is left out

/** The sequence's own time above the blocks, from 0:00 to its total; off while the blocks scroll, when it could not line up with them. */
function drawSbRuler() {
  const el = $('#sbRuler');
  const list = $('#sbList');
  const total = sbLength();
  const width = list.clientWidth || 600;
  el.hidden = !total || list.scrollWidth > width + 1;
  if (el.hidden) return;
  const step = RULER_STEPS.find((x) => (x / total) * width >= SB_RULER_GAP) ?? RULER_STEPS.at(-1);
  const ticks = [];
  for (let t = 0; ((total - t) / total) * width >= SB_RULER_END; t += step) {
    ticks.push(h('span', { style: `left: ${((t / total) * 100).toFixed(2)}%`, text: fmtLen(t) }));
  }
  el.replaceChildren(...ticks, h('span', { class: 'end', text: fmtLen(total) }));
}

/** The storyboard's playhead: in the open clip's block, as far along as the playhead is in its export; hidden for a clip outside the sequence or scrolled away. */
function drawSbHead(c = current(), t = playhead(c)) {
  const el = $('#sbHead');
  const cell = c?.seg && sbBlocks().find((b) => b.dataset.id === c.id)?.parentElement;
  el.hidden = !cell;
  if (!cell) return;
  const e = planned(c);
  const list = $('#sbList');
  const span = e.end - e.start; // 0 for a clip the export drops
  const f = span > 1e-6 ? Math.max(0, Math.min(1, (t - e.start) / span)) : 0;
  const x = cell.offsetLeft - list.scrollLeft + f * cell.offsetWidth;
  el.style.left = `${x.toFixed(1)}px`;
  el.hidden = x < 0 || x > list.clientWidth;
}
new ResizeObserver(() => {
  drawSbRuler();
  drawSbHead();
}).observe($('#sbList'));
$('#sbList').addEventListener('scroll', () => drawSbHead());

function sbBlock(s, i, n, stop) {
  const cand = cands().find((c) => c.id === s.id);
  const name = shownName(cand, s.label) || s.id;
  const len = exportedLen(s);
  const kind = kindOf(cand);
  const cells = () => [...$('#sbList').children];
  const b = h('button', {
    type: 'button', class: `sb-block${seqNow() === s.id ? ' playing' : ''}`, 'data-id': s.id, 'data-kind': kind,
    tabindex: s.id === stop ? 0 : -1, title: `${i + 1}. ${name}, ${len.toFixed(1)} s`, 'aria-current': s.id === S.focus ? 'true' : null,
    'aria-label': `Clip ${i + 1} of ${n}: ${name}, ${len.toFixed(1)} s`,
    onclick: () => focusMoment(s.id),
    ondblclick: () => seqStart(s.id),
    onkeydown: (e) => {
      if (e.ctrlKey || e.metaKey) return;
      if (e.key === 'Enter' && e.shiftKey) {
        e.preventDefault();
        return seqStart(s.id);
      }
      const step = { ArrowLeft: -1, ArrowRight: 1 }[e.key];
      const to = step != null ? i + step : e.key === 'Home' ? 0 : e.key === 'End' ? n - 1 : null;
      if (to == null || e.shiftKey) return;
      e.preventDefault();
      if (e.altKey) return moveSeg(s.id, to); // re-rendered: the block keeps focus (see renderStoryboard)
      const next = sbBlocks()[Math.max(0, Math.min(n - 1, to))];
      for (const el of sbBlocks()) el.tabIndex = el === next ? 0 : -1;
      next.focus();
    },
  }, h('span', { class: 'sb-n tc', 'aria-hidden': 'true', text: String(i + 1) }));
  // the cell is what drags (a draggable <button> does not drag in Firefox)
  const cell = h('div', {
    class: 'sb-cell', style: `flex-grow: ${Math.max(len, 0.1).toFixed(2)}`, draggable: true,
    ondragstart: (e) => seqDragStart(s.id, 'sb', e, cell),
    ondragend: () => endDrag(),
    ondragenter: (e) => dragHover(e, cells(), i),
    ondragover: (e) => dragHover(e, cells(), i),
    ondrop: (e) => dragDrop(e, i),
  }, b);
  return cell;
}

// ---------- cut mode ----------

const round3 = (t) => Math.round(t * 1000) / 1000; // segments.py keeps times to the millisecond
let cutStillTimer = 0;
// a roll under the pointer: {id, pid, a, b, x0, x, end0, out, into, moved, kept, raf}; a, b: the
// clips' ids; end0, out, into: clip 1's trim end and the shown out and in when it started
let cutDrag = null;

/** The index of the current cut's first clip: the selected clip, or the one before it when that is the last clip; -1 when there is no cut. While a cut plays, that cut. */
function cutIndex() {
  const id = S.seqPlay?.across ? S.seqPlay.order[0] : S.focus;
  const i = S.segments.findIndex((s) => s.id === id);
  return S.segments.length < 2 || i < 0 ? -1 : Math.min(i, S.segments.length - 2);
}

const cutName = (s) => shownName(cands().find((c) => c.id === s.id), s.label) || s.id;

/** Enter or leave Cut mode: only the stage's layout, the caller renders. The mode switch sits in the title row, as the player's bar is hidden. */
function setCut(on) {
  if (cutDrag && !on) dropCutDrag();
  S.cutMode = on;
  $('#cutRollMsg').textContent = '';
  $('#stageBody').classList.toggle('cut-mode', on);
  $('#cutView').hidden = !on;
  const had = document.activeElement;
  (on ? $('.stage-head') : $('.transport')).append($('#stageMode'));
  if (had?.closest?.('#stageMode')) had.focus(); // moving an element drops its focus
  renderSeqPlay();
  if (!on) precueCut(); // releases a spare opened for a cut
}

function enterCut() {
  if (S.cutMode) return;
  if (S.segments.length < 2) return setTrimMsg(CUT_NEEDS, false, true); // on screen: a title is out of a touch user's reach
  seqStop('');
  cancelResume(); // a resume at a window's end would start the hidden player again
  v.pause();
  setCut(true);
  if (S.segments.some((s) => s.id === S.focus)) renderStage(false);
  else showMoment(S.segments[0].id); // a moment outside the sequence has no cut
}

/** Back to the selected clip, at its start. */
function leaveCut() {
  seqStop('');
  setCut(false);
  renderStage(true);
}

function renderCutHead() {
  const k = cutIndex();
  const [a, b] = [S.segments[k], S.segments[k + 1]];
  $('#stageTitle').textContent = `Cut ${k + 1} → ${k + 2}`;
  $('#stageTitle').title = '';
  $('#stageTitle').setAttribute('aria-label', `Cut ${k + 1} to ${k + 2}`);
  const meta = $('#stageMeta');
  meta.textContent = `${cutName(a)} into ${cutName(b)}`;
  meta.title = meta.textContent;
}

/** The whole Cut view; every value is read from the sequence and the export plan. */
function renderCut() {
  const k = cutIndex();
  if (!S.cutMode || k < 0) return;
  if (k !== S.cutAt) $('#cutRollMsg').textContent = ''; // about the cut before
  S.cutAt = k;
  if (cutDrag && (S.id !== cutDrag.pid || S.segments[k].id !== cutDrag.a || S.segments[k + 1].id !== cutDrag.b)) dropCutDrag();
  renderCutHead();
  const [a, b] = [S.segments[k], S.segments[k + 1]];
  const out = planned(a).end;
  const into = planned(b).start;
  $('#cutOutName').textContent = `${k + 1} · last frame`;
  $('#cutInName').textContent = `${k + 2} · first frame`;
  $('#cutOutTime').textContent = `out ${tenths(out)}${out - a.end > 0.05 ? ' with tail' : ''}`;
  $('#cutInTime').textContent = `in ${tenths(into)}`;
  $('#cutOutFrame').alt = `Last frame of clip ${k + 1}: ${cutName(a)}`;
  $('#cutInFrame').alt = `First frame of clip ${k + 2}: ${cutName(b)}`;
  clearTimeout(cutStillTimer);
  cutStillTimer = setTimeout(() => { // a roll asks once it rests: each still is cut from the source
    cutStill($('#cutOutFrame'), a, Math.max(planned(a).start, planned(a).end - FRAME));
    cutStill($('#cutInFrame'), b, planned(b).start);
  }, FILM_WAIT);
  renderCutTransition();
  drawCutRoll();
  $('#cutRollHint').textContent = `Drag the join: clip ${k + 1}'s out and clip ${k + 2}'s in move together and the length stays ${fmtLen(sbLength())}.`;
  renderCutStrip(k);
  const nav = [$('#prevCut'), $('#nextCut')];
  const had = nav.find((x) => x === document.activeElement); // read first: disabling drops focus
  nav[0].disabled = k <= 0;
  nav[1].disabled = k >= S.segments.length - 2;
  if (had?.disabled) (nav.find((x) => !x.disabled) ?? $('#cutJoin')).focus();
  precueCut();
}

/** Show the still of ``s`` at ``t`` in ``img``; the last one stays, dimmed, until it has loaded. */
function cutStill(img, s, t) {
  const src = srcOf(s.source);
  const [path, query] = src?.url?.split('?') ?? []; // query carries the session token, as for <video>
  const url = path ? `${path}/frame?${query}&at=${round2(t)}&w=${CUT_STILL_W}` : '';
  if (img.dataset.url === url) return;
  img.dataset.url = url;
  const frame = img.closest('.cut-frame');
  frame.classList.toggle('failed', !url);
  frame.classList.toggle('loading', !!url); // the times are new at once, the 640 px still takes a while: dim the old picture
  if (url) img.src = url; // bindCut's load and error listeners clear "loading"
  else img.removeAttribute('src');
}

/** The transition control shows the Export panel's Video transition: the default of every join with no choice of its own. */
function renderCutTransition() {
  const on = $('#optTransition').value;
  for (const b of $('#cutTransition').children) b.setAttribute('aria-pressed', String(b.dataset.value === on));
}

/** The choice of clip ``a`` into clip ``b``, or undefined for Default. */
const joinOf = (a, b) => S.joins.find((j) => j.a === a.id && j.b === b.id)?.transition;
/** What clip ``a`` into clip ``b`` plays and exports: its own choice, or the transition of the whole video; always a Cut next to a word splice (assemble cuts there). */
const joinTransition = (a, b) => (a.exact || b.exact ? 'cut' : joinOf(a, b) ?? $('#optTransition').value);
/** The Fade length as Assemble uses it: the field, or its default when blank or out of range. */
const fadeLen = () => ($('#optFadeDur').value.trim() && $('#optFadeDur').checkValidity() ? Number($('#optFadeDur').value) : FADE_DUR);
/** Seconds clip ``a`` and clip ``b`` overlap: 0 for a Cut. All Fades share one length, as in assemble: at most a share of the shortest clip they join. */
function joinLap(a, b) {
  if (joinTransition(a, b) !== 'fade') return 0;
  const fading = S.segments.slice(1).flatMap((y, i) => (joinTransition(S.segments[i], y) === 'fade' ? [exportedLen(S.segments[i]), exportedLen(y)] : []));
  return Math.min(fadeLen(), FADE_ROOM * Math.min(...fading));
}

/** Set the join of clip ``a`` into clip ``b`` (index ``i``) to ``transition``, or to Default with null: one undo step, saved with the sequence. */
function setJoin(a, b, i, transition) {
  if (S.seqPlay?.across) seqStop();
  remember();
  S.joins = S.joins.filter((j) => j.a !== a.id || j.b !== b.id);
  if (transition) S.joins.push({ a: a.id, b: b.id, transition });
  changed();
  renderCut();
  $('#sideLive').textContent = `Transition of cut ${i + 1} set to ${transition ? JOIN_NAMES[transition] : `Default (${JOIN_NAMES[joinTransition(a, b)]})`}${joinsLive()}`;
}

/** How far a roll can go from where it is, and why not further: the limits of a trim (validate, handleRange) on clip 1's end and clip 2's start. */
function rollRange(k) {
  const [a, b] = [S.segments[k], S.segments[k + 1]];
  const shortA = a.start + MIN_CLIP - a.end;
  const srcEnd = handleRange('end', a).max - a.end; // Infinity while the source length is unknown
  const shortB = b.end - MIN_CLIP - b.start;
  return {
    min: Math.min(0, Math.max(shortA, -b.start)),
    max: Math.max(0, Math.min(srcEnd, shortB)),
    whyMin: shortA > -b.start ? `Clip ${k + 1} is at its shortest, ${MIN_CLIP} s.` : `Clip ${k + 2} is at the start of its source.`,
    whyMax: srcEnd < shortB ? `Clip ${k + 1} is at the end of its source.` : `Clip ${k + 2} is at its shortest, ${MIN_CLIP} s.`,
  };
}

/** The join and the waveforms: each clip's audio around the cut, held still under a drag while the join moves over them. */
function drawCutRoll() {
  const k = cutIndex();
  if (!S.cutMode || k < 0) return;
  const [a, b] = [S.segments[k], S.segments[k + 1]];
  const d = cutDrag;
  const out = d ? d.out : planned(a).end;
  const into = d ? d.into : planned(b).start;
  const track = $('#cutTrack');
  const width = track.clientWidth || 600;
  const off = d ? (a.end - d.end0) / (2 * CUT_PLAY) : 0;
  track.style.setProperty('--join', `${(50 + off * 100).toFixed(2)}%`);
  [[$('#cutSideA'), a, out, k + 1], [$('#cutSideB'), b, into, k + 2]].forEach(([el, s, t, n]) => {
    el.dataset.kind = kindOf(cands().find((c) => c.id === s.id)) ?? '';
    el.querySelector('.cut-side-name').textContent = `${n} · ${cutName(s)}`;
    paintWave(el.querySelector('.cut-wave'), envelopeOf(s), s.source, t - CUT_PLAY, t + CUT_PLAY, width);
  });
  const r = rollRange(k);
  const join = $('#cutJoin');
  join.setAttribute('aria-valuemin', round3(a.end + r.min));
  join.setAttribute('aria-valuemax', round3(a.end + r.max));
  join.setAttribute('aria-valuenow', a.end);
  join.setAttribute('aria-valuetext', `Clip ${k + 1} out ${fmt(planned(a).end)}, clip ${k + 2} in ${fmt(planned(b).start)}`);
}
new ResizeObserver(() => drawCutRoll()).observe($('#cutTrack'));

/** Move what the export plans for ``s`` along with its point ``which``, so tails and totals hold until the server's new plan comes in. */
function shiftPlan(s, which, t) {
  if (planned(s) === s) return;
  const e = exportPlan.map[s.id];
  e[which] = round3(e[which] + t - s[which]);
  e.from = which === 'start' ? [t, s.end] : [s.start, t];
}

/** The note for a roll that ran into a third clip of the same source: allowed, but the video then shows that footage twice. */
function rollRepeat(k) {
  return [k, k + 1].map((i) => {
    const s = S.segments[i];
    const hits = S.segments.flatMap((o, n) => (n !== k && n !== k + 1 && o.source === s.source && s.start < o.end && o.start < s.end ? [n + 1] : []));
    if (!hits.length) return '';
    return `Clip ${i + 1} repeats part of ${hits.length > 1 ? `clips ${hits.slice(0, -1).join(', ')} and ${hits.at(-1)}` : `clip ${hits[0]}`}.`;
  }).filter(Boolean).join(' ');
}

/**
 * Roll cut ``k`` by ``by`` seconds: clip 1's end and clip 2's start move the same amount, inside
 * both clips' limits, so the two lengths still add up. ``tag`` goes to remember (false: this
 * edit's step is already kept, a drag). True when something moved.
 */
function rollCut(k, by, tag) {
  const [a, b] = [S.segments[k], S.segments[k + 1]];
  const r = rollRange(k);
  const d = Math.max(r.min, Math.min(r.max, by));
  const say = (text) => {
    if ($('#cutRollMsg').textContent !== text) $('#cutRollMsg').textContent = text;
  };
  const limit = d === by ? '' : by < 0 ? r.whyMin : r.whyMax;
  const note = () => say([limit, rollRepeat(k)].filter(Boolean).join(' '));
  const out = round3(a.end + d);
  const into = round3(b.start + d);
  if (out === a.end && into === b.start) {
    note();
    return false;
  }
  const err = validate(a.start, out, durOf(a)) || validate(into, b.end, durOf(b));
  if (err) {
    say(err);
    return false;
  }
  if (tag !== false) remember(tag);
  shiftPlan(a, 'end', out);
  shiftPlan(b, 'start', into);
  a.end = out;
  b.start = into;
  changed();
  renderCut();
  note();
  return true;
}

function dropCutDrag() {
  const d = cutDrag;
  cancelAnimationFrame(d.raf);
  cutDrag = null;
  $('#cutJoin').classList.remove('dragging');
}

/** One drag step: the join follows the pointer, up to the track's ends; the first move keeps the undo step. */
function cutDragFrame() {
  const d = cutDrag;
  d.raf = 0;
  const k = cutIndex();
  const a = S.segments[k];
  if (!a || a.id !== d.a || S.segments[k + 1].id !== d.b || S.id !== d.pid) return dropCutDrag();
  const moved = ((d.x - d.x0) / ($('#cutTrack').clientWidth || 600)) * 2 * CUT_PLAY;
  const by = round3(d.end0 + round2(Math.max(-CUT_PLAY, Math.min(CUT_PLAY, moved))) - a.end);
  if (rollCut(k, by, d.kept ? false : null)) d.kept = true;
}

function stepCut(d) {
  const k = cutIndex() + d;
  if (k >= 0 && k <= S.segments.length - 2) focusMoment(S.segments[k].id);
}

/** Play the last seconds of clip 1 and the first of clip 2 through "Play sequence": the spare player holds clip 2, so the step is a swap. */
function cutPlay() {
  const k = cutIndex();
  if (k < 0) return;
  $('#cutRollMsg').textContent = ''; // it may say the last play could not show the Fade
  seqStart(S.segments[k].id, [S.segments[k].id, S.segments[k + 1].id], { back: S.focus });
}

/** The play button and the player's place: over the frames while a cut plays. */
function renderCutPlay() {
  const on = !!S.seqPlay?.across;
  const opens = on && !$('#stageBody').classList.contains('cut-playing');
  $('#stageBody').classList.toggle('cut-playing', on);
  if (opens) $('.screen').scrollIntoView({ block: 'nearest' }); // on a scrolled narrow page it opens above the view
  const b = $('#cutPlayBtn');
  b.replaceChildren(icon(on ? 'stop' : 'play'), on ? 'Stop' : 'Play across the cut');
  b.title = on ? 'Stop (Escape)' : 'Play across the cut (Space)';
  b.setAttribute('aria-keyshortcuts', on ? 'Escape' : 'Space');
}

/** The button of the join of clip ``a`` into clip ``b`` (cut ``i`` of ``n - 1``): the letter of what it plays, solid when it is the join's own choice, dashed for Default. Next to a word splice it is always a cut and cannot be changed. */
function joinButton(a, b, i, n, k, onkeydown) {
  if (a.exact || b.exact) {
    const why = `Cut ${i + 1}: always a cut next to a word splice`;
    return h('button', {
      type: 'button', class: 'cut-mark', 'data-join': i, 'data-own': 'false', 'aria-disabled': 'true', tabindex: i === k ? 0 : -1,
      'aria-label': why, title: why, onkeydown,
    }, JOIN_NAMES.cut[0]);
  }
  const own = joinOf(a, b);
  const on = JOIN_NAMES[joinTransition(a, b)];
  const next = JOIN_NEXT[own ?? 'default'];
  const name = next ? JOIN_NAMES[next] : 'Default';
  return h('button', {
    type: 'button', class: 'cut-mark', 'data-join': i, 'data-own': own ? 'true' : 'false', tabindex: i === k ? 0 : -1,
    'aria-label': `Transition of cut ${i + 1} of ${n - 1}: ${own ? `${on}, set for this cut` : `Default, ${on}`}. Press for ${name}`,
    title: `Cut ${i + 1} → ${i + 2}: ${own ? on : `Default (${on})`}. Click for ${name}`,
    onclick: () => setJoin(a, b, i, next), onkeydown,
  }, on[0]);
}

/** One block per clip, in sequence order, sized by exported length; each is the cut after its clip, the last clip's belongs to the cut before it. Between two blocks, the button of their join. Blocks and join buttons are a tab stop each. */
function renderCutStrip(k) {
  const el = $('#cutStrip');
  const n = S.segments.length;
  const focused = el.contains(document.activeElement) ? document.activeElement : null;
  const had = focused && (focused.dataset.cut != null ? `[data-cut="${focused.dataset.cut}"]` : `[data-join="${focused.dataset.join}"]`);
  /** Arrow keys, Home and End from item ``i`` of the ``sel`` items: they move the focus, a click or Enter acts. */
  const moves = (i, sel) => (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey || e.shiftKey) return;
    const step = { ArrowLeft: -1, ArrowRight: 1 }[e.key];
    const to = step != null ? i + step : e.key === 'Home' ? 0 : e.key === 'End' ? n - 2 : null;
    if (to == null) return;
    e.preventDefault();
    const items = [...el.querySelectorAll(sel)];
    const go = items[Math.max(0, Math.min(n - 2, to))];
    for (const x of items) x.tabIndex = x === go ? 0 : -1;
    go.focus();
  };
  el.replaceChildren(...S.segments.flatMap((s, i) => {
    const props = {
      class: `cut-block${i === k || i === k + 1 ? ' on' : ''}${i === k ? ' at' : ''}`, 'data-kind': kindOf(cands().find((c) => c.id === s.id)),
      style: `flex-grow: ${Math.max(exportedLen(s), 0.1).toFixed(2)}`,
    };
    if (i === n - 1) return [h('span', { ...props, 'aria-hidden': 'true', onclick: () => focusMoment(S.segments[n - 2].id) })];
    const next = S.segments[i + 1];
    return [h('button', {
      ...props, type: 'button', 'data-cut': i, tabindex: i === k ? 0 : -1, 'aria-current': i === k ? 'true' : null,
      'aria-label': `Cut ${i + 1} of ${n - 1}: ${cutName(s)} into ${cutName(next)}`, title: `Cut ${i + 1} → ${i + 2}`,
      onclick: () => focusMoment(s.id),
      onkeydown: moves(i, '[data-cut]'),
    }), joinButton(s, next, i, n, k, moves(i, '[data-join]'))];
  }));
  if (had) el.querySelector(had)?.focus();
  const cur = el.querySelector(`[data-cut="${k}"]`); // a very long sequence scrolls here, not the page
  if (el.scrollWidth > el.clientWidth && (cur.offsetLeft < el.scrollLeft || cur.offsetLeft > el.scrollLeft + el.clientWidth)) el.scrollLeft = cur.offsetLeft - el.clientWidth / 2;
}

function bindCut() {
  $('#cutPlayBtn').addEventListener('click', () => (S.seqPlay?.across ? seqStop() : cutPlay()));
  $('#prevCut').addEventListener('click', () => stepCut(-1));
  $('#nextCut').addEventListener('click', () => stepCut(1));
  for (const img of [$('#cutOutFrame'), $('#cutInFrame')]) {
    // a src replaced before it loaded fires neither event, so these are always for the latest url (cutStill sets "loading")
    img.addEventListener('load', () => img.closest('.cut-frame').classList.remove('failed', 'loading'));
    img.addEventListener('error', () => {
      const frame = img.closest('.cut-frame');
      frame.classList.add('failed');
      frame.classList.remove('loading');
    });
  }
  const video = $('#optTransition');
  video.addEventListener('change', renderCut); // the control and the joins on Default
  video.addEventListener('change', renderJoinsNote);
  $('#cutTransition').addEventListener('click', (e) => {
    const value = e.target.closest('button[data-value]')?.dataset.value;
    if (!value || value === video.value) return;
    video.value = value;
    video.dispatchEvent(new Event('change')); // as if chosen in the Export panel: stored, audio linked
    $('#sideLive').textContent = `Default transition set to ${JOIN_NAMES[value]}${joinsLive()}`;
  });
  const join = $('#cutJoin');
  join.addEventListener('keydown', (e) => {
    const k = cutIndex();
    if (k < 0 || e.ctrlKey || e.metaKey || e.altKey) return;
    const step = HANDLE_STEPS[e.key];
    const r = rollRange(k);
    let by;
    if (step != null) by = e.shiftKey && e.key.startsWith('Arrow') ? step * 10 : step;
    else if (e.key === 'Home') by = r.min;
    else if (e.key === 'End') by = r.max;
    else return;
    e.preventDefault();
    if (S.seqPlay?.across) seqStop();
    rollCut(k, by, `roll:${S.segments[k].id}:${S.segments[k + 1].id}`);
  });
  join.addEventListener('pointerdown', (e) => {
    const k = cutIndex();
    if (k < 0 || cutDrag || (e.pointerType === 'mouse' && e.button !== 0)) return;
    e.preventDefault(); // no text selection or touch scrolling
    join.focus();
    join.setPointerCapture(e.pointerId);
    if (S.seqPlay?.across) seqStop();
    const [a, b] = [S.segments[k], S.segments[k + 1]];
    cutDrag = { id: e.pointerId, pid: S.id, a: a.id, b: b.id, x0: e.clientX, x: e.clientX, end0: a.end, out: planned(a).end, into: planned(b).start, moved: false, kept: false, raf: 0 };
    join.classList.add('dragging');
  });
  join.addEventListener('pointermove', (e) => {
    const d = cutDrag;
    if (!d || e.pointerId !== d.id) return;
    d.x = e.clientX;
    d.moved ||= Math.abs(e.clientX - d.x0) >= 3; // a plain click moves nothing
    if (d.moved && !d.raf) d.raf = requestAnimationFrame(cutDragFrame);
  });
  join.addEventListener('pointerup', (e) => {
    if (cutDrag && e.pointerId === cutDrag.id) cutDrag.x = e.clientX;
  });
  join.addEventListener('lostpointercapture', (e) => { // after pointerup or pointercancel
    const d = cutDrag;
    if (!d || e.pointerId !== d.id) return;
    cancelAnimationFrame(d.raf);
    if (d.moved) cutDragFrame();
    if (cutDrag) dropCutDrag();
    renderCut(); // the waveforms centre on the new cut
  });
}

// ---------- play sequence ----------

/** During "Play sequence": move on at the clip end, read live so trims apply at once. False when nothing plays on. */
function seqCheck() {
  const sp = S.seqPlay;
  if (!sp || sp.wait || v.paused) return true;
  const seg = seqSeg(sp.i);
  if (!seg) {
    seqStop('The playing clip left the sequence. Sequence stopped.');
    return false;
  }
  stepBlend();
  const fade = sp.i === 0 ? seqLap() : 0;
  // the spare player is not ready as the Fade starts: a next clip in this player's file dips to black here (#1028)
  if (fade && !dip && playhead() >= seg.end - fade && !spareReady() && dipFits()) dip = { lap: fade };
  stepDip();
  // the Fade of a cut starts this early; a spare player not ready yet leaves a cut (or a dip, when clip 2 is in this player's file)
  const lap = fade && !dip && spareReady() ? fade : 0;
  if (dip && sp.i === 0) {
    if (playhead() >= seg.end - dip.lap / 2) { // black: seek to the next clip, half the Fade in
      cued = null;
      unload(spare()); // its open is late: let the seek have the file
      seqGo(1);
    }
  } else if (playhead() >= seg.end - (lap || FRAME)) seqGo(sp.i + 1, lap);
  return !v.paused && !S.seqPlay?.wait;
}

/** Clip ``j`` of the playing sequence, running where its export runs; over a cut only the last CUT_PLAY s of the first clip and the first of the second (a longer Fade: its length and a second), tails included. */
const seqSeg = (j) => {
  const s = S.segments.find((x) => x.id === S.seqPlay?.order[j]);
  if (!s) return s;
  const { start, end } = planned(s);
  if (!S.seqPlay.across) return { ...s, start, end };
  const span = Math.max(CUT_PLAY, seqLap() + 1);
  return j ? { ...s, start, end: Math.min(end, start + span) } : { ...s, start: Math.max(start, end - span), end };
};
/** Clip ``j`` has left the sequence, or the export drops it (zero length): nothing to play. */
const seqEmpty = (j) => { const s = seqSeg(j); return !s || s.end - s.start <= 1e-6; };
/** Seconds the two clips of the playing cut overlap, read live; 0 for a Cut or any other play. */
function seqLap() {
  if (!S.seqPlay?.across) return 0;
  const [a, b] = S.seqPlay.order.map((id) => S.segments.find((s) => s.id === id));
  return a && b ? joinLap(a, b) : 0;
}
const seqNow = () => S.seqPlay?.order[S.seqPlay.i];
const nowIcon = () => h('span', { class: 'seq-now', 'aria-hidden': 'true', title: 'Playing' }, icon('play'));

function renderSeqPlay() {
  const b = $('#seqPlayBtn');
  b.disabled = !S.seqPlay && !S.segments.length;
  const hide = !!S.selected.size && !S.seqPlay;
  if (hide && document.activeElement === b) $('#seqClearSel').focus();
  b.hidden = hide;
  b.textContent = S.seqPlay ? 'Stop' : 'Play sequence';
  if (S.seqPlay) b.setAttribute('aria-keyshortcuts', 'Escape');
  else b.removeAttribute('aria-keyshortcuts');
  b.title = S.seqPlay ? 'Stop (Escape)' : 'Play the clips in sequence order';
  // the stage mode follows the playback: Sequence while it plays, else Clip; Cut until it is left
  const mode = S.cutMode ? 'cut' : S.seqPlay ? 'sequence' : 'clip';
  for (const m of $('#stageMode').children) {
    m.setAttribute('aria-pressed', String(m.dataset.mode === mode));
    if (m.dataset.mode === 'sequence') m.disabled = !S.seqPlay && !S.segments.length;
    if (m.dataset.mode === 'cut') { // stays focusable and pressable: a press says why (enterCut)
      m.setAttribute('aria-disabled', String(S.segments.length < 2));
      m.title = S.segments.length < 2 ? CUT_NEEDS : '';
    }
  }
  renderCutPlay();
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
  for (const b of sbBlocks()) b.classList.toggle('playing', !!now && b.dataset.id === now);
}

/** Scroll ``li`` into view inside its column only: when the page scrolls as a whole (narrow windows), it stays put, so the video stays on screen. */
function revealRow(li) {
  const col = li.closest('.col');
  if (!col || getComputedStyle(col).overflowY === 'visible') return;
  const r = li.getBoundingClientRect();
  if (!r.height) return; // hidden (the Export panel covers the list): it measures as 0, not as a row to scroll to
  const cs = getComputedStyle(col);
  const box = col.getBoundingClientRect();
  const top = box.top + (parseFloat(cs.scrollPaddingTop) || 0); // below the sticky tabs and header
  // above the sticky chat input and the Sequence footer
  const bottom = box.bottom - (parseFloat(cs.scrollPaddingBottom) || 0) - (parseFloat(cs.getPropertyValue('--seq-foot-h')) || 0);
  if (r.top < top) col.scrollTop -= top - r.top;
  else if (r.bottom > bottom) col.scrollTop += Math.min(r.bottom - bottom, r.top - top);
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
  const k = cutIndex() + 1;
  const lap = seqLap(); // counted in both clips
  return `${sp.across ? `Cut ${k} → ${k + 1}` : `Clip ${sp.i + 1} of ${sp.order.length}`} · ${fmtLen(done - (sp.i > 0 ? lap : 0))} / ${fmtLen(total - lap)}`;
}

/** Play the sequence from its first clip, or from clip ``from`` (an id); with ``across`` ({back}) only ``order``, the two clips of a cut, over the cut. */
function seqStart(from, order = S.segments.map((s) => s.id), across = null) {
  if (!S.segments.length) return;
  if (!across) {
    if (S.seqPlay?.across) seqStop('', false);
    if (S.cutMode) setCut(false);
  }
  for (const el of els) {
    if (!el.paused) continue;
    el.play().catch(() => {}); // inside the click: lets later play() calls start it (Safari)
    el.pause();
  }
  S.seqPlay = { order, i: -1, wait: false, across };
  banner('');
  renderSeqPlay();
  seqGo(Math.max(0, S.seqPlay.order.indexOf(from)));
  if (S.seqPlay) seqPrepare(); // seqGo may have stopped it: nothing could be previewed
}

/** ``back``: over a cut, the clip selected before it is selected again, so the cut stays the current one. */
function seqStop(msg = S.seqPlay?.across ? 'Stopped playing across the cut' : 'Sequence stopped', back = true) {
  if (!S.seqPlay) return;
  const { across } = S.seqPlay;
  S.seqPlay = null;
  endBlend();
  endDip();
  cued = null;
  clearTimeout(S.seqTimer);
  v.pause();
  clearStall(); // a paused element fires no pause event, so the note would stay
  unload(spare()); // the visible player stays the stage player
  for (const el of els) el.preload = 'metadata'; // the stage preview buffers on demand again
  if (across && back && S.focus !== across.back && S.segments.some((s) => s.id === across.back)) showMoment(across.back);
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

/** While a clip of the sequence has no ready file covering it, poll the sources every 2 s (the stage polls only its own). */
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
    await pollSources();
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
  let j = sp.i + 1;
  while (j < sp.order.length && seqEmpty(j)) j += 1;
  const cur = seqSeg(sp.i);
  const seg = seqSeg(j);
  const src = srcOf(seg?.source);
  const p = readyFor(src, seg) && fileFor(src, seg); // the whole source, or a window covering the clip
  // a file the visible player has open plays the clip there, sought to (seqGo): opening it again
  // costs seconds on a network share (#1020); only a Fade plays two clips of it at once
  if (!p || (cur && !seqLap() && v.dataset.url === p.url)) {
    cued = null; // not playable yet, or it plays in the visible player
    return;
  }
  cueSpare(seg, src, p);
}

/** Hold ``seg`` of ``src`` in the spare player, paused at its start, from ``p`` (fileFor). */
function cueSpare(seg, src, p) {
  const o = spare();
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
  } else if (o.readyState >= HTMLMediaElement.HAVE_METADATA && Math.abs(o.currentTime - cued.vt) > GAP) {
    o.currentTime = cued.vt; // a spare already there (precueCut) is not sought again
  }
}

let precueTimer = 0;
/**
 * In Cut mode, open clip 2 of a Fade cut in the spare player once the cut rests on screen, so
 * the first "Play across the cut" has its second picture in time: a fragmented MP4 without an
 * index takes seconds to open a second time (#1028). Only for a Fade between clips of the file
 * the stage player has open, and only once that player is idle (canplay asks again), so two
 * opens of one file never compete; otherwise a spare it opened is released. A roll keeps it.
 */
function precueCut(now = false) {
  clearTimeout(precueTimer);
  if (S.seqPlay || cutDrag) return; // a play cues its own clip
  const k = S.cutMode ? cutIndex() : -1;
  const [a, b] = k < 0 ? [] : [S.segments[k], S.segments[k + 1]];
  const seg = b && { ...b, ...planned(b) };
  const src = srcOf(seg?.source);
  const p = a && joinLap(a, b) > 0 && !v.error && readyFor(src, seg) && fileFor(src, seg);
  if (!p || p.url !== v.dataset.url) {
    if (cued) unload(spare()); // outside a play only this function cues
    cued = null;
    return;
  }
  if (cued && spare().dataset.url !== p.url) { // it holds another file
    unload(spare());
    cued = null;
  }
  if (v.seeking || v.readyState < HTMLMediaElement.HAVE_FUTURE_DATA) return; // the stage player is still opening or seeking
  if (now) cueSpare(seg, src, p);
  else precueTimer = setTimeout(() => precueCut(true), PRECUE_WAIT);
}

/**
 * Play clip ``j`` of the sequence. Clips that left it are passed over and clips whose preview
 * could not be built are skipped with a note; after the last one the sequence stops. The next
 * clip waits in the spare player, so a step is a swap; one in the file the visible player has
 * open is sought to there; otherwise the stage loads it (and waits for its proxy). ``lap``: the
 * seconds of the Fade starting here (startBlend), the old clip playing on under it.
 */
function seqGo(j, lap = 0) {
  const sp = S.seqPlay;
  const skipped = [];
  for (; j < sp.order.length; j += 1) {
    if (seqEmpty(j)) continue;
    const seg = seqSeg(j);
    const src = srcOf(seg.source);
    if (src && src.status !== 'failed') break;
    skipped.push(seg.label || seg.id);
  }
  if (skipped.length) banner(`Skipped ${skipped.map((n) => `"${n}"`).join(', ')}: the preview could not be built.`);
  if (j >= sp.order.length) return seqStop(sp.i < 0 ? 'Nothing in the sequence can be previewed.' : sp.across ? 'Finished playing across the cut' : 'Sequence finished');
  const cur = seqSeg(sp.i);
  const seg = seqSeg(j);
  const src = srcOf(seg.source);
  const o = spare();
  sp.i = j;
  sp.wait = false;
  const url = readyFor(src, seg) && fileFor(src, seg).url; // the whole source, or a window covering the clip
  if (cur && !lap && seqLap()) $('#cutRollMsg').textContent = `Clip ${j + 1} was not ready in time, so this play shows ${dip ? 'a dip to black' : 'a cut'} in place of the Fade. Play it again to see the Fade.`;
  if (!lap && !dip && cur && seg.source === cur.source && Math.abs(playhead(cur) - seg.start) <= FRAME + GAP && v.dataset.url === url && !v.paused) {
    showMoment(seg.id, true); // contiguous: plays on without a cut
  } else if (cued?.id === seg.id && o.dataset.url === url && !o.error) {
    const old = v;
    setV(o); // hard cut to the preloaded player
    setSeekWant(null);
    setPendingSeek(null);
    if (o.readyState < HTMLMediaElement.HAVE_METADATA) setPendingSeek(seg.start); // applied on loadedmetadata
    else if (Math.abs(o.currentTime - (seg.start - offsetOf())) > GAP) o.currentTime = seg.start - offsetOf();
    o.classList.add('active');
    old.classList.remove('active');
    v.play().catch(() => {});
    if (lap) startBlend(old, cur.end, lap);
    else old.pause();
    showMoment(seg.id, true);
  } else if (!lap && url && v.dataset.url === url && !v.error) {
    showMoment(seg.id, true); // the visible player's own file: a seek, not a new open (#1020)
    seekTo(dip ? seg.start + dip.lap / 2 : seg.start); // a dip comes back half the Fade in, so the play keeps its length
    v.play().catch(() => {});
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

/** The spare player holds the second clip of the playing cut, at its start, with data to play on. */
function spareReady() {
  const o = spare();
  return cued?.id === S.seqPlay.order[1] && !o.error && !o.seeking && o.readyState >= HTMLMediaElement.HAVE_FUTURE_DATA;
}

/** The second clip of the playing cut plays from the file the visible player has open, so a seek reaches it without the spare player. */
function dipFits() {
  const seg = seqSeg(1);
  const src = srcOf(seg?.source);
  return !!seg && !v.error && readyFor(src, seg) && fileFor(src, seg).url === v.dataset.url;
}

/**
 * One frame of a dip: the first clip fades to black over the first half of the Fade, the second
 * fades in from black over its second half, each on its own clock, so a slow seek holds black.
 * Picture and sound follow together; with reduced motion the picture changes at once.
 */
function stepDip() {
  if (!dip) return;
  const sp = S.seqPlay;
  const seg = seqSeg(sp.i);
  const half = dip.lap / 2;
  const p = sp.i === 0 ? (seg.end - half - playhead()) / half : (playhead() - seg.start - half) / half;
  if (sp.i > 0 && p >= 1) return endDip();
  const x = Math.min(Math.max(p, 0), 1);
  v.style.opacity = String(reduceMotion.matches ? Math.round(x) : x);
  v.volume = previewVolume() * x;
}

/** End a dip, at its end or when the play stops: the visible player whole again, at the shared volume. */
function endDip() {
  if (!dip) return;
  dip = null;
  for (const el of els) {
    el.style.opacity = '';
    el.volume = previewVolume();
  }
}

/** Start a Fade of ``lap`` seconds: ``old``, the outgoing player, plays on to ``end`` (source time) under the visible player, which just took over. */
function startBlend(old, end, lap) {
  blend = { old, end: end - (Number(old.dataset.offset) || 0), lap };
  mixBlend(0);
}

/**
 * Show the share ``p`` (0-1) of a Fade. Only the player on top changes its opacity, so the two
 * add up to a whole picture; volumes follow ``p`` under the shared volume. With reduced motion
 * the picture changes once, half way, and the sound still ramps.
 */
function mixBlend(p) {
  const mix = reduceMotion.matches ? Math.round(p) : p;
  const [under, top] = els;
  under.style.opacity = '1';
  top.style.opacity = String(top === v ? mix : 1 - mix);
  v.volume = previewVolume() * p;
  blend.old.volume = previewVolume() * (1 - p);
}

/** One frame of a Fade, on the clock of the incoming clip, so a stall holds the mix. The outgoing clip holds its last frame past its out point. */
function stepBlend() {
  if (!blend) return;
  const p = (playhead() - seqSeg(S.seqPlay.i).start) / blend.lap;
  if (p >= 1) return endBlend();
  if (blend.old.currentTime >= blend.end) blend.old.pause();
  mixBlend(Math.max(p, 0));
}

/** End a Fade, at its end or when the play stops: the visible player alone, at the shared volume. */
function endBlend() {
  if (!blend) return;
  const { old } = blend;
  blend = null;
  old.pause();
  for (const el of els) {
    el.style.opacity = '';
    el.volume = previewVolume();
  }
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
  const sent = reviewKey();
  S.saving = true;
  renderSave();
  try {
    await api(`/api/projects/${encodeURIComponent(S.id)}/review`, { method: 'PUT', body: JSON.parse(sent) });
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
    const b = c.original_end ?? c.pick?.end ?? c.end; // the sequence's end: ``pick.end`` and ``end`` include the tail
    return c.id === s.id && Math.abs(a - s.start) < 0.02 && Math.abs(b - s.end) < 0.02;
  });
}

/** A join would now come out other than in the current export's final: by its own choice or the Video transition. Never for a final whose joins are not known (``final_joins`` null). */
const joinsOutdated = () => !!S.project?.final_current && !!S.project.final_joins?.some((f) => (S.joins.find((j) => j.a === f.a && j.b === f.b)?.transition ?? $('#optTransition').value) !== f.transition);
/** The note for ``#sideLive`` after a change in Cut mode: an open Export panel's ``#joinsNote`` says it itself. */
const joinsLive = () => (joinsOutdated() && $('#panelExport').hidden ? `. ${JOINS_NOTE}` : '');

function renderJoinsNote() {
  const text = joinsOutdated() ? JOINS_NOTE : '';
  const note = $('#joinsNote');
  note.hidden = !text;
  if (note.textContent !== text) note.textContent = text;
}

function renderExport() {
  syncJoinFields();
  renderJoinsNote();
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

/** The preview's Analyze footage button follows the Moments column's: gone while an analysis runs or with no videos. */
function syncStageAnalyze() {
  const b = $('#stageAnalyze');
  if (!b) return;
  const big = $('#analyzeBig');
  b.hidden = !big || big.hidden || !!$('#resumeBig') || !S.project?.sources?.length;
  b.disabled = !!big?.disabled;
  b.title = big?.title ?? '';
}

function renderBusy() {
  syncRankingBusy(); // the ranking cards' buttons follow a run on the project
  const p = S.project;
  const busy = !!S.active;
  const why = busy ? 'Another job is running.' : '';
  const set = (id, off, reason) => {
    const b = $(id);
    b.disabled = off;
    b.title = off ? reason : '';
  };
  renderProjectMenu(); // Analyze again follows the running job
  const resuming = S.active?.project_id === S.id && S.active?.action === 'resume';
  const held = heldElsewhere(p);
  const note = $('#runBusy'); // before any job is tried: a run elsewhere holds this project
  note.hidden = !held;
  if (note.textContent !== (held ? BUSY_TEXT : '')) note.textContent = held ? BUSY_TEXT : '';
  for (const b of [$('#resumeBtn'), $('#resumeBig')]) {
    if (!b) continue;
    b.disabled = !p || busy || held;
    b.title = resuming ? 'Resuming… progress is in Jobs.' : busy ? why : held ? BUSY_TEXT : '';
    const label = resuming ? 'Resuming…' : b.dataset.label ?? b.textContent;
    if (b.textContent !== label) b.textContent = label;
  }
  const big = $('#analyzeBig');
  if (big) {
    // a first analysis, or the resume of one (#811)
    const own = S.active?.project_id === S.id
      && (S.active.action === 'analyze' || (S.active.action === 'resume' && /analyze$/.test(p?.run?.command ?? '')));
    const steps = own && !!stepsJob(); // an older CLI sends no steps: the Jobs log is all there is
    const reason = own ? (steps ? 'Analyzing…' : 'Analyzing… progress is in Jobs.') : busy ? why
      : held ? BUSY_TEXT : p?.source_error ? sourceErrorText(p) : 'Add videos to the source folder first.';
    set('#analyzeBig', !p || busy || held || !p.sources?.length, reason);
    set('#analyzeOptsBtn', !p || busy || held || !p.sources?.length, reason);
    const label = own ? 'Analyzing…' : 'Analyze footage';
    if (big.textContent !== label) big.textContent = label;
    // #811: while the first analysis runs only Show folder, the steps and Pause analysis stay
    const empty = $('#momentsEmpty');
    const pause = $('#pauseBtn');
    const hiding = [big, $('#analyzeOptsBtn'), $('#resumeBig')];
    const lost = [...hiding, $('#stageAnalyze')].includes(document.activeElement) || (!!empty.dataset.refocus && document.activeElement === document.body);
    empty.classList.toggle('running', own);
    for (const b of hiding) if (b) b.hidden = own || b.dataset.gone === '1';
    if (!own && pause.getAttribute('aria-disabled')) {
      pause.removeAttribute('aria-disabled');
      pause.textContent = 'Pause analysis';
    }
    const wasPause = !pause.hidden && document.activeElement === pause;
    pause.hidden = !own;
    if (own && lost) pause.focus();
    else if (wasPause && !own) ([$('#resumeBig'), big, $('#revealBtn')].find((b) => b && !b.hidden && !b.disabled) ?? $('#momentsTitle')).focus();
    const out = $('#analyzeWhy');
    out.classList.toggle('progress', own);
    if (out.textContent !== big.title) out.textContent = big.title; // live region: announce changes only
  }
  syncStageAnalyze();
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
  const final = p?.outputs?.some((o) => /\.(mp4|fcpxml|otio)$/i.test(o.name)); // a video or a timeline-only export
  if (p?.clips) return final && p.final_current ? 'outputs' : 'clips';
  return p?.outputs?.length ? 'outputs' : null;
}

// ---------- analysis steps ----------

const STEP_TEXT = {
  running: 'Running', done: 'Done', skipped: 'Skipped', failed: 'Failed', interrupted: 'Interrupted',
  cancelled: 'Cancelled', waiting: 'Waiting', notrun: 'Not run',
};
const STEP_ICON = { done: '✓', skipped: '–', failed: '✕', interrupted: '!', cancelled: '■', waiting: '○', notrun: '–' }; // running: a spinner
// The CLI's step names, in order (a stable API): analyze sends the first 7, edit all 9. Labels for
// steps not sent yet; a label the CLI sends wins.
const STEP_LABELS = {
  transcribe: 'Transcription', reactions: 'Reactions', keyframes: 'Keyframes', crashes: 'Crashes',
  dialogue: 'Dialogue', ranking: 'Ranking', vision: 'Vision rating', clips: 'Clips', assemble: 'Assemble',
};
const STEP_NAMES = Object.keys(STEP_LABELS);

/**
 * The rows to show: the steps sent, plus the ones not sent yet, from the job's ``plan`` (the
 * server's list of the command's steps) or, without one, when ``total`` is a known step count (7
 * or 9): "waiting" while the job runs, "not run" after it ended (a cancel or fatal error).
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
  const planned = plan ?? (total === 7 || total === 9 ? STEP_NAMES.slice(0, total).map((step) => ({ step, label: STEP_LABELS[step] })) : null);
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
  if (!S.id || j?.project_id !== S.id || (j.action !== 'analyze' && j.action !== 'resume') || j.status === 'completed') return null;
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
  // false: the current export is not assembled yet, or its latest assembly is a timeline without a video
  const current = (S.project?.final_current ?? true) && final === outs[0];
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
    if (action === 'clips' || action === 'assemble') await saveReview(); // clips use the saved sequence, assemble its joins
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
    format: $('#optFormat').value,
    // off with a timeline format (syncTiming), which the server refuses them with
    loudnorm: $('#optLoudnorm').checked && !$('#optLoudnorm').disabled,
    compress_peaks: peaks(),
    captions: $('#optCaptions').checked && !$('#optCaptions').disabled,
    ...rest,
    ...(audioFade === undefined || $('#optFinalAudioFade').disabled ? {} : { audio_fade: audioFade }),
    ...(model.value && !model.disabled ? { caption_model: model.value } : {}),
  };
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
    // a busy project (folder_busy) is a plain state, not a failure to read up on in the log
    if (finished.some((j) => j.status === 'failed' && j.error_code !== 'folder_busy')) S.logOpen = true;
    renderJobs();
    renderBusy();
    if (finished.some((j) => j.project_id === S.id)) refreshProject();
    if (finished.length) loadProjects().catch(() => {}); // the list marks projects with a stopped run
    else if (heldElsewhere(S.project) && !S.heldCheck) // until the other run lets the project go; one at a time
      S.heldCheck = refreshProject().finally(() => { S.heldCheck = null; });
  } catch (e) {
    if (S.expired) return; // the banner says why; polling stops
    $('#jobStatus').textContent = e.message;
    $('#jobStatus').dataset.status = 'failed';
  }
  S.jobTimer = setTimeout(pollJobs, S.active ? 1000 : 5000);
}

/** A clips run that kept the finished clips of a stopped export (``taken_over`` of its result, #109). */
const takenOver = (n) => `${n} clip${n === 1 ? '' : 's'} taken over from the stopped export`;

function jobLine(j) {
  const name = S.projects.find((p) => p.id === j.project_id)?.name ?? j.project_id;
  const n = JOB_NAMES[j.action] ?? j.action;
  if (j.status === 'running') return `${n} running for ${name}`;
  if (j.status === 'cancelling') return `${n} stopping for ${name}`;
  if (j.status === 'completed') return `${n} finished for ${name}${j.taken_over ? `. ${takenOver(j.taken_over)}` : ''}`;
  if (j.status === 'cancelled') return `${n} cancelled for ${name}`;
  if (j.error_code === 'folder_busy') return `${n} did not run for ${name}: ${j.error}`; // the server's plain text
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
  const index = $('#setVisualIndex');
  index.checked = st.visual_index === true;
  index.dataset.saved = String(index.checked);
  $('#setNerdStats').checked = S.nerdStats;
  S.nerdStatsWas = S.nerdStats; // Cancel or Escape goes back to this
  $('#setAdvExport').checked = S.advExport;
  S.advExportWas = S.advExport;
  $('#setVerboseMoments').checked = S.verboseMoments;
  S.verboseMomentsWas = S.verboseMoments;
  $('#setSingleKeys').checked = S.singleKeys;
  S.singleKeysWas = S.singleKeys;
  S.themeWas = $('#setTheme').value = theme();
  $('#setDeletePanel').hidden = !S.id; // only the open project can be deleted
  d.showModal();
  input.focus();
}

async function saveSettings() {
  const input = $('#setProjects');
  const err = $('#settingsError');
  const note = $('#settingsNote');
  const value = input.value.trim();
  err.textContent = note.textContent = '';
  const body = {};
  if (!input.readOnly) {
    if (!value) return (err.textContent = 'Enter the full path of a folder (~ is your home folder).');
    if (value !== input.dataset.saved) body.projects_dir = value;
  }
  const index = $('#setVisualIndex');
  if (index.checked !== (index.dataset.saved === 'true')) body.visual_index = index.checked;
  if (!Object.keys(body).length) return $('#settingsDialog').close();
  if (body.projects_dir && dirty() && !confirm('The project list will show the new folder. Discard the unsaved changes to this sequence?')) return;
  $('#settingsSave').disabled = true;
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
  $('#settingsDialog').close();
  if (!body.projects_dir) return;
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

const LIST_SHOWN = 8; // names listed in the delete dialog

/** Delete the open project (Settings): confirm with its size and what stays, move it to the Recycle Bin / Trash, or, when there is none, ask again before deleting for good. */
async function deleteProject() {
  const id = S.id;
  if (!id) return;
  const url = `/api/projects/${encodeURIComponent(id)}`;
  const err = $('#settingsError');
  err.textContent = '';
  let plan;
  try {
    plan = await api(`${url}/delete-plan`);
  } catch (e) {
    err.textContent = e.message;
    return;
  }
  const kept = plan.kept ?? [];
  const gone = plan.deleted ?? [];
  const list = (names, count) => [
    ...names.slice(0, LIST_SHOWN).map((n) => `  ${n}`),
    ...(count > Math.min(names.length, LIST_SHOWN) ? [`  and ${count - Math.min(names.length, LIST_SHOWN)} more`] : []),
  ];
  const lines = [
    `Moves the ${plan.files} file${plan.files === 1 ? '' : 's'} Cutpoint made for this project, including the final videos and exported clips, to the Recycle Bin or Trash. Your source videos are not touched.`,
  ];
  if (plan.deleted_count) lines.push('Final videos and timelines that go:', ...list(gone, plan.deleted_count));
  if (plan.kept_count) lines.push('These stay in the project folder:', ...list(kept, plan.kept_count));
  if (!(await confirmAction(`Delete project ${plan.name} (${fmtBytes(plan.size)})?`, lines.join('\n'), 'Delete project'))) return;
  const btn = $('#setDelete');
  const label = btn.textContent;
  btn.disabled = true; // one request at a time
  btn.setAttribute('aria-busy', 'true');
  btn.textContent = 'Deleting…';
  let permanent = false;
  let out;
  try {
    for (;;) {
      try {
        out = await api(url, { method: 'DELETE', body: { permanent } });
        break;
      } catch (e) {
        if (e.code !== 'no_trash' || permanent) {
          err.textContent = e.message;
          return;
        }
        const text = `${e.message}\nDelete the files permanently instead? They cannot be restored.`;
        if (!(await confirmAction('Delete permanently?', text, 'Delete permanently'))) return;
        permanent = true;
      }
    }
  } finally {
    btn.disabled = false;
    btn.removeAttribute('aria-busy');
    btn.textContent = label;
  }
  $('#settingsCancel').click(); // closes the dialog and undoes any change not saved
  const stayed = out?.kept ?? 0;
  const done = `Deleted project ${plan.name}.${stayed ? ` ${stayed} file${stayed === 1 ? '' : 's'} stayed in its folder.` : ''}`;
  try {
    closeProject();
    banner(done);
    await loadProjects();
  } catch (e) {
    fail(e);
  }
}

function bindSettings() {
  const d = $('#settingsDialog');
  const gear = $('#settingsBtn');
  gear.append(icon('gear'));
  gear.addEventListener('click', openSettings);
  d.addEventListener('close', () => gear.focus());
  $('#setTheme').addEventListener('change', (e) => setTheme(e.target.value));
  $('#setDelete').addEventListener('click', deleteProject);
  const undo = () => {
    NERD.set(S.nerdStatsWas);
    // a field revealed for an out-of-range value (showAdvExport(true)) stays unless the box changed
    if ($('#setAdvExport').checked !== S.advExportWas) ADV.set(S.advExportWas);
    VERBOSE.set(S.verboseMomentsWas);
    KEYS.set(S.singleKeysWas);
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
  renderStyleMenu(); // the detected style follows the project's analysis
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
 * Export & Assembly Options: each change is remembered per browser (only the options the user
 * changed, so the others follow OPT_DEFAULTS); without storage they last the session. Link
 * Transitions makes the audio transition follow the video one; a crossfade needs a video fade
 * (the CLI refuses it on cuts), so it is unavailable with Cut.
 */
function initExportOptions() {
  const v = readStore(OPTS_KEY, null, { json: true }); // blocked or unreadable: the defaults
  const saved = v && typeof v === 'object' ? v : {};
  delete saved.ids; // candidate ids belong to one project (A284): an older page stored them
  const store = (key, value) => {
    saved[key] = value;
    writeStore(OPTS_KEY, saved); // blocked or full: the choice still applies until reload
  };
  const checks = { vertical: '#optVertical', captions: '#optCaptions', speech_aware: '#optSpeech', smart_tail: '#optTail', loudnorm: '#optLoudnorm', link: '#optLink' };
  for (const [key, id] of Object.entries(checks)) {
    $(id).checked = typeof saved[key] === 'boolean' ? saved[key] : OPT_DEFAULTS[key];
    $(id).addEventListener('change', (e) => {
      store(key, e.target.checked);
      if (key === 'link') syncAudio();
      else syncTiming();
    });
  }
  for (const [key, id] of Object.entries(OPT_FIELDS)) {
    if (typeof saved[key] === 'string') $(id).value = saved[key];
    $(id).addEventListener('change', (e) => {
      if (key !== 'ids') store(key, e.target.value); // ids are per project: never stored (A284)
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

/** Turn a field of the Export panel and its label on or off. */
function able(id, on) {
  const el = $(id);
  el.disabled = !on;
  document.querySelector(`label[for=${el.id}]`)?.classList.toggle('off', !on);
}

/**
 * The Export fields that follow the joins of the sequence, on Default or by their own choice:
 * Fade length while one fades; Assembly audio fade unless every edge has an audio crossfade, so
 * under the Crossfade audio transition while one is a Cut (its edges and the video's two ends use it).
 */
function syncJoinFields() {
  const joins = S.segments.slice(1).map((b, i) => joinTransition(S.segments[i], b));
  able('#optFadeDur', $('#optTransition').value === 'fade' || joins.includes('fade'));
  able('#optFinalAudioFade', $('#optFormat').value === 'mp4' && ($('#optAudioTransition').value !== 'crossfade' || joins.includes('cut')));
}

/**
 * Enable the Export panel's fields that apply: the automatic-pick fields with Automatic picks, Top
 * moments without candidate ids (the CLI takes one or the other), the speech caps with Speech-aware
 * cuts, the tail lengths with Smart tails. The sequence's moments keep their exact ends with Smart
 * tails off, which turns event tails off too, so Event tails is not sent then. The fields the CLI
 * would not use are off too: Output size with Vertical 9:16, Fade length without a Fade to use it,
 * Assembly audio fade with the Crossfade audio transition and no Cut join, Captions model without captions. A
 * timeline Output format (fcpxml, otio) joins and encodes nothing, so Normalize loudness, Burn in
 * captions, Captions model and Assembly audio fade are off with it, and say why. The crop box
 * shows with Crop Custom.
 */
function syncTiming() {
  const auto = picks() === 'auto';
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
  syncJoinFields();
  const video = $('#optFormat').value === 'mp4';
  for (const id of ['#optLoudnorm', '#optCaptions']) {
    const el = $(id);
    el.disabled = !video;
    el.closest('label').classList.toggle('off', !video);
  }
  able('#optCaptionModel', video && $('#optCaptions').checked);
  for (const id of ['#optLoudnorm', '#optCaptions', '#optCaptionModel', '#optFinalAudioFade']) timelineNote($(id), !video);
  renderAdvInUse();
}

/** Add to or take from ``el``'s tooltip and description the reason a timeline Output format turns it off. */
function timelineNote(el, on) {
  const label = el.closest('label') ?? document.querySelector(`label[for=${el.id}]`);
  label.dataset.title ??= label.title;
  const note = $('#dTimelineOnly').textContent;
  label.title = on ? `${label.dataset.title} ${note}` : label.dataset.title;
  const ids = (el.getAttribute('aria-describedby') ?? '').split(' ').filter((d) => d && d !== 'dTimelineOnly');
  el.setAttribute('aria-describedby', [...ids, ...(on ? ['dTimelineOnly'] : [])].join(' '));
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

/** The Color sliders as ``{mode, strength}`` (strength 0-1), as the assistant's color op and history have it. */
function colorChoice() {
  return { mode: color(), strength: Number($('#optColorStrength').value) / 100 };
}

/** Set the Color sliders to ``c`` ({mode, strength}; no strength leaves Strength) as if the user moved them, so the choice is saved for the project. */
function chooseColor(c) {
  if (Number.isFinite(c.strength)) {
    $('#optColorStrength').value = String(Math.round(c.strength * 100));
    $('#optColorStrength').dispatchEvent(new Event('input'));
  }
  $('#optColor').value = String(Math.max(0, COLOR_STOPS.indexOf(c.mode)));
  $('#optColor').dispatchEvent(new Event('input'));
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
  initTheme();
  bindSettings();
  bindModelsTab();
  NERD.load();
  showTab = initSideTabs();
  initMomentOrder();
  initMomentFilters();
  initKindFilter();
  bindAssistant();
  S.savedOpts = initExportOptions();
  showAdvExport(ADV.load());
  VERBOSE.set(VERBOSE.load());
  KEYS.set(KEYS.load());
  showPlaying(false);
  $('#resetBtn').addEventListener('click', () => {
    const c = current();
    if (!c?.cand) return;
    remember();
    if (c.seg) {
      c.seg.start = c.cand.start;
      c.seg.end = c.cand.end;
      changed();
    } else S.drafts[c.id] = { at: S.drafts[c.id]?.at }; // keep where a removed moment goes back
    renderTrim();
    seekTo(c.cand.start);
  });
  bindCut();
  $('#prevClip').addEventListener('click', () => stepClip(-1));
  $('#nextClip').addEventListener('click', () => stepClip(1));

  initProjectMenu();
  initStyleMenu();
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
    $('#seqLive').textContent = `${shownName(c.cand, c.seg ? c.label : null) || c.label} ${c.seg ? 'removed from' : 'added to'} the sequence`;
  });
  $('#seqTotal').addEventListener('click', () => editSeqTarget($('#seqTargetForm').hidden));
  $('#seqTargetForm').addEventListener('submit', (e) => {
    e.preventDefault();
    commitSeqTarget();
  });
  $('#seqTargetCancel').addEventListener('click', () => editSeqTarget(false));
  $('#seqTargetForm').addEventListener('keydown', (e) => {
    if (e.key !== 'Escape') return;
    e.preventDefault(); // the field only; the page's own Escape (stop Play sequence) waits
    e.stopPropagation();
    editSeqTarget(false);
  });
  $('#seqTargetInput').addEventListener('input', () => {
    $('#seqTargetErr').textContent = '';
    $('#seqTargetInput').removeAttribute('aria-invalid');
  });
  $('#seqPlayBtn').addEventListener('click', () => (S.seqPlay ? seqStop() : seqStart()));
  $('#stageMode').addEventListener('click', (e) => {
    const mode = e.target.closest('button[data-mode]')?.dataset.mode;
    if (mode === 'cut') enterCut();
    else if (mode === 'sequence' && (!S.seqPlay || S.seqPlay.across)) seqStart();
    else if (mode === 'clip' && S.cutMode) leaveCut();
    else if (mode === 'clip' && S.seqPlay) seqStop('Playing the selected clip. Sequence stopped.');
  });
  $('#undoBtn').append(icon('undo'));
  $('#redoBtn').append(icon('redo'));
  $('#undoBtn').addEventListener('click', undo);
  $('#redoBtn').addEventListener('click', redo);
  // a held clip scrolls the list near its edges; the drag ends wherever it is dropped or leaves the window
  document.addEventListener('dragover', dragScroll);
  document.addEventListener('dragenter', dragScroll); // Edge: only dragenter fires when the element under the pointer changes
  gapDrag($('#seqList'), 'seq', false);
  gapDrag($('#sbList'), 'sb', true);
  addEventListener('drop', endDrag);
  addEventListener('dragend', endDrag);
  addEventListener('dragleave', (e) => {
    if (!e.relatedTarget && (e.clientX <= 0 || e.clientY <= 0 || e.clientX >= innerWidth || e.clientY >= innerHeight)) endDrag();
  });
  $('#seqRemoveSel').addEventListener('click', removeSelected);
  $('#seqSelAll').addEventListener('change', selectAll);
  $('#seqClearSel').addEventListener('click', () => {
    S.selected.clear();
    renderBatch();
    focusRow(0, 'select');
  });
  $('#resumeBtn').addEventListener('click', (e) => resumeRun(e.currentTarget));
  $('#rankUndoBtn').addEventListener('click', () => undoRanking());
  // an out-of-range Analyze option blocks Analyze: open the options so the browser can point at it
  $('#againOk').addEventListener('click', () => {
    if (!$('#analyzeDialog form').checkValidity()) $('#analyzeOpts').open = true;
  });
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
  // the lengths follow the export's cuts
  for (const id of ['#optTail', '#optSpeech', '#optEventTails', '#optEventWindow', '#optTailMin', '#optTailMax', '#optSpeechExtend', '#optSpeechLead', '#optExtendStart', '#optExtendEnd']) {
    $(id).addEventListener('input', askEdges);
  }
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
    // Ctrl+A inside the sequence list (and its Select all row) ticks every clip; elsewhere (and in text fields) it keeps its meaning
    if ((e.ctrlKey || e.metaKey) && !e.altKey && k === 'a' && e.target.closest('#seqList, #seqSelAllRow')
      && !e.target.closest('input:not([type=range]):not([type=checkbox]):not([type=radio]),textarea,select,[contenteditable]')) {
      e.preventDefault();
      if (S.selected.size < S.segments.length) selectAll();
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    // text fields, the column splitters (their own arrow keys) and any dialog or popover keep their keys
    if (e.target.closest('input:not([type=range]),textarea,select,[role=menu],[role=separator],dialog,[popover]')) return;
    if (!$('#filterPop').hidden && e.target.closest('#filterWrap')) return; // a plain div, and its button stays outside it
    const single = S.singleKeys; // Settings > Single-key shortcuts: off, M [ ] I O do nothing (WCAG 2.1.4)
    if (single && (e.key === '[' || e.key === ']' || (!e.shiftKey && (k === 'i' || k === 'o')))) {
      e.preventDefault(); // [ ] and I O: the trim point at the playhead; Cut mode shows none
      if (!S.cutMode) setRange(e.key === '[' || k === 'i' ? 'start' : 'end', playhead());
    } else if (e.key === ' ' && (!e.target.closest('button,a,summary') || e.target.closest('.vol'))) {
      e.preventDefault();
      togglePlay();
    } else if (e.key === 'Escape' && S.seqPlay) {
      e.preventDefault();
      seqStop();
    } else if (single && e.key.toLowerCase() === 'm' && !e.repeat) {
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
    S.sessionFailed = true;
    renderProjectMenu();
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
  let wanted = '';
  try {
    wanted = decodeURIComponent(location.hash.slice(1));
  } catch {
    // a malformed hash such as #% names no project: open the first one
  }
  const first = S.projects.find((p) => p.id === wanted) ?? S.projects[0];
  if (first) await openProject(first.id);
  else renderAll(true);
  pollJobs();
}

main();
