// cutpoint editing workspace: ranked moments, trim preview, sequence, export and assemble.
import { api, startSession, whenExpired } from './api.js';
import { initNewProject, pickPath } from './picker.js';
import { $, baseName, fmt, fmtLen, h, parseTime, round2, svg } from './util.js';

const MIN_CLIP = 0.1; // seconds
const MARGIN = 10; // seconds: the least rail context around a moment; playback and trims use the whole source
const MIN_VIEW = 2; // seconds: the rail's deepest zoom
const LONG_CLIP = 600; // seconds: longer clips get a note under the trim fields
const LABEL_MAX = 200; // characters; ui.py validate_segments rejects longer labels
const THUMB_W = 160; // pixels requested for sequence thumbnails
const JOB_NAMES = { analyze: 'Analysis', clips: 'Clip export', assemble: 'Assembly' };
const live = (status) => status === 'running' || status === 'cancelling'; // the job still holds the server
const SIDE_KEY = 'cutpoint.sideWidth'; // localStorage: the sequence column width the user dragged to
const SIDE_MIN = 290; // pixels; the column's CSS default is 290-380
const SIDE_MAX = 640;
const STAGE_MIN = 360; // pixels the preview keeps when the sequence column widens
const SIDE_STEP = 16; // pixels per arrow key; Shift moves four steps
const VOL_KEY = 'cutpoint.volume'; // localStorage: {volume: 0-1, muted} for every preview video
const NERD_KEY = 'cutpoint.nerdStats'; // localStorage: "1" while Stats for Nerds is on (off by default)
const THEME_KEY = 'cutpoint.theme'; // localStorage: "light" or "dark"; absent for Blue, the default. index.html reads it too
const THEMES = ['blue', 'light', 'dark'];
const FRAME = 0.04; // seconds: "Play sequence" moves on this close to a clip's end
const GAP = 0.05; // seconds: a clip starting this close to the previous one's end plays on without a cut

const S = {
  styles: [],
  canReveal: false, // the server can open a folder in this machine's file manager
  canPick: false, // the server can show this machine's own open dialog
  projects: [],
  id: null,
  project: null,
  segments: [], // working sequence (may hold unsaved edits)
  savedKey: '[]',
  drafts: {}, // trims of moments not (yet) in the sequence; whole segments for removed non-moments; ``at``: index a removed clip had
  focus: null,
  filter: '',
  jobs: [],
  active: null, // the running job, if any
  tracked: new Map(), // job id -> last seen status
  logOpen: false,
  saving: false,
  selected: new Set(), // sequence ids ticked for batch actions
  nerdStats: false, // Settings > Stats for Nerds: the scoring overlay on the preview
  nerdStatsWas: false, // nerdStats when Settings opened, restored by Cancel or Escape
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
};

// Two stage players: the visible one (``v``) plays, the other preloads the next clip of "Play sequence".
const els = [$('#video'), $('#video2')];
let v = els[0];
const spare = () => (v === els[0] ? els[1] : els[0]);
const seek = $('#seek');
let openDialog;
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
function playable(src, t) {
  const dur = sourceDur(src.path) ?? (v.dataset.url === src.url && Number.isFinite(v.duration) ? v.duration : null);
  if (src.status === 'ready') return { url: src.url, offset: 0, lo: 0, hi: dur };
  const w = (src.windows ?? []).filter((x) => t != null && covers(x, t, dur)).sort((a, b) => b.hi - a.hi)[0];
  return w ? { url: w.url, offset: w.lo, lo: w.lo, hi: w.hi } : null;
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
  if (r && ofSource(src)) {
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
  return src?.status === 'ready' ? playable(src, 0).hi : sourceDur(c.source);
};
/** The source time at the stage player's 0: a window proxy starts later in its source. */
const offsetOf = () => Number(v.dataset.offset) || 0;
/** The source range the stage player's file covers; ``hi`` null: to the end of the source. Null when nothing is loaded. */
const loaded = () => (v.dataset.url ? { lo: offsetOf(), hi: v.dataset.hi ? Number(v.dataset.hi) : null } : null);
/** The stage player holds a file of ``src``: its original or whole proxy, or one of its window proxies. */
const ofSource = (src) => !!v.dataset.url && (v.dataset.url === src.url || v.dataset.url.startsWith(`${src.url.split('?')[0]}/window/`));
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
  renderAll(switched);
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

  const q = S.filter.trim().toLowerCase();
  const max = Math.max(...list.map((c) => c.score), 1e-9);
  const multi = new Set(list.map((c) => c.source)).size > 1;
  const items = list
    .map((c, i) => ({ c, rank: i + 1 }))
    .filter(({ c }) => !q || `${c.label} ${c.transcript ?? ''} ${baseName(c.source)}`.toLowerCase().includes(q));
  preserveFocus(() => {
    $('#candList').replaceChildren(
      ...items.map(({ c, rank }) => {
        const seg = S.segments.find((s) => s.id === c.id);
        const picked = !!seg;
        return h('li', { class: `cand${picked ? ' picked' : ''}`, 'data-id': c.id, 'aria-current': c.id === S.focus ? 'true' : null },
          h('button', { type: 'button', class: 'cand-main', 'data-act': 'focus', onclick: () => focusMoment(c.id) },
            h('span', { class: 'cand-top' },
              h('span', { class: 'rank tc', text: `#${rank}` }),
              h('span', { class: 'cand-title', text: c.label || c.id }),
              c.must_include ? h('span', { class: 'tag', text: 'must include' }) : null,
              prepBadge(srcOf(c.source), seg ?? { ...c, ...S.drafts[c.id] })),
            h('span', { class: 'cand-time tc' },
              `${fmt(c.start)} to ${fmt(c.end)}, ${(c.end - c.start).toFixed(1)} s`,
              multi ? ` in ${baseName(c.source)}` : ''),
            h('span', { class: 'score', title: `Score ${c.score.toFixed(3)}` },
              h('span', { style: `width:${Math.max(2, (c.score / max) * 100)}%` })),
            c.transcript ? h('span', { class: 'cand-text', text: c.transcript }) : null),
          h('button', {
            type: 'button', class: 'btn small', 'data-act': 'toggle',
            'aria-label': `${picked ? 'Remove' : 'Add'} ${c.label || c.id} ${picked ? 'from' : 'to'} sequence`,
            onclick: () => (picked ? removeSeg(c.id) : addSeg(c.id)),
          }, picked ? 'Remove' : 'Add'),
        );
      }),
    );
  });
  if (list.length && !items.length)
    $('#candList').replaceChildren(h('li', { class: 'empty', text: 'No moments match this filter.' }));
  if (hadFocus && !empty.contains(document.activeElement))
    ([$('#analyzeBig'), $('#candList button'), $('#analyzeBtn'), $('#revealHeadBtn')].find((b) => b && !b.hidden && !b.disabled) ?? $('#momentsTitle')).focus();
  renderBusy();
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

/** Project open but not analyzed: source count (or why the folder cannot be read) and the one button that starts analysis; ``pano``: a 360 project (n counts its clips). */
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
          ? (n ? `${n} 360 clip${n === 1 ? '' : 's'} found` : 'No 360 clips found')
          : (n ? `${n} video${n === 1 ? '' : 's'} found in directory` : 'No videos found in directory')),
    h('h3', { text: 'No moments analyzed yet' }),
    h('p', {
      text: pano
        ? 'Analyze footage to read the 360 clips, transcribe audio and rank the strongest moments. Clips are reframed when you export them. Vision rating is not used for 360 footage.'
        : 'Analyze footage to transcribe audio and rank the strongest moments.',
    }),
    h('button', {
      type: 'button', id: 'analyzeBig', class: 'btn primary', 'aria-describedby': 'analyzeWhy analyzeNote',
      onclick: async (e) => {
        e.currentTarget.disabled = true; // no second start while the request is in flight
        el.dataset.refocus = '1';
        // only until the user moves on: their next click or focus change ends it
        const stop = new AbortController();
        const drop = () => { delete el.dataset.refocus; stop.abort(); };
        document.addEventListener('pointerdown', drop, { capture: true, signal: stop.signal });
        document.addEventListener('focusin', drop, { capture: true, signal: stop.signal });
        await startJob('analyze'); // vision: the server applies the Settings value
        if (![...S.tracked.values()].some(live)) renderBusy(); // it failed: enable again
      },
    }, 'Analyze footage'),
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
  const gaps = [];
  if (src && ['needs-proxy', 'preparing'].includes(src.status)) {
    const wins = [...(src.windows ?? [])].sort((a, b) => a.lo - b.lo);
    const end = dur || Math.max(hi, ...wins.map((w) => w.hi));
    let at = 0;
    for (const w of wins) {
      if (w.lo > at) gaps.push([at, w.lo]);
      at = Math.max(at, w.hi);
    }
    if (at < end) gaps.push([at, end]);
  }
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
  seekWant = null;
  pendingSeek = t;
  screenMsg('Loading video…');
  v.src = p.url;
  waitFor = null;
  if (resumeAt && src.path !== resumeAt.source) resumeAt = null; // another source: no resume
  showPlaying(!!resumeAt); // a new src stops playback without a pause event; a resume plays on
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
    if (fits && ((mine && inLoaded(at, durOf(c))) || p)) {
      if (src.status === 'needs-proxy' && !prepareRequested.has(k)) prepareSource(src); // no window request brings it
      if (!mine || !inLoaded(at, durOf(c))) loadFile(src, p, at);
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
  // A window proxy playing on switches to the whole proxy at the next pause or seek.
  const playingOn = v.dataset.hi && !v.paused && !v.error && inLoaded(want ?? playhead(c), durOf(c));
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

onStage('play', () => {
  resumeAt = null;
  showPlaying(true);
  follow(playhead()); // Play shows the playhead, also after a pan away from it
  loop();
});
onStage('pause', () => {
  showPlaying(!!resumeAt); // waiting at a window's end to play on
  updatePlayhead();
  const src = current() && srcOf(current().source);
  if (v.dataset.hi && src?.status === 'ready' && !S.seqPlay) ensureVideo(null); // the whole proxy is built: switch to it now
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
  keepDraft(S.segments[at], at);
  S.segments.splice(at, 1);
  renderAfterSeqChange();
}

function moveSeg(id, to) {
  const from = S.segments.findIndex((s) => s.id === id);
  to = Math.max(0, Math.min(S.segments.length - 1, to));
  if (from < 0 || from === to) return;
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
  clock: [['path', { d: 'M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm.5-13H11v6l5.25 3.15.75-1.23-4.5-2.67z' }]],
};
const icon = (name) => svg(`icon icon-${name}`, ICONS[name]);

// Thumbnails are stills the server cuts (and caches) on request; remember how each URL went so a
// re-render neither flashes the skeleton again nor retries a source that cannot be read.
const thumbs = new Map(); // url -> 'ok' | 'failed'
// New stills load a few at a time so they cannot take every connection the player's range requests need.
const THUMB_LOADS = 3;
const thumbQueue = []; // <img> elements waiting for their data-src
let thumbLoading = 0;

function pumpThumbs() {
  while (thumbLoading < THUMB_LOADS && thumbQueue.length) {
    const img = thumbQueue.shift();
    if (!img.isConnected) continue; // a re-render replaced it before its turn
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

/** "Analyze again": confirm in a dialog; vision follows Settings, as for a first analysis. */
function analyzeAgain() {
  const d = $('#analyzeDialog');
  const back = document.activeElement;
  d.returnValue = '';
  d.showModal();
  $('#againCancel').focus();
  d.addEventListener('close', () => {
    if (back?.isConnected) back.focus();
    if (d.returnValue === 'ok') startJob('analyze');
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
  const pad = Number($('#optExtend').value) || 0;
  $('#seqTotal').textContent = segs.length
    ? `${segs.length} clip${segs.length > 1 ? 's' : ''}, ${fmtLen(total)}${pad > 0 ? ` (about ${fmtLen(total + 2 * pad * segs.length)} padded)` : ''}`
    : '';
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
  if (!p) st.textContent = '';
  else if (!S.segments.length) st.textContent = 'Add moments to the sequence to export them.';
  else if (!clips) st.textContent = 'Nothing exported yet. Exporting saves the sequence first.';
  else if (exportMatches()) st.textContent = `Latest export (${clips.dir ?? 'folder'}) has ${clips.clips.length} clips and matches this sequence.`;
  else st.textContent = `Latest export (${clips.dir ?? 'folder'}) has ${clips.clips.length} clips and differs from this sequence. Export again to update it.`;
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
    const label = own ? 'Analyzing…' : 'Analyze footage';
    if (big.textContent !== label) big.textContent = label;
    const out = $('#analyzeWhy');
    out.classList.toggle('progress', own);
    if (out.textContent !== big.title) out.textContent = big.title; // live region: announce changes only
  }
  set('#clipsBtn', !p || busy || !S.segments.length, busy ? why : 'Add moments to the sequence first.');
  set('#assembleBtn', !p || busy || !p.clips, busy ? why : 'Export clips first.');
  // Never disabled by a job: showing a folder is harmless. Without a project both are hidden too,
  // so the fieldsets being disabled below never catches them.
  $('#revealClipsBtn').hidden = !p?.clips || !S.canReveal;
  $('#revealOutBtn').hidden = !p?.outputs?.length || !S.canReveal;
  $('#exportPanel').disabled = !p;
  $('#assemblePanel').disabled = !p;
  renderSteps();
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

function exportOptions() {
  const extend = Number($('#optExtend').value);
  if (!Number.isFinite(extend) || extend < 0 || extend > 30) throw new Error('Padding must be between 0 and 30 seconds.');
  return { vertical: $('#optVertical').checked, extend, compress_peaks: $('#optPeaks').value };
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
  $('#settingsError').textContent = '';
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
  $('#setNerdStats').checked = S.nerdStats;
  S.nerdStatsWas = S.nerdStats; // Cancel or Escape goes back to this
  S.themeWas = $('#setTheme').value = theme();
  d.showModal();
  // a locked folder cannot be edited: start on the first control that can
  (st.projects_dir_locked ? box : input).focus();
}

async function saveSettings() {
  const input = $('#setProjects');
  const box = $('#setVision');
  const err = $('#settingsError');
  const value = input.value.trim();
  err.textContent = '';
  const body = {};
  if (!input.readOnly) {
    if (!value) return (err.textContent = 'Enter the full path of a folder.');
    if (value !== input.dataset.saved) body.projects_dir = value;
  }
  if (String(box.checked) !== box.dataset.saved) body.vision = box.checked;
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
  }
  input.value = input.dataset.saved = st.projects_dir;
  box.dataset.saved = String(st.vision === true);
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
  renderMoments();
  renderStage(seekToStart);
  renderSequence();
  renderSave();
  renderExport();
  renderOutputs();
}

/**
 * The separator on the sequence column's left edge: drag, arrow keys (Home/End for the limits) or
 * double-click to reset. The width is remembered per browser; without storage it lasts the session.
 */
function initSplitter() {
  const bar = $('#sideSplit');
  const ws = $('#workspace');
  const side = $('#side');
  let pref = null; // dragged width in pixels, or null for the CSS default
  try {
    pref = Number(localStorage.getItem(SIDE_KEY)) || null;
  } catch {
    // storage blocked: start from the default
  }
  const store = () => {
    try {
      if (pref == null) localStorage.removeItem(SIDE_KEY);
      else localStorage.setItem(SIDE_KEY, String(pref));
    } catch {
      // storage blocked or full: the width still applies until reload
    }
  };
  const max = () => Math.max(SIDE_MIN, Math.min(SIDE_MAX, Math.floor(innerWidth / 2),
    ws.clientWidth - $('.moments').offsetWidth - STAGE_MIN));
  const clamp = (w, hi = max()) => Math.round(Math.min(Math.max(w, SIDE_MIN), hi));
  // pref stays as chosen; only what is applied is clamped, so a wider window gets it back.
  // Layout is read once, before the write: reading it after would force a synchronous relayout.
  const apply = () => {
    const hi = max();
    const w = pref == null ? null : clamp(pref, hi);
    if (w == null) ws.style.removeProperty('--side-w');
    else ws.style.setProperty('--side-w', `${w}px`);
    bar.setAttribute('aria-valuemin', SIDE_MIN);
    bar.setAttribute('aria-valuemax', hi);
    bar.setAttribute('aria-valuenow', w ?? side.offsetWidth);
  };
  const set = (w) => {
    pref = w == null ? null : clamp(w);
    apply();
  };

  bar.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    bar.setPointerCapture(e.pointerId);
    bar.classList.add('dragging');
    const x0 = e.clientX;
    const w0 = side.offsetWidth;
    // At most one resize per frame, however fast the pointer reports.
    let next = null;
    let frame = 0;
    const flush = () => {
      frame = 0;
      if (next != null) set(next);
      next = null;
    };
    const move = (ev) => {
      next = w0 + x0 - ev.clientX; // dragging left widens the column
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
    const step = e.shiftKey ? SIDE_STEP * 4 : SIDE_STEP;
    const w = side.offsetWidth;
    const next = { ArrowLeft: w + step, ArrowRight: w - step, Home: SIDE_MIN, End: max() }[e.key];
    if (next == null) return;
    e.preventDefault();
    set(next);
    store();
  });
  bar.addEventListener('dblclick', () => {
    set(null);
    store();
  });
  addEventListener('resize', apply);
  apply();
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

function bind() {
  initSplitter();
  const toggleMute = initVolume();
  bindSettings();
  initNerdStats();
  showPlaying(false);
  trimInputs.start = trimField('start', 'Clip Start');
  trimInputs.end = trimField('end', 'Clip End');
  $('#trimMsg').after(
    h('button', { type: 'button', class: 'btn small', id: 'resetBtn', hidden: true, onclick: () => {
      const c = current();
      if (!c?.cand) return;
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
  $('#optExtend').addEventListener('input', renderSequence);
  $('#seqPlayBtn').addEventListener('click', () => (S.seqPlay ? seqStop() : seqStart()));
  $('#seqRemoveSel').addEventListener('click', removeSelected);
  $('#seqClearSel').addEventListener('click', () => {
    S.selected.clear();
    renderBatch();
    focusRow(0, 'select');
  });
  $('#analyzeBtn').addEventListener('click', analyzeAgain);
  $('#revealHeadBtn').prepend(icon('folder'));
  $('#revealHeadBtn').addEventListener('click', (e) => revealFolder(e.currentTarget));
  for (const [id, target] of [['#revealClipsBtn', 'clips'], ['#revealOutBtn', 'outputs']]) {
    $(id).prepend(icon('folder'));
    $(id).addEventListener('click', (e) => revealFolder(e.currentTarget, target));
  }
  $('#clipsBtn').addEventListener('click', () => {
    try {
      startJob('clips', exportOptions());
    } catch (e) {
      fail(e);
    }
  });
  $('#assembleBtn').addEventListener('click', () =>
    startJob('assemble', {
      transition: $('#optTransition').value,
      loudnorm: $('#optLoudnorm').checked,
      compress_peaks: $('#optPeaks').value,
      captions: $('#optCaptions').checked,
    }),
  );
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

  addEventListener('keydown', (e) => {
    if (e.defaultPrevented || $('#newDialog').open || $('#confirmDialog').open || $('#analyzeDialog').open || $('#settingsDialog').open || popOpen()) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
      e.preventDefault();
      saveReview().catch(fail);
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    // text fields, the column splitter (its own arrow keys) and any dialog or popover keep their keys
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
    // compress_peaks in config.json is the default; the select keeps Off if it cannot be read
    $('#optPeaks').value = (await api('/api/settings')).compress_peaks ?? 'off';
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
