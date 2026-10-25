// cutpoint editing workspace: ranked moments, trim preview, sequence, export and assemble.
import { api, startSession, whenExpired } from './api.js';
import { initNewProject, pickPath } from './picker.js';
import { $, baseName, fmt, fmtLen, h, parseTime, round2, svg } from './util.js';

const MIN_CLIP = 0.1; // seconds
const MARGIN = 10; // seconds of rail context around a moment; trims may use the whole source
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
  renaming: null, // sequence id whose label is being edited
  renameDraft: '',
  renameErr: '',
  expired: false, // the server rejected the session token: nothing polls or retries any more
};

const v = $('#video');
const seek = $('#seek');
let openDialog;
let pendingSeek = null;

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

function win(c) {
  const dur = sourceDur(c.source);
  const lo = Math.min(c.start, c.cand?.start ?? c.start);
  const hi = Math.max(c.end, c.cand?.end ?? c.end) + MARGIN;
  return { lo: Math.max(0, lo - MARGIN), hi: dur ? Math.min(dur, hi) : hi };
}

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
  clearTimeout(S.jobTimer);
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
  else if (!hl) renderUnanalyzed(empty, p.sources?.length ?? 0, p.source_error ? sourceErrorText(p) : '');
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
        const picked = S.segments.some((s) => s.id === c.id);
        return h('li', { class: `cand${picked ? ' picked' : ''}`, 'data-id': c.id, 'aria-current': c.id === S.focus ? 'true' : null },
          h('button', { type: 'button', class: 'cand-main', 'data-act': 'focus', onclick: () => focusMoment(c.id) },
            h('span', { class: 'cand-top' },
              h('span', { class: 'rank tc', text: `#${rank}` }),
              h('span', { class: 'cand-title', text: c.label || c.id }),
              c.must_include ? h('span', { class: 'tag', text: 'must include' }) : null),
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

/** Project open but not analyzed: source count (or why the folder cannot be read) and the one button that starts analysis. */
function renderUnanalyzed(el, n, error) {
  const state = `${n}|${error}`;
  if (el.dataset.n === state) return; // polling re-renders; keep the button (and its focus)
  el.dataset.n = state;
  setEmpty(el,
    folderArt(),
    error
      ? h('p', { class: 'fresh-error', text: error })
      : h('p', { class: `fresh-count${n ? '' : ' none'}` },
        n ? `${n} video${n === 1 ? '' : 's'} found in directory` : 'No videos found in directory'),
    h('h3', { text: 'No moments analyzed yet' }),
    h('p', { text: 'Analyze footage to transcribe audio and rank the strongest moments.' }),
    h('label', { class: 'check' },
      h('input', { type: 'checkbox', id: 'optVision', 'aria-describedby': 'optVisionHint', checked: S.project?.vision_default === true }),
      'Rate frames with a local vision model (Ollama)'),
    h('p', { id: 'optVisionHint', class: 'hint', text: 'Optional. Slower, and uses several GB of GPU memory while it runs; needs Ollama with qwen3-vl:8b.' }),
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
        await startJob('analyze', { vision: $('#optVision').checked });
        if (![...S.tracked.values()].some(live)) renderBusy(); // it failed: enable again
      },
    }, 'Analyze footage'),
    h('button', {
      type: 'button', id: 'revealBtn', class: 'btn outline', hidden: !S.canReveal,
      onclick: (e) => revealFolder(e.currentTarget),
    }, icon('folder'), 'Show folder'),
    h('p', { id: 'analyzeWhy', class: 'fresh-why', 'aria-live': 'polite' }),
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

function focusMoment(id) {
  S.focus = id;
  for (const li of [...$('#candList').children, ...$('#seqList').children, ...$('#removedList').children]) {
    if (li.dataset.id === id) li.setAttribute('aria-current', 'true');
    else li.removeAttribute('aria-current');
  }
  renderStage(true);
}

// ---------- rendering: stage ----------

function trimField(which, label) {
  const input = h('input', {
    id: `t-${which}`, class: 'tc', inputMode: 'decimal', autocomplete: 'off', spellcheck: false,
    onchange: () => {
      const ok = setRange(which, parseTime(input.value));
      if (ok) seekTo(which === 'start' ? current().start : current().end);
    },
  });
  const nudge = (d) => () => {
    const c = current();
    if (c && setRange(which, c[which] + d)) seekTo(current()[which]);
  };
  const key = which === 'start' ? '[' : ']';
  $(`#${which}Field`).append(
    h('label', { for: input.id, text: label }),
    input,
    h('div', { class: 'stepper' },
      h('button', { type: 'button', class: 'btn small', 'aria-label': `-0.5 s, ${label} earlier by half a second`, onclick: nudge(-0.5), text: '-0.5 s' }),
      h('button', {
        type: 'button', class: 'btn small accent', 'aria-label': `Use playhead as ${label}`, title: `Use playhead as ${label} "${key}"`,
        'aria-keyshortcuts': key, onclick: () => setRange(which, v.currentTime), text: 'Use playhead',
      }),
      h('button', { type: 'button', class: 'btn small', 'aria-label': `+0.5 s, ${label} later by half a second`, onclick: nudge(0.5), text: '+0.5 s' })),
  );
  return input;
}

const trimInputs = {};

function setTrimMsg(text, bad = false) {
  const m = $('#trimMsg');
  m.textContent = text;
  m.classList.toggle('error', bad);
}

function setRange(which, t) {
  const c = current();
  if (!c) return false;
  t = round2(t);
  const s = which === 'start' ? t : c.start;
  const e = which === 'end' ? t : c.end;
  const err = validate(s, e, sourceDur(c.source));
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
  renderTrim(true);
  return true;
}

function renderStage(seekToStart) {
  const c = current();
  $('#stageBody').hidden = !c;
  $('#toggleBtn').hidden = !c;
  $('#stageMeta').textContent = c ? `${c.label} from ${baseName(c.source)}` : '';
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
  const inSeq = !!c.seg;
  $('#toggleBtn').textContent = inSeq ? 'Remove from sequence' : 'Add to sequence';
  $('#transcript').textContent = c.cand?.transcript || 'No speech was transcribed for this moment.';
  const sig = c.cand?.signals;
  const bits = sig
    ? Object.entries(sig).filter(([, n]) => typeof n === 'number' && n > 0).map(([k, n]) => `${k} ${n.toFixed(2)}`)
    : [];
  const kw = sig?.keywords?.length ? `Keywords: ${sig.keywords.join(', ')}. ` : '';
  $('#signals').textContent = c.cand ? `${kw}Score ${c.cand.score.toFixed(2)}${bits.length ? `. Signals: ${bits.join(', ')}` : ''}` : '';
  ensureVideo(seekToStart ? c : null);
  renderTrim(!!seekToStart);
}

function renderTrim(force = false) {
  const c = current();
  if (!c) return;
  const w = win(c);
  $('#winLo').textContent = fmt(w.lo);
  $('#winHi').textContent = fmt(w.hi);
  seek.min = w.lo;
  seek.max = w.hi;
  const pct = (t) => `${(((t - w.lo) / (w.hi - w.lo || 1)) * 100).toFixed(2)}%`;
  Object.assign($('#railSel').style, { left: pct(c.start), width: `${(((c.end - c.start) / (w.hi - w.lo || 1)) * 100).toFixed(2)}%` });
  const o = $('#railOrig');
  o.hidden = !c.cand;
  if (c.cand) Object.assign(o.style, { left: pct(c.cand.start), width: `${(((c.cand.end - c.cand.start) / (w.hi - w.lo || 1)) * 100).toFixed(2)}%` });
  for (const which of ['start', 'end']) {
    const input = trimInputs[which];
    if (force || document.activeElement !== input) input.value = fmt(c[which]);
  }
  const changedFromOrig = c.cand && (c.start !== c.cand.start || c.end !== c.cand.end);
  $('#resetBtn').hidden = !changedFromOrig;
  setTrimMsg(`Clip length ${(c.end - c.start).toFixed(2)} s${c.cand ? `, analyzed as ${fmt(c.cand.start)} to ${fmt(c.cand.end)}` : ''}.`);
  updatePlayhead();
}

// ---------- video ----------

function screenMsg(text) {
  $('#screenMsg').textContent = text;
  $('#screenMsg').hidden = !text;
}

// Source proxies are made on demand: only the focused source is prepared, and only it is polled.
const prepareRequested = new Set();

async function prepareSource(src) {
  const k = `${S.id}:${src.path}`;
  if (prepareRequested.has(k)) return;
  prepareRequested.add(k);
  const sid = src.id ?? decodeURIComponent(src.url.split('?')[0].split('/').pop());
  try {
    await api(`/api/projects/${encodeURIComponent(S.id)}/sources/${encodeURIComponent(sid)}/prepare`, { method: 'POST' });
    refreshProject();
  } catch (e) {
    prepareRequested.delete(k);
    fail(e);
  }
}

function ensureVideo(seekC) {
  clearTimeout(S.prepTimer);
  if (S.expired) return;
  $('#retryBtn').hidden = true;
  const c = current();
  const src = c && srcOf(c.source);
  if (!src) {
    drop('This source file is not part of the project, so it cannot be previewed. You can still trim by timecode.');
    return;
  }
  if (src.status === 'unknown') {
    // not probed yet (the project view never probes): preparing probes it and refreshes. Still
    // unknown after an earlier prepare (the file changed, or the server restarted): prepare again.
    const k = `${S.id}:${src.path}`;
    if (!prepareRequested.has(k)) prepareSource(src);
    else
      S.prepTimer = setTimeout(() => {
        prepareRequested.delete(k);
        prepareSource(src);
      }, 2000);
    return drop('Loading video…');
  }
  if (src.status === 'needs-proxy' || src.status === 'preparing') {
    const k = `${S.id}:${src.path}`;
    if (src.status === 'needs-proxy' && !prepareRequested.has(k)) prepareSource(src);
    else if (src.status === 'needs-proxy')
      // requested before, but not building (the server restarted): prepare again
      S.prepTimer = setTimeout(() => {
        prepareRequested.delete(k);
        prepareSource(src);
      }, 2000);
    else S.prepTimer = setTimeout(refreshProject, 2000);
    return drop('Preparing a browser-playable preview of this source. It will start when ready.');
  }
  if (src.status === 'failed') {
    $('#retryBtn').hidden = false;
    return drop('The browser-playable preview could not be built for this source. Retry it, or trim by timecode.');
  }
  prepareRequested.delete(`${S.id}:${src.path}`); // ready: a later change to the file prepares again
  if (v.dataset.url !== src.url) {
    v.dataset.url = src.url;
    pendingSeek = seekC ? seekC.start : (c?.start ?? 0);
    screenMsg('Loading video…');
    v.src = src.url;
    showPlaying(false); // a new src stops playback without a pause event
  } else if (seekC) seekTo(seekC.start);
}

/** Stop a video element and release its file. */
function unload(el) {
  el.pause();
  el.removeAttribute('src');
  delete el.dataset.url;
  el.load();
}

function drop(msg) {
  unload(v);
  screenMsg(msg);
}

function seekTo(t) {
  const c = current();
  if (!c) return;
  const w = win(c);
  v.currentTime = Math.min(Math.max(t, w.lo), w.hi);
  updatePlayhead();
}

function updatePlayhead() {
  const c = current();
  const t = v.currentTime || 0;
  seek.value = t;
  $('#clock').textContent = c ? `${fmt(t)} in source` : '';
}

function tick() {
  if (v.paused) return;
  const c = current();
  if (c && v.currentTime >= c.end) {
    v.pause();
    v.currentTime = c.end;
  }
  updatePlayhead();
  requestAnimationFrame(tick);
}

function togglePlay() {
  const c = current();
  if (!c || !v.src) return;
  if (!v.paused) return v.pause();
  if (v.currentTime >= c.end - 0.05 || v.currentTime < win(c).lo) v.currentTime = c.start;
  v.play().catch(() => {});
}

v.addEventListener('loadedmetadata', () => {
  if (pendingSeek != null) seekTo(pendingSeek);
  pendingSeek = null;
});
v.addEventListener('canplay', () => screenMsg(''));
v.addEventListener('playing', () => screenMsg(''));
v.addEventListener('waiting', () => screenMsg('Buffering…'));
v.addEventListener('error', () => {
  if (!v.dataset.url) return;
  const url = v.dataset.url;
  const msg = `The browser could not play this video${v.error?.message ? `: ${v.error.message}` : ''}.`;
  // The element hides the status; one API call tells an expired token (banner only) from a bad
  // file, so the message is shown, and announced, only when the session is still valid.
  const show = () => {
    if (!S.expired && v.dataset.url === url) screenMsg(msg);
  };
  startSession().then(show, show);
});
/** The play button is an icon; its name and tooltip say what a click does. */
function showPlaying(playing) {
  const name = playing ? 'Pause' : 'Play';
  const b = $('#playBtn');
  b.setAttribute('aria-label', name);
  b.title = `${name} (Space)`;
  b.replaceChildren(icon(playing ? 'pause' : 'play'));
}

v.addEventListener('play', () => {
  showPlaying(true);
  requestAnimationFrame(tick);
});
v.addEventListener('pause', () => {
  showPlaying(false);
  updatePlayhead();
});
v.addEventListener('seeked', updatePlayhead);
v.addEventListener('click', togglePlay);
$('#playBtn').addEventListener('click', togglePlay);
$('#retryBtn').addEventListener('click', () => {
  const c = current();
  const src = c && srcOf(c.source);
  if (!src) return;
  prepareRequested.delete(`${S.id}:${src.path}`);
  prepareSource(src);
});
seek.addEventListener('input', () => seekTo(Number(seek.value)));

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

/** "Analyze again": confirm in a dialog whose vision box starts as the last analysis asked
 * (before a first analysis: the Settings default). */
function analyzeAgain() {
  const d = $('#analyzeDialog');
  const back = document.activeElement;
  const p = S.project;
  $('#againVision').checked = p?.highlights ? !!p.vision_requested : !!p?.vision_default;
  d.returnValue = '';
  d.showModal();
  $('#againCancel').focus();
  d.addEventListener('close', () => {
    if (back?.isConnected) back.focus();
    if (d.returnValue === 'ok') startJob('analyze', { vision: $('#againVision').checked });
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
  h('span', { class: 'vh', text: `, ${range}` }));

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
    class: `seq${S.selected.has(s.id) ? ' selected' : ''}`, 'data-id': s.id, 'aria-current': s.id === S.focus ? 'true' : null,
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
  preserveFocus(() => {
    $('#seqList').replaceChildren(...segs.map((s, i) => seqRow(s, i, segs.length)));
  });
  renderBatch();
  renderRemoved();
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
    const reason = own ? 'Analyzing… progress is in Jobs.' : busy ? why
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
  return { vertical: $('#optVertical').checked, extend };
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
  $('#setVisionHint').textContent = `Off by default. Slower, and uses several GB of GPU memory while it runs; needs Ollama with ${st.vision_model}. New analyses start with this choice and can still change it per run. Saved in ${st.config_file}.`;
  d.showModal();
  input.focus();
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
  if ('vision' in body) {
    // the open project's boxes start from the new default; a first analysis's box changes now
    if (S.project) S.project.vision_default = st.vision === true;
    const opt = $('#optVision');
    if (opt) opt.checked = st.vision === true;
  }
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

function bindSettings() {
  const d = $('#settingsDialog');
  const gear = $('#settingsBtn');
  gear.append(icon('gear'));
  gear.addEventListener('click', openSettings);
  d.addEventListener('close', () => gear.focus());
  $('#settingsCancel').addEventListener('click', () => d.close());
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
  const vids = [v, $('#finalVideo')];
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
      setRange(e.key === '[' ? 'start' : 'end', v.currentTime);
    } else if (e.key === ' ' && (!e.target.closest('button,a,summary') || e.target.closest('.vol'))) {
      e.preventDefault();
      togglePlay();
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
  const wanted = decodeURIComponent(location.hash.slice(1));
  const first = S.projects.find((p) => p.id === wanted) ?? S.projects[0];
  if (first) await openProject(first.id);
  else renderAll(true);
  pollJobs();
}

main();
