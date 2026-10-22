// cutpoint editing workspace: ranked moments, trim preview, sequence, export and assemble.
import { api, startSession } from './api.js';
import { initNewProject } from './picker.js';
import { $, baseName, fmt, fmtLen, h, parseTime, round2 } from './util.js';

const MIN_CLIP = 0.1; // seconds
const MARGIN = 10; // seconds of rail context around a moment; trims may use the whole source
const JOB_NAMES = { analyze: 'Analysis', clips: 'Clip export', assemble: 'Assembly' };

const S = {
  styles: [],
  projects: [],
  id: null,
  project: null,
  segments: [], // working sequence (may hold unsaved edits)
  savedKey: '[]',
  drafts: {}, // trims of moments not (yet) in the sequence
  focus: null,
  filter: '',
  jobs: [],
  active: null, // the running job, if any
  tracked: new Map(), // job id -> last seen status
  logOpen: false,
  saving: false,
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
  const base = seg ?? (cand && { ...cand, ...S.drafts[id] });
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
  const b = $('#banner');
  b.textContent = msg || '';
  b.hidden = !msg;
};
const fail = (e) => banner(e.message || String(e));

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
  S.projects = (await api('/api/projects')).projects;
  const sel = $('#projectSelect');
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

function adopt(p, switched) {
  S.project = p;
  if (switched) {
    S.id = p.id;
    S.drafts = {};
    S.focus = null;
    S.segments = structuredClone(p.review?.segments ?? []);
    S.savedKey = key();
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
  $('#analyzeBtn').hidden = !p;
  $('#analyzeBtn').textContent = hl ? 'Analyze again' : 'Analyze footage';
  $('#filterWrap').hidden = !list.length;
  $('#momentsCount').textContent = hl ? `${list.length} found, ${hl.style ?? p.style} style` : '';
  if (!p) setEmpty($('#momentsEmpty'));
  else if (!hl)
    setEmpty($('#momentsEmpty'), h('p', {}, 'This project has not been analyzed yet. Analysis transcribes the audio and ranks the strongest moments. It can take a while on long footage.'));
  else if (!list.length)
    setEmpty($('#momentsEmpty'), h('p', {}, 'Analysis found no moments. Try another style, or add footage and analyze again.'));
  else setEmpty($('#momentsEmpty'));

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
  renderBusy();
}

function focusMoment(id) {
  S.focus = id;
  for (const li of [...$('#candList').children, ...$('#seqList').children]) {
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
  $(`#${which}Field`).append(
    h('label', { for: input.id, text: label }),
    h('div', { class: 'stepper' },
      h('button', { type: 'button', class: 'btn small', 'aria-label': `${label} earlier by half a second`, onclick: nudge(-0.5), text: '-0.5 s' }),
      input,
      h('button', { type: 'button', class: 'btn small', 'aria-label': `${label} later by half a second`, onclick: nudge(0.5), text: '+0.5 s' })),
    h('button', { type: 'button', class: 'btn small linkish', onclick: () => setRange(which, v.currentTime), text: 'Use playhead' }),
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
  } else S.drafts[c.id] = { start: s, end: e };
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
  $('#retryBtn').hidden = true;
  const c = current();
  const src = c && srcOf(c.source);
  if (!src) {
    drop('This source file is not part of the project, so it cannot be previewed. You can still trim by timecode.');
    return;
  }
  if (src.status === 'needs-proxy' || src.status === 'preparing') {
    if (src.status === 'needs-proxy' && !prepareRequested.has(`${S.id}:${src.path}`)) prepareSource(src);
    else S.prepTimer = setTimeout(refreshProject, 2000);
    return drop('Preparing a browser-playable preview of this source. It will start when ready.');
  }
  if (src.status === 'failed') {
    $('#retryBtn').hidden = false;
    return drop('The browser-playable preview could not be built for this source. Retry it, or trim by timecode.');
  }
  if (v.dataset.url !== src.url) {
    v.dataset.url = src.url;
    pendingSeek = seekC ? seekC.start : (c?.start ?? 0);
    screenMsg('Loading video…');
    v.src = src.url;
  } else if (seekC) seekTo(seekC.start);
}

function drop(msg) {
  v.pause();
  v.removeAttribute('src');
  delete v.dataset.url;
  v.load();
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
  if (v.dataset.url) screenMsg(`The browser could not play this video${v.error?.message ? `: ${v.error.message}` : ''}.`);
});
v.addEventListener('play', () => {
  $('#playBtn').textContent = 'Pause';
  requestAnimationFrame(tick);
});
v.addEventListener('pause', () => {
  $('#playBtn').textContent = 'Play';
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
  const c = cands().find((x) => x.id === id);
  if (!c || S.segments.some((s) => s.id === id)) return;
  const d = S.drafts[id];
  S.segments.push({ id, source: c.source, start: d?.start ?? c.start, end: d?.end ?? c.end, label: c.label || id });
  delete S.drafts[id];
  renderAfterSeqChange();
}

function removeSeg(id) {
  const seg = S.segments.find((s) => s.id === id);
  if (!seg) return;
  S.drafts[id] = { start: seg.start, end: seg.end }; // keep the trim if re-added
  S.segments = S.segments.filter((s) => s !== seg);
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
  preserveFocus(() => {
    $('#seqList').replaceChildren(
      ...segs.map((s, i) =>
        h('li', {
          class: 'seq', 'data-id': s.id, draggable: true, 'aria-current': s.id === S.focus ? 'true' : null,
          ondragstart: (e) => {
            dragId = s.id;
            e.dataTransfer.effectAllowed = 'move';
            e.dataTransfer.setData('text/plain', s.id);
          },
          ondragend: () => {
            dragId = null;
            for (const li of $('#seqList').children) li.classList.remove('drop');
          },
          ondragover: (e) => {
            if (!dragId) return;
            e.preventDefault();
            for (const li of $('#seqList').children) li.classList.toggle('drop', li.dataset.id === s.id && s.id !== dragId);
          },
          ondrop: (e) => {
            e.preventDefault();
            if (dragId) moveSeg(dragId, i);
          },
        },
          h('button', { type: 'button', class: 'seq-main', 'data-act': 'focus', onclick: () => focusMoment(s.id) },
            h('span', { class: 'rank tc', text: String(i + 1) }),
            h('span', { class: 'seq-title', text: s.label || s.id }),
            h('span', { class: 'cand-time tc', text: `${fmt(s.start)} to ${fmt(s.end)}, ${(s.end - s.start).toFixed(1)} s` })),
          h('span', { class: 'seq-actions' },
            h('button', { type: 'button', class: 'btn small', 'data-act': 'up', disabled: i === 0, 'aria-label': `Move ${s.label || s.id} up`, onclick: () => moveSeg(s.id, i - 1), text: 'Up' }),
            h('button', { type: 'button', class: 'btn small', 'data-act': 'down', disabled: i === segs.length - 1, 'aria-label': `Move ${s.label || s.id} down`, onclick: () => moveSeg(s.id, i + 1), text: 'Down' }),
            h('button', { type: 'button', class: 'btn small', 'data-act': 'remove', 'aria-label': `Remove ${s.label || s.id} from sequence`, onclick: () => removeSeg(s.id), text: 'Remove' })),
        ),
      ),
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
  set('#clipsBtn', !p || busy || !S.segments.length, busy ? why : 'Add moments to the sequence first.');
  set('#assembleBtn', !p || busy || !p.clips, busy ? why : 'Export clips first.');
  $('#exportPanel').disabled = !p;
  $('#assemblePanel').disabled = !p;
}

function renderOutputs() {
  const outs = S.project?.outputs ?? [];
  const final = outs.find((o) => /^final.*\.mp4$/i.test(o.name));
  $('#outputs').hidden = !outs.length;
  $('#outTitle').textContent = final ? 'Final video' : 'Outputs';
  const fv = $('#finalVideo');
  fv.hidden = !final;
  if (final && fv.dataset.url !== final.url) {
    fv.dataset.url = final.url;
    fv.src = final.url;
  } else if (!final && fv.dataset.url) {
    fv.pause();
    fv.removeAttribute('src');
    delete fv.dataset.url;
    fv.load();
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
    const finished = jobs.filter((j) => S.tracked.get(j.id) === 'running' && j.status !== 'running');
    const fresh = jobs.some((j) => j.status === 'running' && !S.tracked.has(j.id));
    for (const j of jobs) S.tracked.set(j.id, j.status);
    S.jobs = jobs;
    S.active = jobs.find((j) => j.status === 'running') ?? null;
    if (fresh) S.logOpen = true;
    if (finished.some((j) => j.status === 'failed')) S.logOpen = true;
    renderJobs();
    renderBusy();
    if (finished.some((j) => j.project_id === S.id)) refreshProject();
  } catch (e) {
    $('#jobStatus').textContent = e.message;
    $('#jobStatus').dataset.status = 'failed';
  }
  S.jobTimer = setTimeout(pollJobs, S.active ? 1000 : 5000);
}

function jobLine(j) {
  const name = S.projects.find((p) => p.id === j.project_id)?.name ?? j.project_id;
  const n = JOB_NAMES[j.action] ?? j.action;
  if (j.status === 'running') return `${n} running for ${name}`;
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
  cancel.disabled = false;
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

// ---------- wiring ----------

function renderAll(seekToStart) {
  renderMoments();
  renderStage(seekToStart);
  renderSequence();
  renderSave();
  renderExport();
  renderOutputs();
}

function bind() {
  trimInputs.start = trimField('start', 'Start');
  trimInputs.end = trimField('end', 'End');
  $('#trimMsg').after(
    h('button', { type: 'button', class: 'btn small', id: 'resetBtn', hidden: true, onclick: () => {
      const c = current();
      if (!c?.cand) return;
      if (c.seg) {
        c.seg.start = c.cand.start;
        c.seg.end = c.cand.end;
        changed();
      } else delete S.drafts[c.id];
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
    const c = current();
    if (c) (c.seg ? removeSeg : addSeg)(c.id);
  });
  $('#optExtend').addEventListener('input', renderSequence);
  $('#analyzeBtn').addEventListener('click', () => startJob('analyze'));
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
    if (e.defaultPrevented || $('#newDialog').open) return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
      e.preventDefault();
      saveReview().catch(fail);
      return;
    }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target.closest('input:not([type=range]),textarea,select')) return;
    if (e.key === '[' || e.key === ']') {
      e.preventDefault();
      setRange(e.key === '[' ? 'start' : 'end', v.currentTime);
    } else if (e.key === ' ' && !e.target.closest('button,a,summary')) {
      e.preventDefault();
      togglePlay();
    }
  });
  addEventListener('beforeunload', (e) => {
    if (dirty()) e.preventDefault();
  });
}

async function main() {
  bind();
  try {
    S.styles = (await startSession()).styles ?? [];
    openDialog = initNewProject(async (created) => {
      await loadProjects();
      await openProject(created.id);
    });
    await loadProjects();
  } catch (e) {
    fail(e);
    $('#newBtn').disabled = true;
    $('#projectSelect').disabled = true;
    setEmpty($('#stageEmpty'), h('p', {}, e.message));
    return;
  }
  const wanted = decodeURIComponent(location.hash.slice(1));
  const first = S.projects.find((p) => p.id === wanted) ?? S.projects[0];
  if (first) await openProject(first.id);
  else renderAll(true);
  pollJobs();
}

main();
