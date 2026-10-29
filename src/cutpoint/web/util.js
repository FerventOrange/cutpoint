// Small DOM and timecode helpers shared by the UI modules.

/** h('button', {class: 'btn', onclick: fn, 'aria-pressed': true}, 'Text', child) */
export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'text') el.textContent = v;
    else if (k in el && k !== 'list') el[k] = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  el.append(...children.flat().filter((c) => c != null && c !== false));
  return el;
}

/** Decorative inline icon: svg('icon', [['path', {d: '...'}], ['circle', {cx, cy, r}]]). */
export function svg(cls, shapes, viewBox = '0 0 24 24') {
  const NS = 'http://www.w3.org/2000/svg';
  const el = document.createElementNS(NS, 'svg');
  for (const [k, v] of Object.entries({ class: cls, viewBox, 'aria-hidden': 'true', focusable: 'false', fill: 'currentColor' }))
    el.setAttribute(k, v);
  for (const [tag, attrs] of shapes) {
    const s = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) s.setAttribute(k, v);
    el.append(s);
  }
  return el;
}

export const $ = (sel, root = document) => root.querySelector(sel);

/** 83.5 -> "1:23.50"; 3723 -> "1:02:03.00". Tabular digits come from CSS. */
export function fmt(t) {
  if (!Number.isFinite(t)) return '–';
  const sign = t < 0 ? '-' : '';
  t = Math.abs(t);
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = (t % 60).toFixed(2).padStart(5, '0');
  return `${sign}${h ? `${h}:${String(m).padStart(2, '0')}` : m}:${s}`;
}

/** Duration without centiseconds: 125 -> "2:05". */
export function fmtLen(t) {
  const s = Math.round(t);
  const m = Math.floor(s / 60);
  return `${m}:${String(s % 60).padStart(2, '0')}`;
}

/** A length in words (#459): 45 -> "45 s", 540 -> "9 min", 8040 -> "2 h 14 min", 7200 -> "2 h". */
export function spanText(t) {
  if (Math.round(t) < 60) return `${Math.max(1, Math.round(t))} s`;
  const m = Math.round(t / 60);
  if (m < 60) return `${m} min`;
  return m % 60 ? `${Math.floor(m / 60)} h ${m % 60} min` : `${m / 60} h`;
}

/** "83.5", "1:23.5", "1:02:03" -> seconds, or NaN. */
export function parseTime(text) {
  const parts = String(text).trim().split(':');
  if (parts.length > 3 || parts.some((p) => !/^\d+(\.\d*)?$/.test(p))) return NaN;
  return parts.reduce((acc, p) => acc * 60 + Number(p), 0);
}

export const baseName = (p) => String(p).split(/[\\/]/).pop();
export const round2 = (t) => Math.round(t * 100) / 100;

const NATURAL = new Intl.Collator('en', { numeric: true, sensitivity: 'base' });
/** Natural order of names: "clip2" before "clip10", case ignored; names equal but for case keep a fixed order. */
export const naturalCompare = (a, b) => NATURAL.compare(a, b) || (a < b ? -1 : a > b ? 1 : 0);

/**
 * Moments in video order (#469): by source file name in natural order, then start time in each
 * source. Two folders' files of the same name sort by full path, so each source stays together.
 */
export const positionCompare = (a, b) =>
  naturalCompare(baseName(a.source), baseName(b.source)) || naturalCompare(String(a.source), String(b.source)) || a.start - b.start;

/**
 * The moments list's quick filters (Sort and filter) with nothing filtered: ``seq`` 'all', 'in'
 * (in the sequence) or 'out'; ``ready``: only moments whose preview plays; ``source``: one
 * source path, '' for every file.
 */
export const NO_FILTERS = Object.freeze({ seq: 'all', ready: false, source: '' });

/** Any quick filter of ``f`` filters (the search text is not one). */
export const filtersOn = (f) => f.seq !== 'all' || !!f.ready || !!f.source;

/**
 * A moment shows in the list: it holds the search text ``q`` (in its label, CLI id such as h001,
 * transcript or file name, case ignored) and passes every quick filter of ``f``. ``m``: the candidate with
 * ``picked`` (in the sequence) and ``ready`` (its preview plays).
 */
export function momentMatches(m, q, f) {
  const text = String(q ?? '').trim().toLowerCase();
  if (text && !`${m.label ?? ''} ${m.cli_id ?? ''} ${m.transcript ?? ''} ${baseName(m.source)}`.toLowerCase().includes(text)) return false;
  if ((f.seq === 'in' && !m.picked) || (f.seq === 'out' && m.picked)) return false;
  if (f.ready && !m.ready) return false;
  return !f.source || m.source === f.source;
}

// Why a clip got no grade, in plain words: ``color.skip`` in clips/manifest.json (cutpoint.color's
// skip reasons). A count follows a reason shared by several ("already looked good: 2").
const COLOR_SKIPS = {
  good: 'already looked good',
  log: 'log footage',
  360: '360 footage',
  no_frames: 'frames could not be read',
  veto_clipped: 'too much highlight clipping',
  veto_cast: 'would add a color cast',
  veto_m3: 'would oversaturate',
  veto_m3_drop: 'would wash out the color',
  not_underwater: 'did not look underwater',
  veto_faded: 'would wash out the color',
};

/**
 * The Export state's Color sentence for an export's clips (manifest entries), starting with a space:
 * how many clips the grade changed and why the others were left as is, so a grade every clip
 * skipped is not silent. '' when the export had no Color (a LUT grade has no ``method``) or only at
 * Strength 0%.
 */
export function colorNote(clips) {
  const asked = (clips ?? []).filter((c) => c?.color?.method);
  if (!asked.some((c) => (c.color.strength ?? 1) > 0)) return ''; // Strength 0%: no grade asked for
  const why = new Map();
  for (const c of asked) {
    const skip = c.color.skip ? (COLOR_SKIPS[c.color.skip] ?? c.color.skip) : (c.color.strength ?? 1) > 0 ? '' : 'strength 0%';
    if (skip) why.set(skip, (why.get(skip) ?? 0) + 1);
  }
  const left = [...why.values()].reduce((a, b) => a + b, 0);
  const done = asked.length - left;
  if (!left) return asked.length === 1 ? ' Color: 1 clip graded.' : ` Color: all ${asked.length} clips graded.`;
  const reasons = why.size === 1 ? [...why.keys()][0] : [...why].map(([text, n]) => `${text}: ${n}`).join(', ');
  if (!done) return ` Color: no clips graded (${reasons}).`;
  return ` Color: ${done} of ${asked.length} clips graded; ${left} left as is (${reasons}).`;
}

/** The value of the Model select's last option, "Other model…": no model id has it. */
export const AI_OTHER = '\u0000other';

/**
 * The Model select's options ({ value, text }) and the value to select (#521). ``current`` is the
 * model to show ('' for the provider's default), ``listed`` the provider's models ({ id, name };
 * empty or missing when the list did not load, when the preset's ``fallback`` ids are offered
 * instead) and ``typing`` true while an id is entered under "Other model…", which then stays
 * chosen unless the options have that id. Any other model they lack stays chosen as an extra
 * option, marked when a list came, so opening Settings never changes the model.
 */
export function aiModelChoices(current, listed, { typing = false, required = false, placeholder = '', fallback = [] } = {}) {
  const first = { value: '', text: required ? 'Choose a model' : placeholder ? `Default (${placeholder})` : 'Default' };
  const came = listed?.length > 0;
  const models = came
    ? listed.map((m) => ({ value: m.id, text: m.name && m.name !== m.id ? `${m.name} (${m.id})` : m.id }))
    : fallback.map((id) => ({ value: id, text: id }));
  const options = [first, ...models];
  const known = current === '' && !typing ? true : models.some((o) => o.value === current);
  let value = current;
  if (!known && typing) value = AI_OTHER;
  else if (!known) options.splice(1, 0, { value: current, text: came ? `${current} (not in the provider's list)` : current });
  options.push({ value: AI_OTHER, text: 'Other model…' });
  return { options, value };
}
