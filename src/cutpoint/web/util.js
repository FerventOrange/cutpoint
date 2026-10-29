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
