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
