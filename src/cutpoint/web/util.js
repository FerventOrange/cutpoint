// Small DOM and timecode helpers shared by the UI modules.

/** h('button', {class: 'btn', onclick: fn, 'aria-pressed': true}, 'Text', child) */
export function h(tag, props = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) continue;
    if (k.startsWith('on')) el.addEventListener(k.slice(2), v);
    else if (k === 'text') el.textContent = v;
    else if (k in el) el[k] = v;
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

/** 1536 -> "1.5 KB"; 52428800 -> "50 MB" (1024 steps, one decimal under 10). */
export function fmtBytes(n) {
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  let v = Math.max(0, Number(n) || 0);
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${i && v < 10 ? v.toFixed(1).replace(/\.0$/, '') : Math.round(v)} ${units[i]}`;
}

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

/** The row of ``n`` that takes the place of a row at index ``at`` once it left its list: the one now
 * at that index, the last one past the end, or -1 with no row left. */
export const rowInPlace = (at, n) => (n ? Math.min(at, n - 1) : -1);

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

/** The kinds a moment is shown as (the Moments grid's dot and filter, the sequence row's bar). */
export const KINDS = Object.freeze({ action: 'Action', scenic: 'Scenic', funny: 'Funny', dialogue: 'Dialogue' });

// The word sets of cutpoint.assistant.pacing (moment_kind), so the page and AI Chat name a moment's
// kind alike; tests/test_web_kind.py fails when they drift apart.
const GENERIC = /^(?:[\w-]+-moment|vlm)?$/;
const SCENIC_WORDS = ['view', 'views', 'scenic', 'scenery', 'beauty', 'beautiful', 'vista', 'vistas', 'landscape', 'panorama', 'panoramic'];
const FUNNY_WORDS = ['laugh', 'laughs', 'laughing', 'laughter', 'funny', 'joke', 'jokes', 'hilarious', 'giggle', 'giggles', 'comedy'];
const ACTION_WORDS = ['ski', 'skis', 'skiing', 'skier', 'skiers', 'fast', 'speed', 'speeding', 'crash', 'crashes', 'wipeout', 'cruise', 'cruising', 'crew', 'jump', 'jumps', 'airtime', 'air', 'carve', 'carving', 'run', 'runs', 'descent', 'powder', 'ride', 'riding', 'groomer', 'trees'];
// The strongest signal cue of a generic label decides next (pacing.moment_kind). The cues of the
// events VLM windows (funny, action-cam tags) are not in the payload, so a moment only they name
// falls through to its transcript here; that gap is accepted.
const CUE_STRONG = 5;
const SIGNAL_CUES = ['airtime', 'crash', 'wipeout', 'speed', 'carve', 'stopped', 'beauty', 'scenic', 'people', 'group', 'faces', 'social', 'voice_laughter', 'laughter', 'loudness'];
const CUE_KIND = { airtime: 'action', crash: 'action', wipeout: 'action', speed: 'action', carve: 'action', beauty: 'scenic', scenic: 'scenic', voice_laughter: 'funny', laughter: 'funny' };
const roundEven = (x) => { // Python's round(): halves go to the even number
  const f = Math.floor(x);
  const d = x - f;
  return d < 0.5 ? f : d > 0.5 ? f + 1 : f + (f % 2 !== 0 ? 1 : 0);
};
const TALK_WORDS = ['talk', 'talking', 'speech', 'dialogue', 'dialog', 'conversation', 'banter', 'chat', 'commentary', 'voice', 'speaking'];

/**
 * The kind a moment is shown as, for display only; highlights.json holds no such field. It follows
 * pacing.moment_kind (what AI Chat calls the moment) in the same order: the words of its labels
 * (not a generic "x-moment") and ``kind`` decide first (``view`` or a scenic word: scenic; a funny
 * word: funny; ``crew``, ``cruise`` or an action word: action; a talk word: dialogue), then the
 * strongest signal cue of 5 or more on a 0-10 scale (airtime, crash, wipeout, speed, carve: action;
 * beauty, scenic: scenic; voice_laughter, laughter: funny), then a transcript of four words or more
 * (dialogue), else action. The cues pacing also reads from the events VLM windows are not in the
 * payload. Null for no moment.
 */
export function kindOf(c) {
  if (!c) return null;
  const kind = typeof c.kind === 'string' ? c.kind : '';
  const named = [c.display_label, c.label, c.vision_label].filter((x) => x && !GENERIC.test(String(x)));
  const tokens = new Set(`${named.join(' ')} ${kind}`.toLowerCase().split(/[^a-z]+/));
  const has = (words) => words.some((w) => tokens.has(w));
  if (kind === 'view' || has(SCENIC_WORDS)) return 'scenic';
  if (has(FUNNY_WORDS)) return 'funny';
  if (kind === 'crew' || kind === 'cruise' || has(ACTION_WORDS)) return 'action';
  if (has(TALK_WORDS)) return 'dialogue';
  const sig = c.signals && typeof c.signals === 'object' ? c.signals : {};
  const cues = SIGNAL_CUES.filter((k) => typeof sig[k] === 'number' && Number.isFinite(sig[k]))
    .map((k) => [k, Math.min(10, roundEven(sig[k] * 10))])
    .filter(([, v]) => v > 0)
    .sort((a, b) => b[1] - a[1]); // stable: equal strengths keep SIGNAL_CUES order
  const strong = cues.find(([k, v]) => v >= CUE_STRONG && k in CUE_KIND);
  if (strong) return CUE_KIND[strong[0]];
  return String(c.transcript ?? '').trim().split(/\s+/).filter(Boolean).length >= 4 ? 'dialogue' : 'action';
}

/** A moment's score as the grid shows it: one decimal for the visual pool's 0-10 scale (``visual``), else whole numbers (0-100). */
export const scoreText = (score, visual = false) => (visual ? Number(score).toFixed(1) : String(Math.round(score)));

/** The Settings > Theme choices; Graphite is the default. */
export const THEMES = Object.freeze(['graphite', 'light', 'blue']);

/**
 * The theme a stored ``cutpoint.theme`` value means. Graphite is stored as no key. "dark", the
 * theme Graphite replaced, and anything unknown are Graphite. index.html's inline script does the same.
 */
export const themeFromStored = (v) => (v === 'light' || v === 'blue' ? v : 'graphite');

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
    ? listed.map((m) => ({ value: m.id, text: m.name && m.name !== m.id && !m.name.startsWith(m.id + ' ') ? `${m.name} (${m.id})` : m.name || m.id }))
    : fallback.map((id) => ({ value: id, text: id }));
  const options = [first, ...models];
  const known = current === '' && !typing ? true : models.some((o) => o.value === current);
  let value = current;
  if (!known && typing) value = AI_OTHER;
  else if (!known) options.splice(1, 0, { value: current, text: came ? `${current} (not in the provider's list)` : current });
  options.push({ value: AI_OTHER, text: 'Other model…' });
  return { options, value };
}

// Inline Markdown of a reply: `code`, **bold**, *italic* and _italic_ (not inside a word, so
// snake_case stays). A marker without its pair is left as it is.
const INLINE = /`([^`]+)`|\*\*(?=\S)(.+?)(?<=\S)\*\*|\*(?=[^\s*])(.+?)(?<=[^\s*])\*|(?<!\w)_(?=\S)(.+?)(?<=\S)_(?!\w)/g;

function inlineNodes(text) {
  const out = [];
  let at = 0;
  for (const m of text.matchAll(INLINE)) {
    if (m.index > at) out.push(text.slice(at, m.index));
    if (m[1] != null) out.push(h('code', { text: m[1] }));
    else if (m[2] != null) out.push(h('strong', {}, ...inlineNodes(m[2])));
    else out.push(h('em', {}, ...inlineNodes(m[3] ?? m[4])));
    at = m.index + m[0].length;
  }
  if (at < text.length) out.push(text.slice(at));
  return out;
}

/**
 * A model's reply as DOM nodes: paragraphs, line breaks, `-`/`*`/`1.` lists, **bold**, *italic*
 * and `code`. Built with createElement and text nodes only (no markup parsing, no links), so
 * nothing in the text can run.
 */
export function markdown(text) {
  const root = h('div', { class: 'md' });
  let para = null;
  let list = null;
  for (const line of String(text ?? '').split(/\r?\n/)) {
    const item = /^\s*(?:([-*])|(\d+)\.)\s+(\S.*)$/.exec(line);
    if (!line.trim()) {
      para = list = null;
    } else if (item) {
      const tag = item[1] ? 'ul' : 'ol';
      if (!list || list.tagName.toLowerCase() !== tag) {
        root.append((list = h(tag, item[1] ? {} : { start: item[2] })));
      }
      list.append(h('li', {}, ...inlineNodes(item[3].trim())));
      para = null;
    } else {
      if (!para) root.append((para = h('p')));
      else para.append(h('br'));
      para.append(...inlineNodes(line.trim()));
      list = null;
    }
  }
  return root;
}
