// The Graphite refresh: the theme default and the old Dark theme's migration, the Moments grid
// (kind, score, kind filter), the header Export button and the sequence row's kind and tail line.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, importPure, makeProject } from './harness.mjs';

const u = await importPure('util.js');

async function open(t, { project = makeProject(), storage = {}, routes = {}, setup } = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ project, storage, routes, setup });
  t.after(() => page.close());
  return page;
}
const theme = (page) => page.document.documentElement.dataset.theme;

// ---- theme ----

test('themeFromStored: Graphite is the default; Light and Blue are kept; a stored "dark" is Graphite', () => {
  assert.deepEqual([null, undefined, '', 'graphite', 'dark', 'nope'].map(u.themeFromStored), Array(6).fill('graphite'));
  assert.equal(u.themeFromStored('light'), 'light');
  assert.equal(u.themeFromStored('blue'), 'blue');
  assert.deepEqual([...u.THEMES], ['graphite', 'light', 'blue']);
});

test('no saved theme boots as Graphite: no data-theme, the select says so and offers Graphite, Light, Blue', async (t) => {
  const page = await open(t);
  assert.equal(theme(page), undefined);
  const sel = page.document.querySelector('#setTheme');
  assert.equal(sel.value, 'graphite');
  assert.deepEqual([...sel.options].map((o) => o.value), ['graphite', 'light', 'blue']);
});

test('a saved "dark" becomes Graphite and its key is dropped', async (t) => {
  const page = await open(t, { storage: { 'cutpoint.theme': 'dark' } });
  assert.equal(theme(page), undefined);
  assert.equal(page.document.querySelector('#setTheme').value, 'graphite');
  assert.equal(page.window.localStorage.getItem('cutpoint.theme'), null);
});

for (const saved of ['light', 'blue']) {
  test(`a saved "${saved}" stays ${saved}`, async (t) => {
    const page = await open(t, { storage: { 'cutpoint.theme': saved } });
    assert.equal(theme(page), saved);
    assert.equal(page.document.querySelector('#setTheme').value, saved);
    assert.equal(page.window.localStorage.getItem('cutpoint.theme'), saved);
  });
}

test('picking a theme in Settings applies it at once; Graphite is stored as no key, Blue as "blue"', async (t) => {
  const page = await open(t);
  const sel = page.document.querySelector('#setTheme');
  const pick = async (v) => {
    sel.value = v;
    sel.dispatchEvent(new page.window.Event('change', { bubbles: true }));
    await page.settle();
  };
  await pick('blue');
  assert.equal(theme(page), 'blue');
  assert.equal(page.window.localStorage.getItem('cutpoint.theme'), 'blue');
  await pick('graphite');
  assert.equal(theme(page), undefined);
  assert.equal(page.window.localStorage.getItem('cutpoint.theme'), null);
});

// ---- kind ----

test('kindOf: scenic views, laughter, a few spoken words, else action; scoreText fits both scales', () => {
  assert.equal(u.kindOf({ pool: 'visual', kind: 'view' }), 'scenic');
  assert.equal(u.kindOf({ pool: 'visual', kind: 'cruise', transcript: '' }), 'action');
  assert.equal(u.kindOf({ signals: { laughter: 0.9 }, transcript: 'one two three four five' }), 'funny');
  assert.equal(u.kindOf({ label: 'laugh', transcript: '' }), 'funny');
  assert.equal(u.kindOf({ transcript: 'we are going down the hill' }), 'dialogue');
  assert.equal(u.kindOf({ transcript: 'wow' }), 'action');
  assert.equal(u.kindOf(undefined), null);
  assert.deepEqual([88.4, 9.06, 100].map((n) => u.scoreText(n, n < 10)), ['88', '9.1', '100']);
  assert.equal(u.scoreText(9.6, false), '10');
  assert.equal(u.scoreText(10, true), '10.0');
  // pacing.moment_kind's order: scenic word, funny word, action word, talk word, then laughter, then the transcript
  assert.equal(u.kindOf({ label: 'Beautiful vista', transcript: 'a b c d e' }), 'scenic');
  assert.equal(u.kindOf({ label: 'Fast joke' }), 'funny');
  assert.equal(u.kindOf({ label: 'Banter at speed' }), 'action');
  assert.equal(u.kindOf({ label: 'Banter', signals: { laughter: 0.9 } }), 'dialogue');
  assert.equal(u.kindOf({ label: 'ski-moment', vision_label: 'two friends talking' }), 'dialogue');
  assert.equal(u.kindOf({ label: 'ski-moment', transcript: '' }), 'action'); // a generic label names nothing
  // then the strongest signal cue of 5+ of 10 (0-1 signals scaled, rounded half to even)
  assert.equal(u.kindOf({ label: 'ski-moment', signals: { beauty: 0.8 }, transcript: 'a b c d e' }), 'scenic');
  assert.equal(u.kindOf({ label: 'ski-moment', signals: { laughter: 0.6, crash: 0.9 } }), 'action');
  assert.equal(u.kindOf({ label: 'ski-moment', signals: { laughter: 0.45 }, transcript: 'a b c d e' }), 'dialogue'); // 4.5 rounds to 4
  assert.equal(u.kindOf({ label: 'ski-moment', signals: { laughter: 0.55 } }), 'funny'); // 5.5 rounds to 6
  assert.equal(u.kindOf({ label: 'ski-moment', signals: { laughter: 0.5 } }), 'funny'); // 5
  assert.equal(u.kindOf({ label: 'ski-moment', signals: { loudness: 0.9, laughter: 0.6 } }), 'funny'); // loudness has no kind
});

function kinds() {
  const p = makeProject();
  const [a, b, c] = p.highlights.candidates;
  p.highlights.candidates = [
    { ...a, label: 'Fast run', score: 88.4, transcript: '' },
    { ...b, id: 'v001', cli_id: 'v001', label: 'view', pool: 'visual', kind: 'view', score: 9.06, transcript: '' },
    { ...c, signals: { laughter: 0.8 } },
  ];
  return p;
}

test('a Moments tile has its kind, the kind and score line, the length and a still box; the buttons keep their order', async (t) => {
  const page = await open(t, { project: kinds() });
  const d = page.document;
  const tile = (id) => d.querySelector(`#candList li.cand[data-id="${id}"]`);
  assert.equal(tile('h001').dataset.kind, 'action');
  assert.equal(tile('v001').dataset.kind, 'scenic');
  assert.equal(tile('h003').dataset.kind, 'funny');
  const meta = (id) => tile(id).querySelector('.cand-meta').textContent;
  assert.equal(meta('h001'), 'Action · , score 88');
  assert.equal(meta('v001'), 'Scenic · , score 9.1');
  assert.equal(tile('h001').querySelector('.cand-len').textContent, '8.0 s');
  assert.equal(tile('h001').querySelector('.cand-thumb .cand-dot')?.getAttribute('aria-hidden'), 'true');
  // the sequence row's URL, so the two share one cached still: a box that shows the kind tint until it loads
  assert.ok(tile('h001').querySelector('.cand-thumb'));
  assert.deepEqual([...tile('h001').children].map((e) => e.dataset.act), ['focus', 'toggle']);
});

test('a tile asks for its still only once it scrolls into view, through one shared observer', async (t) => {
  const made = [];
  const seen = new Set();
  class Watch {
    constructor(cb) { this.cb = cb; made.push(this); }
    observe(el) { seen.add(el); }
    unobserve(el) { seen.delete(el); }
    disconnect() {}
  }
  const page = await open(t, { project: kinds(), setup: (w) => { w.IntersectionObserver = Watch; } });
  const d = page.document;
  const img = (id) => d.querySelector(`#candList li.cand[data-id="${id}"] .cand-thumb img`);
  assert.equal(made.length, 1);
  assert.equal(img('h001').getAttribute('src'), null);
  assert.equal(page.requests.filter((r) => /\/frame\?/.test(String(r.url ?? r))).length, 0);
  const box = img('h001').parentElement;
  assert.ok(seen.has(box));
  made[0].cb([{ target: box, isIntersecting: false }]);
  await page.settle();
  assert.equal(img('h001').getAttribute('src'), null);
  made[0].cb([{ target: box, isIntersecting: true }]);
  await page.settle();
  assert.match(img('h001').getAttribute('src'), /\/frame\?/);
  assert.equal(seen.has(box), false); // not watched any more
  assert.equal(img('h003').getAttribute('src'), null); // its tile has not scrolled in
});

test('the kind filter shows one kind, All brings the rest back, and the pressed segment says which', async (t) => {
  const page = await open(t, { project: kinds() });
  const d = page.document;
  const ids = () => [...d.querySelectorAll('#candList li.cand')].map((li) => li.dataset.id);
  const seg = (k) => d.querySelector(`#kindSeg [data-kind="${k}"]`);
  assert.equal(d.querySelector('#kindSeg').hidden, false);
  assert.equal(seg('').getAttribute('aria-pressed'), 'true');
  seg('scenic').click();
  await page.settle();
  assert.deepEqual(ids(), ['v001']);
  assert.equal(seg('scenic').getAttribute('aria-pressed'), 'true');
  assert.equal(seg('').getAttribute('aria-pressed'), 'false');
  seg('dialogue').click();
  await page.settle();
  assert.deepEqual(ids(), []);
  assert.match(d.querySelector('#candList').textContent, /No moments match these filters/);
  seg('').click();
  await page.settle();
  assert.deepEqual(ids(), ['h001', 'v001', 'h003']);
});

// ---- header Export ----

test('the header Export button opens the Export tab by the tab path, and the tab stays', async (t) => {
  const page = await open(t);
  const d = page.document;
  const btn = d.querySelector('.top #exportBtn');
  assert.ok(btn, 'Export sits in the header');
  assert.equal(btn.nextElementSibling.id, 'saveBtn', 'next to Save sequence');
  assert.equal(d.querySelector('#panelExport').hidden, true);
  btn.click();
  await page.settle();
  assert.equal(d.querySelector('#panelExport').hidden, false);
  assert.equal(d.querySelector('#tabExport').getAttribute('aria-selected'), 'true');
  assert.equal(d.querySelector('#panelSequence').hidden, true);
  d.querySelector('#tabSequence').click();
  assert.equal(d.querySelector('#panelExport').hidden, true);
  d.querySelector('#tabExport').click();
  assert.equal(d.querySelector('#panelExport').hidden, false);
});

// ---- sequence rows ----

function withSegment() {
  const p = kinds();
  const c = p.highlights.candidates[0];
  p.review = { segments: [{ id: c.id, source: c.source, start: c.start, end: c.end, label: c.label }] };
  return p;
}

test('a sequence row has its kind and name, and "cut N s + tail" only when the export runs past the cut', async (t) => {
  const page = await open(t, {
    project: withSegment(),
    routes: { 'POST /api/projects/p1/edges': () => ({ edges: { h001: { start: 10, end: 26.22, reason: 'event tail' } } }) },
  });
  const d = page.document;
  const row = () => d.querySelector('#seqList li[data-id="h001"]');
  assert.equal(row().dataset.kind, 'action');
  assert.equal(row().querySelector('.seq-kind').textContent, 'Action');
  assert.equal(row().querySelector('.seq-tail'), null, 'no tail until the server plans one');
  assert.equal(row().querySelector('.seq-dur').firstChild.textContent, '8.0 s');
  t.mock.timers.tick(400);
  await page.settle();
  assert.equal(row().querySelector('.seq-dur').firstChild.textContent, '16.2 s');
  assert.equal(row().querySelector('.seq-tail').textContent, 'cut 8.0 s + tail');
});
