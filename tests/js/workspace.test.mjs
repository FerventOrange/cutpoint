// Workspace behaviour that tests/test_ui.py used to pin by the source text of app.js and
// index.html (#1085 A594): Stats for Nerds, the tile's source strip, the camera path tooltip, the
// 360 empty state, focus after a row leaves the list, the side tabs, the column splitters, the
// Sort and filter popover, the undo stack's cap, the rail filmstrip and the early theme script.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { JSDOM } from 'jsdom';
import { boot, makeProject, answerConfirms, WEB_DIR, ORIGIN } from './harness.mjs';

async function open(t, { project = makeProject(), storage = {}, setup, routes = {}, allowUnhandled = false } = {}) {
  t.mock.timers.reset(); // a test may boot twice
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ project, storage, setup, routes, allowUnhandled });
  t.after(() => page.close());
  const d = page.document;
  const w = page.window;
  return {
    page, d, w,
    $: (s) => d.querySelector(s),
    tile: (id) => d.querySelector(`#candList li.cand[data-id="${id}"]`),
    async click(sel) { (typeof sel === 'string' ? d.querySelector(sel) : sel).click(); await page.settle(); },
    async key(target, key, init = {}) {
      const e = new w.KeyboardEvent('keydown', { key, bubbles: true, cancelable: true, ...init });
      target.dispatchEvent(e);
      await page.settle();
      return e;
    },
    async change(sel, value) {
      const el = d.querySelector(sel);
      if (el.type === 'checkbox' || el.type === 'radio') el.checked = value; else el.value = value;
      el.dispatchEvent(new w.Event('change', { bubbles: true }));
      await page.settle();
    },
    async tick(ms) { t.mock.timers.tick(ms); await page.settle(); },
  };
}

/** Layout for jsdom (which has none): offsetWidth/clientWidth by element id, and the window width. */
const layout = (widths, innerWidth = 1600) => (w) => {
  for (const prop of ['offsetWidth', 'clientWidth']) {
    Object.defineProperty(w.HTMLElement.prototype, prop, { configurable: true, get() { return widths[this.id] ?? 0; } });
  }
  Object.defineProperty(w, 'innerWidth', { configurable: true, value: innerWidth });
};

// ---------- Stats for Nerds (#353) ----------

test('Stats for Nerds: off by default, the Settings box shows the overlay on the preview at once, and it is remembered', async (t) => {
  const KEY = 'cutpoint.nerdStats';
  const m = await open(t);
  await m.click(m.tile('h001').querySelector('.cand-main'));
  const sig = m.$('#signals');
  assert.ok(sig.closest('.screen'), 'the overlay sits on the preview, not under the transcript');
  assert.equal(m.d.querySelectorAll('#signals').length, 1);
  assert.equal(m.$('#setNerdStats').checked, false);
  assert.equal(sig.hidden, true);
  assert.equal(sig.textContent, '');

  await m.change('#setNerdStats', true);
  assert.equal(sig.hidden, false);
  assert.match(sig.textContent, /^Score 0\.90, rank 1 of 3/);
  assert.equal(m.w.localStorage.getItem(KEY), '1');

  await m.change('#setNerdStats', false);
  assert.equal(sig.hidden, true);
  assert.equal(sig.textContent, '');
  assert.equal(m.w.localStorage.getItem(KEY), null);

  m.page.close();
  const again = await open(t, { storage: { [KEY]: '1' } });
  await again.click(again.tile('h001').querySelector('.cand-main'));
  assert.equal(again.$('#signals').hidden, false, 'a reload keeps it on');
});

// ---------- the tile's source strip ----------

test('a tile\'s strip marks the moment\'s place in the source, has no control and is hidden from screen readers', async (t) => {
  const m = await open(t);
  const strip = m.tile('h002').querySelector('.cand-strip');
  assert.equal(strip.getAttribute('aria-hidden'), 'true');
  assert.equal(strip.title, 'Score 0.700');
  assert.equal(strip.querySelector('input, button, [tabindex]'), null);
  // h002 is 100-108 s of a 600 s source
  const clip = strip.querySelector('.cand-strip-clip').getAttribute('style');
  assert.match(clip, /left: ?min\(16\.67%/);
  assert.match(clip, /width: ?1\.33%/);
  assert.equal(m.d.querySelector('#candList .score'), null, 'no score bar beside it');
  m.page.assertNoUnhandled(); // the strip reads the listed duration: no request per card
});

// ---------- camera path flags (#517) ----------

test('a flagged card\'s reasons are a tooltip that describes it; Escape hides the tooltips until focus moves to another card', async (t) => {
  const p = makeProject();
  p.highlights.candidates[0].pano = { flags: ['shaky'], reasons: [{ text: 'the camera shakes' }] };
  const m = await open(t, { project: p });
  const main = m.tile('h001').querySelector('.cand-main');
  const tip = m.d.getElementById(main.getAttribute('aria-describedby'));
  assert.equal(tip.getAttribute('role'), 'tooltip');
  assert.equal(tip.textContent, 'Camera path looks poor: the camera shakes.');

  main.focus();
  await m.key(main, 'Escape');
  assert.ok(m.d.body.classList.contains('tips-off'));
  m.tile('h002').querySelector('.cand-main').focus();
  await m.page.settle();
  assert.ok(!m.d.body.classList.contains('tips-off'), 'another card brings the tooltips back');
});

test('the 360 note is one permanent status region: its text changes, the node stays', async (t) => {
  const p = makeProject({ kind: '360' });
  for (const c of p.highlights.candidates) c.pano = { flags: [], preview: { state: 'none' } };
  const m = await open(t, { project: p, allowUnhandled: true });
  const status = m.$('#panoNote .pano-status');
  assert.equal(status.getAttribute('role'), 'status');
  await m.click(m.tile('h001').querySelector('.cand-main'));
  assert.equal(m.$('#panoNote').hidden, false);
  await m.click(m.tile('h002').querySelector('.cand-main'));
  assert.equal(m.$('#panoNote .pano-status'), status);
});

test('a 360 director preview is asked for once, its render is polled with Cancel, and a cancelled one is offered again', async (t) => {
  const p = makeProject({ kind: '360' });
  for (const c of p.highlights.candidates) c.pano = { path: `/p/${c.cli_id}.json`, reasons: [] };
  let preview = null; // what the server lists for h001 now
  const project = () => {
    const now = structuredClone(p);
    now.highlights.candidates[0].pano.preview = preview;
    return now;
  };
  const posts = [];
  const m = await open(t, { project: project(), routes: {
    'GET /api/projects/p1': () => project(),
    'POST /api/projects/p1/director/h001': (req) => { posts.push(req.path); preview = { state: 'rendering', progress: 0.5, seconds: 8 }; return {}; },
    'POST /api/projects/p1/director/h001/cancel': (req) => { posts.push(req.path); preview = { state: 'cancelled' }; return {}; },
  } });
  const status = () => m.$('#panoNote .pano-status').textContent;
  const button = () => m.$('#panoNote .pano-act button');
  const gets = () => m.page.requests.filter((r) => r.method === 'GET' && r.path === '/api/projects/p1').length;
  const tick = async (ms) => { t.mock.timers.tick(ms); await m.page.settle(); };
  await tick(16); // the first status shows on the next frame
  assert.deepEqual(posts, ['/api/projects/p1/director/h001'], 'the open moment\'s preview is asked for');
  assert.equal(status(), 'Rendering the director view; the raw 360 preview plays meanwhile.');
  assert.equal(m.$('#panoNote .pano-time').textContent, ' 50% (about 8 s of video)');
  assert.equal(button().textContent, 'Cancel');

  const before = gets();
  preview = { state: 'rendering', progress: 0.75, seconds: 8 };
  await tick(1000);
  assert.equal(gets(), before + 1, 'a render is polled every second');
  assert.equal(m.$('#panoNote .pano-time').textContent, ' 75% (about 8 s of video)');
  assert.equal(posts.length, 1, 'polling does not ask again');

  await m.click(button());
  assert.deepEqual(posts.slice(1), ['/api/projects/p1/director/h001/cancel']);
  assert.equal(status(), 'Raw 360 preview: the director view was cancelled.');
  const stopped = gets();
  await tick(5000);
  assert.equal(gets(), stopped, 'a cancelled render is not polled');

  await m.click(button());
  assert.equal(button().textContent, 'Cancel', 'Render director view asks again');
  assert.deepEqual(posts.slice(2), ['/api/projects/p1/director/h001']);
});

// ---------- 360 empty state ----------

for (const [n, text] of [[2, '360 footage: 2 recordings found'], [1, '360 footage: 1 recording found'], [0, 'No 360 recordings found']]) {
  test(`a 360 project with no moments counts its recordings apart from "360" (${n})`, async (t) => {
    const src = (i) => ({ id: `s${i}`, path: `/videos/r${i}.insv`, name: `r${i}.insv`, status: 'ready', url: `/api/projects/p1/sources/s${i}?t=x`, duration: 60 });
    const p = makeProject({ kind: '360', highlights: null, source: '/videos', sources: Array.from({ length: n }, (_, i) => src(i + 1)) });
    const m = await open(t, { project: p, allowUnhandled: true });
    const count = m.$('#momentsEmpty .fresh-count');
    assert.equal(count.textContent, text);
    assert.doesNotMatch(m.$('#momentsEmpty').textContent, /\d 360/, 'no count directly before "360" (read as 5,360)');
  });
}

// ---------- focus after a row leaves the list ----------

test('Add under "Not in sequence" moves focus to the card now in its place, not to the top of the list', async (t) => {
  const m = await open(t, { storage: { 'cutpoint.momentFilters': JSON.stringify({ seq: 'out', ready: false }) } });
  const ids = () => [...m.d.querySelectorAll('#candList li.cand')].map((li) => li.dataset.id);
  assert.deepEqual(ids(), ['h001', 'h002', 'h003']);
  const add = m.tile('h002').querySelector('[data-act=toggle]');
  add.focus();
  await m.click(add);
  assert.deepEqual(ids(), ['h001', 'h003'], 'the added card left the list');
  assert.equal(m.d.activeElement, m.tile('h003').querySelector('[data-act=toggle]'));
  // the last card: focus goes to the new last row
  const last = m.tile('h003').querySelector('[data-act=toggle]');
  await m.click(last);
  assert.equal(m.d.activeElement, m.tile('h001').querySelector('[data-act=toggle]'));
  // the last card leaves: the list is empty, focus goes to Sort and filter, and the empty state names the filters
  await m.click(m.tile('h001').querySelector('[data-act=toggle]'));
  assert.equal(m.d.querySelectorAll('#candList li.cand').length, 0);
  assert.equal(m.d.activeElement, m.$('#filterBtn'));
  assert.match(m.$('#candList').textContent, /No moments match these filters/);
});

// ---------- side tabs (#462, #346) ----------

test('the side column is a tablist: tab and panel name each other, one tab stop, arrow keys select and focus', async (t) => {
  const m = await open(t);
  const tabs = [...m.d.querySelectorAll('#side [role=tablist] [role=tab]')];
  assert.deepEqual(tabs.map((b) => b.textContent.trim()), ['Sequence', 'AI Chat', 'Models'], 'the visible text is the name');
  for (const tab of tabs) {
    const panel = m.d.getElementById(tab.getAttribute('aria-controls'));
    assert.equal(panel.getAttribute('role'), 'tabpanel');
    assert.equal(panel.getAttribute('aria-labelledby'), tab.id);
  }
  const state = () => tabs.map((b) => [b.getAttribute('aria-selected'), b.tabIndex, m.d.getElementById(b.getAttribute('aria-controls')).hidden]);
  assert.deepEqual(state(), [['true', 0, false], ['false', -1, true], ['false', -1, true]]);

  tabs[0].focus();
  await m.key(tabs[0], 'ArrowRight');
  assert.deepEqual(state()[1], ['true', 0, false]);
  assert.deepEqual(state()[0], ['false', -1, true]);
  assert.equal(m.d.activeElement, tabs[1]);
  await m.key(tabs[1], 'Home');
  assert.equal(m.d.activeElement, tabs[0]);
  assert.deepEqual(state()[0], ['true', 0, false]);

  // Export is no tab: the header button controls a region with its own heading
  const exportBtn = m.$('#exportBtn');
  const panel = m.d.getElementById(exportBtn.getAttribute('aria-controls'));
  assert.equal(panel.id, 'panelExport');
  assert.equal(panel.getAttribute('role'), 'region');
  assert.equal(panel.getAttribute('aria-labelledby'), 'exportTitle');
  assert.equal(exportBtn.getAttribute('aria-expanded'), 'false');
  await m.click(exportBtn);
  assert.equal(exportBtn.getAttribute('aria-expanded'), 'true');
  assert.equal(panel.hidden, false);
  assert.ok(tabs.every((b) => b.getAttribute('aria-selected') === 'false'));
});

// ---------- column splitters (#516) ----------

const WIDE = { workspace: 1600, side: 290, moments: 280 };

for (const [bar, prop, key, base, wider] of [
  ['#sideSplit', '--side-w', 'cutpoint.sideWidth', 290, 'ArrowLeft'], // the column is right of its bar
  ['#momSplit', '--mom-w', 'cutpoint.momentsWidth', 280, 'ArrowRight'],
]) {
  test(`${bar}: a focusable separator; keys resize the column, Home and double-click reset, and the width is remembered`, async (t) => {
    const m = await open(t, { setup: layout(WIDE) });
    const el = m.$(bar);
    const ws = m.$('#workspace');
    assert.equal(el.getAttribute('role'), 'separator');
    assert.equal(el.tabIndex, 0);
    assert.equal(el.getAttribute('aria-orientation'), 'vertical');
    assert.ok(m.d.getElementById(el.getAttribute('aria-controls')), 'it names its column');

    await m.key(el, wider);
    assert.equal(ws.style.getPropertyValue(prop), `${base + 16}px`);
    assert.equal(el.getAttribute('aria-valuenow'), String(base + 16));
    assert.equal(m.w.localStorage.getItem(key), String(base + 16));
    await m.key(el, wider, { shiftKey: true });
    assert.equal(ws.style.getPropertyValue(prop), `${base + 64}px`, 'Shift moves four steps');
    await m.key(el, 'Home');
    assert.equal(el.getAttribute('aria-valuenow'), el.getAttribute('aria-valuemin'));
    el.dispatchEvent(new m.w.MouseEvent('dblclick', { bubbles: true }));
    await m.page.settle();
    assert.equal(ws.style.getPropertyValue(prop), '', 'back to the CSS default');
    assert.equal(m.w.localStorage.getItem(key), null);

    m.page.close();
    const again = await open(t, { setup: layout(WIDE), storage: { [key]: String(base + 40) } });
    assert.equal(again.$('#workspace').style.getPropertyValue(prop), `${base + 40}px`, 'a reload keeps the width');
  });
}

test('a column never widens into the preview\'s 360 px', async (t) => {
  // 1000 px workspace, the moments column 280 px: the sequence column may take 1000 - 280 - 360
  const m = await open(t, { setup: layout({ workspace: 1000, side: 290, moments: 280 }) });
  const bar = m.$('#sideSplit');
  await m.key(bar, 'End');
  assert.equal(m.$('#workspace').style.getPropertyValue('--side-w'), '360px');
  assert.equal(bar.getAttribute('aria-valuemax'), '360');
});

// ---------- Sort and filter ----------

test('the funnel button discloses Sort and filter: Escape closes and refocuses it, a press outside closes it', async (t) => {
  const m = await open(t);
  const btn = m.$('#filterBtn');
  const pop = m.$('#filterPop');
  assert.equal(btn.getAttribute('aria-controls'), 'filterPop');
  assert.equal(btn.getAttribute('aria-expanded'), 'false');
  assert.equal(pop.hidden, true);
  assert.deepEqual([...pop.querySelectorAll('fieldset > legend')].map((l) => l.textContent.trim()), ['Sort by', 'Sequence', 'Show only']);

  await m.click(btn);
  assert.equal(btn.getAttribute('aria-expanded'), 'true');
  assert.equal(pop.hidden, false);
  const radio = pop.querySelector('input[name=filterSeq][value=in]');
  radio.focus();
  const esc = await m.key(radio, 'Escape');
  assert.equal(pop.hidden, true);
  assert.equal(btn.getAttribute('aria-expanded'), 'false');
  assert.equal(m.d.activeElement, btn);
  assert.ok(esc.defaultPrevented, 'the page\'s own Escape does not also run');

  await m.click(btn);
  m.d.body.dispatchEvent(new m.w.Event('pointerdown', { bubbles: true }));
  await m.page.settle();
  assert.equal(pop.hidden, true);
});

test('an active filter or sort marks the funnel and its name; the stored filters never hold the source file', async (t) => {
  const p = makeProject();
  p.highlights.candidates[1].source = '/videos/b.mp4';
  p.sources.push({ id: 's2', path: '/videos/b.mp4', name: 'b.mp4', status: 'ready', url: '/api/projects/p1/sources/s2?t=x', duration: 600 });
  const m = await open(t, { project: p });
  const btn = m.$('#filterBtn');
  assert.ok(!btn.classList.contains('on'));
  assert.equal(btn.getAttribute('aria-label'), 'Sort and filter moments');

  await m.change('#filterSource', '/videos/b.mp4');
  assert.ok(btn.classList.contains('on'));
  assert.equal(btn.getAttribute('aria-label'), 'Sort and filter moments, filters on');
  assert.equal(m.w.localStorage.getItem('cutpoint.momentFilters'), null, 'a path of one project is not stored');
  assert.equal(m.$('#orderLive').textContent, '1 of 3 moments shown');

  await m.change('#filterSource', '');
  await m.change('input[name=filterSeq][value=out]', true);
  assert.deepEqual(JSON.parse(m.w.localStorage.getItem('cutpoint.momentFilters')), { seq: 'out', ready: false });
});

// ---------- the undo stack ----------

test('the undo stack keeps the last 50 edits; Shift+Ctrl+Z redoes; Undo and Redo are named icon buttons with their shortcuts', async (t) => {
  const m = await open(t);
  answerConfirms(m.page, [true]);
  for (const [id, name, keys] of [['#undoBtn', 'Undo', 'Control+Z'], ['#redoBtn', 'Redo', 'Control+Y Control+Shift+Z']]) {
    const b = m.$(id);
    assert.equal(b.getAttribute('aria-label'), name);
    assert.match(b.title, new RegExp(`^${name} \\(Ctrl\\+`));
    assert.equal(b.getAttribute('aria-keyshortcuts'), keys);
    assert.ok(b.querySelector('svg'), 'an icon');
  }
  const inSeq = () => !!m.d.querySelector('#seqList li[data-id=h001]');
  for (let i = 0; i < 51; i++) await m.click(m.tile('h001').querySelector('[data-act=toggle]')); // add, remove, ... add
  assert.equal(inSeq(), true);
  for (let i = 0; i < 60; i++) await m.key(m.d.body, 'z', { ctrlKey: true });
  assert.equal(m.$('#undoBtn').disabled, true);
  assert.equal(inSeq(), true, 'the oldest edit (the first add) fell off the stack');
  await m.key(m.d.body, 'Z', { ctrlKey: true, shiftKey: true });
  assert.equal(inSeq(), false, 'redo: the second edit, the remove');
});

// ---------- rail filmstrip (#261) ----------

function railPage(t, { saveData = false } = {}) {
  return open(t, {
    allowUnhandled: false,
    setup: (w) => {
      layout({ rail: 780 })(w);
      if (saveData) Object.defineProperty(w.navigator, 'connection', { configurable: true, value: { saveData: true } });
    },
  });
}

test('the filmstrip asks for keyframe stills once the view rests, three at a time, and stops after three failures', async (t) => {
  const m = await railPage(t);
  const film = m.$('#railFilm');
  assert.equal(film.getAttribute('aria-hidden'), 'true');
  assert.equal(film.parentElement.firstElementChild, film, 'the first layer under the markers');
  await m.click(m.tile('h001').querySelector('.cand-main'));
  const imgs = () => [...film.querySelectorAll('img')];
  assert.ok(imgs().length >= 2, 'one still per grid step of the view');
  assert.ok(imgs().every((i) => !i.getAttribute('src')), 'nothing is asked for while the view may still move');
  assert.ok(imgs().every((i) => /\/api\/projects\/p1\/sources\/s1\/frame\?t=x&at=[\d.]+&w=\d+&key=1$/.test(i.dataset.src)), imgs()[0].dataset.src);

  await m.tick(150);
  const asked = () => imgs().filter((i) => i.getAttribute('src'));
  assert.equal(asked().length, 3, 'at most three stills load at once');
  for (const img of asked()) img.dispatchEvent(new m.w.Event('error'));
  await m.page.settle();
  assert.equal(film.children.length, 0, 'three failures stop the strip for this source; the ticks show');
});

test('a data saver gets no filmstrip', async (t) => {
  const m = await railPage(t, { saveData: true });
  await m.click(m.tile('h001').querySelector('.cand-main'));
  await m.tick(150);
  assert.equal(m.$('#railFilm').children.length, 0);
});

// ---------- the theme before first paint (#445) ----------

/** Run index.html's inline <head> script (as a browser does, before app.css) with ``stored`` as the saved theme. */
function earlyTheme(stored) {
  const html = readFileSync(join(WEB_DIR, 'index.html'), 'utf8');
  const dom = new JSDOM(html, {
    url: `${ORIGIN}/`, runScripts: 'dangerously',
    beforeParse(w) { if (stored != null) w.localStorage.setItem('cutpoint.theme', stored); },
  });
  const d = dom.window.document;
  const out = { theme: d.documentElement.dataset.theme ?? null, kept: dom.window.localStorage.getItem('cutpoint.theme') };
  const head = [...d.head.children];
  out.beforeStyles = head.findIndex((el) => el.tagName === 'SCRIPT' && !el.src) < head.findIndex((el) => el.matches('link[rel=stylesheet][href="app.css"]'));
  dom.window.close();
  return out;
}

test('the inline head script sets the saved theme before the styles apply; an old "dark" or unknown value is Graphite and dropped', () => {
  assert.equal(earlyTheme('light').theme, 'light');
  assert.equal(earlyTheme('blue').theme, 'blue');
  assert.deepEqual(earlyTheme(null), { theme: null, kept: null, beforeStyles: true });
  for (const old of ['dark', 'neon']) {
    const r = earlyTheme(old);
    assert.equal(r.theme, null, old);
    assert.equal(r.kept, null, `${old} is removed`);
  }
  assert.equal(earlyTheme('light').beforeStyles, true);
});

// ---------- trim handles (#352) ----------

test('the trim handles move by 0.1 s, 1 s with Shift and 5 s with PageUp, name their time, and leave Space to play and pause', async (t) => {
  const m = await open(t);
  await m.click(m.tile('h001').querySelector('.cand-main')); // 10-18 s
  const start = m.$('#startHandle');
  const end = m.$('#endHandle');
  const now = (el) => Number(el.getAttribute('aria-valuenow'));
  start.focus();
  await m.key(start, 'ArrowRight');
  assert.equal(now(start), 10.1);
  assert.match(start.getAttribute('aria-valuetext'), /^Clip start, /);
  await m.key(start, 'ArrowRight', { shiftKey: true });
  assert.equal(now(start), 11.1);
  await m.key(end, 'PageUp');
  assert.equal(now(end), 23);
  assert.match(end.getAttribute('aria-valuetext'), /^Clip end, /);

  const v = m.$('#video');
  v.__media.loaded(600);
  await m.page.settle();
  const paused = v.paused;
  await m.key(start, ' ');
  assert.equal(v.paused, !paused, 'Space plays or pauses the stage');
  assert.equal(now(start), 11.1, 'and moves no handle');
});
