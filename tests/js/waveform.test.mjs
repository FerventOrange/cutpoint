// The trim rail's waveform (GET .../envelope) and event markers (GET .../events): the envelope is
// fetched once per source and drawn as one aria-hidden SVG path for the view; the markers show the
// detected events of the source, and the other moments only for a source without any.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { boot, makeProject, reply, WEB_DIR } from './harness.mjs';

const ENVELOPE = '/api/projects/p1/sources/s1/envelope';
const EVENTS = '/api/projects/p1/events';
const SRC = '/videos/a.mp4';

/** 600 s at 0.05 s per value: quiet (0.1) except a loud burst (0.9) from 20 s to 25 s. */
const peaks = () => Array.from({ length: 12000 }, (_, i) => (i >= 400 && i < 500 ? 0.9 : 0.1));
const ev = (over) => ({ source: SRC, start: 30, end: 31, kind: 'audio', type: 'audio', label: 'Laughter', confidence: 0.9, ...over });

async function open(t, routes, project = makeProject()) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ project, routes });
  t.after(() => page.close());
  const d = page.document;
  const m = {
    page, d,
    $: (s) => d.querySelector(s),
    zoom: async (span) => { d.querySelector(`#zoomPresets [data-span="${span}"]`).click(); await page.settle(); },
    marks: () => [...d.querySelectorAll('#railEvents .rail-event')],
    bars: () => [...(m.$('#railWave path').getAttribute('d') ?? '').matchAll(/M([\d.]+) ([\d.]+)V/g)].map((x) => Number(x[2])),
    calls: (path) => page.requests.filter((r) => r.path === path).length,
  };
  await page.settle();
  return m;
}

test('the waveform is one hidden SVG path for the view, tall where the audio is loud', async (t) => {
  const m = await open(t, { [`GET ${ENVELOPE}`]: { step: 0.05, peaks: peaks() } });
  await m.zoom(120);
  const svg = m.$('#railWave');
  assert.equal(svg.getAttribute('aria-hidden'), 'true');
  const cols = Number(svg.getAttribute('viewBox').split(' ')[2]);
  assert.equal(cols, 200); // the 600 px rail, 3 px a bar
  const tops = m.bars();
  assert.equal(tops.length, cols);
  assert.ok(Math.min(...tops) < 10, 'the burst reaches nearly the whole height');
  assert.ok(Math.max(...tops) > 30, 'the quiet parts stay low');
});

test('the envelope is fetched once per source; a new view redraws it from memory', async (t) => {
  const m = await open(t, { [`GET ${ENVELOPE}`]: { step: 0.05, peaks: peaks() } });
  await m.zoom(120);
  const before = m.$('#railWave path').getAttribute('d');
  await m.zoom(30);
  assert.notEqual(m.$('#railWave path').getAttribute('d'), before);
  assert.equal(m.calls(ENVELOPE), 1);
  m.page.assertNoUnhandled();
});

test('an envelope still decoding (202) is asked for again after 5 s until it is ready', async (t) => {
  let asked = 0;
  const m = await open(t, { [`GET ${ENVELOPE}`]: () => (++asked < 3 ? reply(202, { pending: true }) : { step: 0.05, peaks: peaks() }) });
  await m.zoom(120);
  assert.equal(m.$('#railWave path').getAttribute('d'), null);
  assert.equal(m.calls(ENVELOPE), 1);
  for (const n of [2, 3]) {
    t.mock.timers.tick(5000);
    await m.page.settle();
    assert.equal(m.calls(ENVELOPE), n);
  }
  assert.equal(m.bars().length, 200);
  t.mock.timers.tick(20000);
  await m.page.settle();
  assert.equal(m.calls(ENVELOPE), 3, 'no more asks once it is drawn');
});

test('leaving a source during a 202 ends its retry chain, so the source on screen decodes first (#760)', async (t) => {
  const project = makeProject();
  project.sources.push({ id: 's2', path: '/videos/b.mp4', name: 'b.mp4', status: 'ready', url: '/api/projects/p1/sources/s2?t=x', duration: 600 });
  project.highlights.candidates[1].source = '/videos/b.mp4';
  const OTHER = '/api/projects/p1/sources/s2/envelope';
  const m = await open(t, { [`GET ${ENVELOPE}`]: reply(202, { pending: true }), [`GET ${OTHER}`]: reply(202, { pending: true }) }, project);
  await m.zoom(120);
  assert.equal(m.calls(ENVELOPE), 1);
  m.$('#candList li[data-id=h002] [data-act=focus]').click(); // another source
  await m.page.settle();
  assert.equal(m.calls(OTHER), 1);
  t.mock.timers.tick(5000);
  await m.page.settle();
  assert.equal(m.calls(ENVELOPE), 1, 'the source left is not asked for again');
  assert.equal(m.calls(OTHER), 2, 'the source on screen keeps its chain');
});

test('in Cut mode the other side of the cut keeps asking for its envelope during a 202', async (t) => {
  const project = makeProject();
  project.sources.push({ id: 's2', path: '/videos/b.mp4', name: 'b.mp4', status: 'ready', url: '/api/projects/p1/sources/s2?t=x', duration: 600 });
  const [c1, c2] = project.highlights.candidates;
  c2.source = '/videos/b.mp4';
  project.review = { segments: [c1, c2].map((c) => ({ id: c.id, source: c.source, start: c.start, end: c.end, label: c.label })) };
  const OTHER = '/api/projects/p1/sources/s2/envelope';
  const m = await open(t, { [`GET ${ENVELOPE}`]: reply(202, { pending: true }), [`GET ${OTHER}`]: reply(202, { pending: true }) }, project);
  m.$('#stageMode [data-mode=cut]').click();
  await m.page.settle();
  assert.equal(m.calls(OTHER), 1, 'the side not in focus asks once');
  t.mock.timers.tick(5000);
  await m.page.settle();
  assert.equal(m.calls(OTHER), 2, 'and again after 5 s: it is on screen');
  assert.equal(m.calls(ENVELOPE), 2);
  t.mock.timers.tick(5000);
  await m.page.settle();
  assert.equal(m.calls(OTHER), 3, 'the retry chain goes on');
  assert.equal(m.calls(ENVELOPE), 3);
});

test('closing a project and opening it again within 5 s leaves one retry chain, not two', async (t) => {
  const m = await open(t, {
    [`GET ${ENVELOPE}`]: reply(202, { pending: true }),
    'GET /api/projects': { projects: [{ id: 'p1', name: 'Project One' }] },
    'GET /api/settings': { projects_dir: '/projects', projects_dir_locked: false, config_file: '/c.json', compress_peaks_set: false },
    'GET /api/projects/p1/delete-plan': { name: 'Project One', size: 1, files: 1, deleted: [], deleted_count: 0, kept: [], kept_count: 0 },
    'DELETE /api/projects/p1': { deleted: 'p1', kept: 0 },
  });
  await m.zoom(120);
  assert.equal(m.calls(ENVELOPE), 1);
  m.$('#settingsBtn').click();
  await m.page.settle();
  m.$('#setDelete').click();
  await m.page.settle();
  const dlg = m.$('#confirmDialog');
  dlg.returnValue = 'ok';
  dlg.removeAttribute('open');
  dlg.dispatchEvent(new m.page.window.Event('close'));
  await m.page.settle();
  assert.equal(m.$('#projectName').textContent, 'Choose project', 'closed');
  m.$('#projectBtn').click();
  m.$('#projectMenu [data-id=p1]').click();
  await m.page.settle();
  await m.zoom(120);
  assert.equal(m.calls(ENVELOPE), 2, 'the reopened project asks again at once');
  t.mock.timers.tick(5000);
  await m.page.settle();
  assert.equal(m.calls(ENVELOPE), 3, 'one chain asks, not the old one as well');
});

test('a source without audio draws no waveform', async (t) => {
  const none = await open(t, { [`GET ${ENVELOPE}`]: { step: 0.05, peaks: [] } });
  assert.equal(none.$('#railWave path').getAttribute('d'), null);
  assert.equal(none.calls(ENVELOPE), 1);
});

test('events of the source replace the moment markers: labels, tooltips, most confident first, EVENT_GAP apart', async (t) => {
  const events = [
    ev({ start: 30, label: 'Laughter', confidence: 0.9 }),
    ev({ start: 35, label: 'Cheering', confidence: 0.4 }), // 5 s = 25 px from the first: dropped
    ev({ start: 70, end: 70, kind: 'ocr', type: 'kill', label: 'A>B', confidence: 0.8 }),
    ev({ start: 100, end: 100, kind: 'telemetry', type: 'crash', label: 'crash', confidence: null }),
    ev({ source: '/videos/other.mp4', start: 50, label: 'Elsewhere' }),
  ];
  const m = await open(t, { [`GET ${EVENTS}`]: { events } });
  await m.zoom(120);
  assert.deepEqual(m.marks().map((e) => e.textContent), ['Laughter', 'A>B', 'crash']);
  assert.deepEqual(m.marks().map((e) => e.dataset.ev), ['audio', 'ocr', 'telemetry']);
  assert.match(m.marks()[0].title, /^Sound: Laughter, .*confidence 0\.9$/);
  assert.match(m.marks()[1].title, /^On-screen text: A>B, /);
  assert.doesNotMatch(m.marks()[2].title, /confidence/);
  assert.ok(m.marks().every((e) => !e.dataset.kind), 'not the moment markers');
  const text = m.$('#railEventsText');
  assert.equal(m.$('#railEvents').getAttribute('aria-hidden'), 'true');
  assert.match(text.textContent, /^Sound: Laughter at .*; On-screen text: A>B at .*; Camera: crash at /);
});

test('a source with no events keeps the other-moment markers', async (t) => {
  const m = await open(t, { [`GET ${EVENTS}`]: { events: [ev({ source: '/videos/other.mp4' })] } });
  await m.zoom(120);
  assert.ok(m.marks().map((e) => e.textContent).includes('Fall'));
  assert.ok(m.marks().every((e) => e.dataset.kind && !e.dataset.ev));
});

test('a failing events call leaves the other-moment markers', async (t) => {
  const m = await open(t, { [`GET ${EVENTS}`]: reply(500, { detail: 'boom' }) });
  await m.zoom(120);
  assert.ok(m.marks().map((e) => e.textContent).includes('Fall'));
});

test('an events response for an older events_sig arriving late does not overwrite the newer events', async (t) => {
  let release;
  const gate = new Promise((r) => { release = r; });
  let projects = 0;
  let asked = 0;
  const m = await open(t, {
    // the first read (boot) has sig 'a', every later one sig 'b'
    'GET /api/projects': { projects: [{ id: 'p1', name: 'Project One' }, { id: 'p2', name: 'Project Two' }] },
    'GET /api/projects/p1': () => ({ ...makeProject(), events_sig: ++projects === 1 ? 'a' : 'b' }),
    // the menu's other entry reads the same project again (same id), now with sig 'b'
    'GET /api/projects/p2': () => ({ ...makeProject(), events_sig: 'b' }),
    [`GET ${EVENTS}`]: async () => {
      if (++asked === 1) {
        await gate; // the answer for sig 'a' is slow
        return { events: [ev({ label: 'Stale' })] };
      }
      return { events: [ev({ label: 'Fresh' })] };
    },
  });
  await m.zoom(120);
  assert.equal(asked, 1);
  m.$('#projectBtn').click();
  m.$('#projectMenu [data-id=p2]').click(); // the project is read again: sig 'b'
  await m.page.settle();
  assert.equal(asked, 2);
  await m.zoom(120); // opening a project resets the view
  assert.deepEqual(m.marks().map((e) => e.textContent), ['Fresh']);
  release();
  await m.page.settle();
  assert.deepEqual(m.marks().map((e) => e.textContent), ['Fresh'], 'the late older answer is dropped');
});

test('a talk stretch is a range labelled with its title, as wide as its length in the view', async (t) => {
  const events = [ev({ start: 60, end: 90, kind: 'text', type: 'dialogue', label: 'Mia drops the cup', confidence: 0.7 })];
  const m = await open(t, { [`GET ${EVENTS}`]: { events } });
  await m.zoom(120);
  const [talk] = m.marks();
  assert.equal(talk.textContent, 'Mia drops the cup');
  assert.equal(talk.dataset.ev, 'text');
  assert.ok(talk.classList.contains('range'));
  assert.match(talk.getAttribute('style'), /width: 25(\.0+)?%/);
  assert.match(talk.title, /^Talk: Mia drops the cup, .* to .*, confidence 0\.7$/);
  assert.match(m.$('#railEventsText').textContent, /^Talk: Mia drops the cup at /);
});

// ---- contrast of the marker colours ----

const css = readFileSync(join(WEB_DIR, 'app.css'), 'utf8');
const block = (head) => {
  const at = css.indexOf(`${head} {`);
  assert.ok(at >= 0, head);
  return css.slice(at, css.indexOf('\n}', at));
};
const token = (name, ...blocks) => {
  for (const b of blocks) {
    const m = new RegExp(String.raw`${name}:\s*([^;]+);`).exec(b);
    if (!m) continue;
    const v = m[1].trim();
    return v.startsWith('var(') ? token(v.slice(4, -1), ...blocks) : v;
  }
  throw new Error(name);
};
const lum = (hex) => {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255)
    .map((c) => (c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4));
  return 0.2126 * r + 0.7152 * g + 0.0722 * b;
};
const ratio = (a, b) => { const [x, y] = [lum(a), lum(b)].sort((p, q) => q - p); return (x + 0.05) / (y + 0.05); };

test('the event marker bars are 3:1 on the stage ground in Graphite, Blue and Light', () => {
  const root = block(':root');
  const themes = {
    graphite: [[root], '--center'],
    blue: [[block(':root[data-theme="blue"]'), root], '--slate'],
    light: [[block(':root[data-theme="light"]'), root], '--slate'],
  };
  for (const [name, [blocks, ground]] of Object.entries(themes)) {
    const bg = token(ground, ...blocks);
    for (const k of ['--ev-sound', '--ev-text', '--ev-cam', '--ev-talk']) {
      assert.ok(ratio(token(k, ...blocks), bg) >= 3, `${name} ${k}`);
    }
  }
});

test('the moment kind colours are 3:1 on the stage ground in Graphite, Blue and Light, and on their tint', () => {
  const root = block(':root');
  const themes = {
    graphite: [[root], '--center'],
    blue: [[block(':root[data-theme="blue"]'), root], '--slate'],
    light: [[block(':root[data-theme="light"]'), root], '--slate'],
  };
  for (const [name, [blocks, ground]] of Object.entries(themes)) {
    const bg = token(ground, ...blocks);
    for (const k of ['action', 'scenic', 'funny', 'dialogue', 'crash']) {
      assert.ok(ratio(token(`--k-${k}`, ...blocks), bg) >= 3, `${name} --k-${k} on the ground`);
      // Light darkens the colours for its pale ground; the bars and dots on the kind's dark tint (no theme changes it) must stay 3:1
      if (name === 'light') assert.ok(ratio(token(`--k-${k}`, ...blocks), token(`--k-${k}-tint`, ...blocks)) >= 3, `${name} --k-${k} on its tint`);
    }
  }
});

test('the crash badge text is 4.5:1 on the tile grounds in Graphite, Blue and Light', () => {
  const root = block(':root');
  const themes = { graphite: [root], blue: [block(':root[data-theme="blue"]'), root], light: [block(':root[data-theme="light"]'), root] };
  for (const [name, blocks] of Object.entries(themes)) {
    for (const bg of ['--slate', '--panel', '--deep']) {
      assert.ok(ratio(token('--crash-text', ...blocks), token(bg, ...blocks)) >= 4.5, `${name} --crash-text on ${bg}`);
    }
  }
});
