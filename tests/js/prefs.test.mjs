// Browser-stored preferences (#1085): readStore/writeStore survive blocked storage and bad JSON
// (A425), candidate ids are never remembered across projects (A284), and the Settings check boxes
// share one stored-toggle helper that keeps each key and format (A424).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, bootRoutes, importPure, makeProject } from './harness.mjs';

const u = await importPure('util.js');

/** Run ``fn`` with a bare ``localStorage`` global of ``store``; the previous one is put back. */
function withStorage(store, fn) {
  const had = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, writable: true, value: store });
  try {
    return fn();
  } finally {
    if (had) Object.defineProperty(globalThis, 'localStorage', had);
    else delete globalThis.localStorage;
  }
}

function memoryStorage(init = {}) {
  const m = new Map(Object.entries(init));
  return {
    map: m,
    getItem: (k) => (m.has(k) ? m.get(k) : null),
    setItem: (k, v) => m.set(k, String(v)),
    removeItem: (k) => m.delete(k),
  };
}

const blocked = () => {
  throw new Error('SecurityError: storage is disabled');
};

// ---- readStore / writeStore (A425) ----

test('readStore gives the stored string, the parsed JSON with json, and the fallback when absent', () => {
  withStorage(memoryStorage({ a: '1', o: '{"volume":0.5}' }), () => {
    assert.equal(u.readStore('a'), '1');
    assert.equal(u.readStore('missing'), null);
    assert.equal(u.readStore('missing', 'x'), 'x');
    assert.deepEqual(u.readStore('o', null, { json: true }), { volume: 0.5 });
    assert.equal(u.readStore('o'), '{"volume":0.5}', 'raw without json');
  });
});

test('readStore returns the fallback for text that is not JSON', () => {
  withStorage(memoryStorage({ bad: '{not json' }), () => {
    assert.deepEqual(u.readStore('bad', { d: 1 }, { json: true }), { d: 1 });
    assert.equal(u.readStore('bad', null, { json: true }), null);
  });
});

test('writeStore stores strings as is, objects as JSON, and removes on null', () => {
  const s = memoryStorage({ gone: 'x' });
  withStorage(s, () => {
    assert.equal(u.writeStore('k', '0'), true);
    assert.equal(s.map.get('k'), '0');
    u.writeStore('o', { seq: 'in', ready: true });
    assert.equal(s.map.get('o'), '{"seq":"in","ready":true}');
    u.writeStore('gone', null);
    assert.equal(s.map.has('gone'), false);
    u.writeStore('o', undefined);
    assert.equal(s.map.has('o'), false);
  });
});

test('readStore and writeStore survive a storage that throws on every call', () => {
  withStorage({ getItem: blocked, setItem: blocked, removeItem: blocked }, () => {
    assert.equal(u.readStore('k', 'fb'), 'fb');
    assert.equal(u.readStore('k', 7, { json: true }), 7);
    assert.equal(u.writeStore('k', 'v'), false);
    assert.equal(u.writeStore('k', null), false);
  });
});

test('readStore and writeStore survive a storage that cannot even be reached', () => {
  const had = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', { configurable: true, get: blocked });
  try {
    assert.equal(u.readStore('k', 'fb'), 'fb');
    assert.equal(u.writeStore('k', 'v'), false);
  } finally {
    if (had) Object.defineProperty(globalThis, 'localStorage', had);
    else delete globalThis.localStorage;
  }
});

// ---- candidate ids (A284) ----

const OPTS = 'cutpoint.exportOptions';
const P2 = makeProject({ id: 'p2', name: 'Second trip' });

async function twoProjects(t, storage = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({
    storage,
    routes: { ...bootRoutes(P2), 'GET /api/projects': { projects: [{ id: 'p1', name: 'Project One' }, { id: 'p2', name: 'Second trip' }] } },
  });
  t.after(() => page.close());
  return page;
}

test('candidate ids are not stored with the export options, and older stored ids are ignored and dropped', async (t) => {
  const page = await twoProjects(t, { [OPTS]: JSON.stringify({ ids: 'h001, h002', top: '5' }) });
  const $ = (s) => page.document.querySelector(s);
  assert.equal($('#optIds').value, '', 'ids of an earlier project are not pre-filled');
  assert.equal($('#optTop').value, '5', 'the other fields still load');
  const ids = $('#optIds');
  ids.value = 'h003';
  ids.dispatchEvent(new page.window.Event('change', { bubbles: true }));
  const top = $('#optTop');
  top.value = '6';
  top.dispatchEvent(new page.window.Event('change', { bubbles: true }));
  const stored = JSON.parse(page.window.localStorage.getItem(OPTS));
  assert.deepEqual(stored, { top: '6' }, 'no ids key, the old one dropped too');
});

test('candidate ids are cleared when another project opens', async (t) => {
  const page = await twoProjects(t);
  const $ = (s) => page.document.querySelector(s);
  $('#optIds').value = 'h001, h003';
  $('#optIds').dispatchEvent(new page.window.Event('input', { bubbles: true }));
  $('#projectBtn').click();
  $('#projectMenu [data-id=p2]').click();
  await page.settle();
  assert.equal($('#projectName').textContent, 'Second trip');
  assert.equal($('#optIds').value, '');
});

// ---- the stored Settings toggles (A424) ----

async function page(t, { storage = {}, setup } = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const p = await boot({ storage, setup });
  t.after(() => p.close());
  return p;
}

function flip(p, sel, on) {
  const box = p.document.querySelector(sel);
  box.checked = on;
  box.dispatchEvent(new p.window.Event('change', { bubbles: true }));
}

test('each toggle keeps its key and format: "1" while on for the off-by-default ones, "0" while off for Single-key shortcuts', async (t) => {
  const p = await page(t);
  const ls = p.window.localStorage;
  const d = p.document;
  for (const [sel, key] of [['#setNerdStats', 'cutpoint.nerdStats'], ['#setAdvExport', 'cutpoint.advancedExport'], ['#setVerboseMoments', 'cutpoint.verboseMoments']]) {
    assert.equal(ls.getItem(key), null, `${key} absent by default`);
    flip(p, sel, true);
    assert.equal(ls.getItem(key), '1');
    flip(p, sel, false);
    assert.equal(ls.getItem(key), null);
  }
  assert.equal(ls.getItem('cutpoint.singleKeys'), null);
  flip(p, '#setSingleKeys', false);
  assert.equal(ls.getItem('cutpoint.singleKeys'), '0');
  assert.equal(d.querySelector('#startHandle').title, 'Drag, or use the arrow keys.', 'the hints follow');
  flip(p, '#setSingleKeys', true);
  assert.equal(ls.getItem('cutpoint.singleKeys'), null);
  assert.match(d.querySelector('#startHandle').title, /I sets it at the playhead/);
});

test('the toggles load their stored state and apply it at boot', async (t) => {
  const p = await page(t, { storage: { 'cutpoint.advancedExport': '1', 'cutpoint.verboseMoments': '1', 'cutpoint.singleKeys': '0' } });
  const d = p.document;
  assert.ok(d.querySelector('#panelExport').classList.contains('show-adv'));
  assert.ok(!d.querySelector('#candList').classList.contains('moments-compact'));
  assert.equal(d.querySelector('#startHandle').title, 'Drag, or use the arrow keys.');
  flip(p, '#setVerboseMoments', false);
  assert.ok(d.querySelector('#candList').classList.contains('moments-compact'));
  flip(p, '#setAdvExport', false);
  assert.ok(!d.querySelector('#panelExport').classList.contains('show-adv'));
});

test('with storage blocked the page boots on the defaults and the toggles still apply for the session', async (t) => {
  const p = await page(t, {
    setup: (w) => Object.defineProperty(w, 'localStorage', { configurable: true, get: blocked }),
  });
  const d = p.document;
  assert.ok(d.querySelector('#candList').classList.contains('moments-compact'), 'Verbose off');
  assert.ok(!d.querySelector('#panelExport').classList.contains('show-adv'), 'advanced options hidden');
  assert.match(d.querySelector('#startHandle').title, /I sets it at the playhead/, 'Single-key shortcuts on');
  flip(p, '#setVerboseMoments', true);
  assert.ok(!d.querySelector('#candList').classList.contains('moments-compact'));
  flip(p, '#setSingleKeys', false);
  assert.equal(d.querySelector('#startHandle').title, 'Drag, or use the arrow keys.');
});
