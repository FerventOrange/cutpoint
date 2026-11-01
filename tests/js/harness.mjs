// Test harness for the browser workspace: builds a jsdom window from the real
// src/cutpoint/web/index.html, installs the browser globals app.js reads, mocks fetch with
// per-route responses, stubs HTMLMediaElement, then imports the real app.js so its boot code runs.
//
//   const page = await boot({ routes: { 'GET /api/projects/p1': project } });
//   ... assert on page.document ...
//   page.close();
//
// Timers are Node's own (setTimeout, setInterval, requestAnimationFrame is routed to setTimeout),
// so `mock.timers.enable()` from node:test works; enable it BEFORE boot() to also catch the
// timers app.js starts while booting.
import { cp, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { JSDOM } from 'jsdom';

export const WEB_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'src', 'cutpoint', 'web');
export const ORIGIN = 'http://127.0.0.1:8765';
export const TOKEN = 'test-token';

// ---------- fixtures ----------

/** A minimal analyzed project as GET /api/projects/<id> returns it; ``over`` replaces fields. */
export function makeProject(over = {}) {
  const moment = (n, start, label, score) => ({
    id: `h00${n}`, cli_id: `h00${n}`, source: '/videos/a.mp4', start, end: start + 8, score, label,
    transcript: `line ${n}`,
  });
  return {
    id: 'p1', name: 'Project One', style: 'funny', kind: 'video', gaming: false,
    sources: [{ id: 's1', path: '/videos/a.mp4', name: 'a.mp4', status: 'ready', url: '/api/projects/p1/sources/s1?t=x', duration: 600 }],
    highlights: {
      style: 'funny',
      candidates: [moment(1, 10, 'Big laugh', 0.9), moment(2, 100, 'Fall', 0.7), moment(3, 300, 'Intro joke', 0.5)],
    },
    analysis: { video_s: 600, took_s: 30 },
    review: { segments: [] },
    ...over,
  };
}

/** Responses for every route app.js calls while booting with ``project`` open. */
export function bootRoutes(project = makeProject()) {
  return {
    'GET /api/session': { styles: ['funny', 'gaming', 'action'], can_reveal: false, can_pick: false },
    'GET /api/projects': { projects: project ? [{ id: project.id, name: project.name }] : [] },
    'GET /api/settings': { compress_peaks_set: false },
    'GET /api/jobs': { jobs: [] },
    ...(project && {
      [`GET /api/projects/${project.id}`]: project,
      [`GET /api/projects/${project.id}/assistant`]: { provider: null, consent: null, turns: [], running: [] },
      [`GET /api/projects/${project.id}/assistant/history`]: { applied: [] },
    }),
    'GET /api/assistant/logins': { logins: [] },
  };
}

// ---------- fetch mock ----------

/**
 * Route keys are "METHOD /path" (query ignored) or "/path" (any method); "METHOD /path?query"
 * matches the exact query. A value is a JSON body, or a function ``(req) => body | Response-like
 * {status, body}`` (wrap as ``reply(status, body)``). Unmatched requests are recorded in
 * ``unhandled`` and answered with a 501, and ``page.assertNoUnhandled()`` throws if any happened.
 */
export const reply = (status, body) => ({ __reply: true, status, body });

function makeFetch(routes, log, unhandled) {
  return async function fetchMock(input, init = {}) {
    const url = new URL(typeof input === 'string' ? input : input.url, ORIGIN);
    const method = (init.method ?? 'GET').toUpperCase();
    const req = {
      method, path: url.pathname, query: url.search, url: url.pathname + url.search,
      headers: init.headers ?? {}, body: init.body === undefined ? undefined : JSON.parse(init.body),
    };
    log.push(req);
    const keys = [`${method} ${req.url}`, `${method} ${req.path}`, req.url, req.path];
    const key = keys.find((k) => Object.hasOwn(routes, k));
    if (key === undefined) {
      unhandled.push(`${method} ${req.url}`);
      return new Response(JSON.stringify({ detail: `unhandled test route ${method} ${req.url}` }), {
        status: 501, headers: { 'content-type': 'application/json' },
      });
    }
    let out = routes[key];
    if (typeof out === 'function') out = await out(req);
    const { status, body } = out?.__reply ? out : { status: 200, body: out };
    if (status === 204 || body === undefined) return new Response(null, { status: status === 200 ? 204 : status });
    return new Response(JSON.stringify(body), { status, headers: { 'content-type': 'application/json' } });
  };
}

// ---------- HTMLMediaElement stubs ----------

/**
 * jsdom implements no media playback: play/pause/load only log "not implemented". These stubs keep
 * the state app.js reads (paused, currentTime, duration, volume, muted, playbackRate, readyState,
 * ended) and fire the usual events. Drive a video from a test with ``el.__media``:
 *   el.__media.loaded(120)   // duration known: loadedmetadata + durationchange + canplay
 *   el.__media.tick(12.5)    // currentTime = 12.5, fires timeupdate
 *   el.__media.end()         // paused + ended
 * ``el.__media.calls`` lists play/pause/load calls in order.
 */
function installMediaStubs(window) {
  const proto = window.HTMLMediaElement.prototype;
  const state = (el) => {
    if (!el.__media) {
      const m = {
        paused: true, ended: false, currentTime: 0, duration: NaN, readyState: 0, volume: 1, muted: false,
        playbackRate: 1, calls: [],
        fire: (type) => el.dispatchEvent(new window.Event(type)),
        loaded(duration) {
          m.duration = duration; m.readyState = 4;
          for (const t of ['loadedmetadata', 'durationchange', 'loadeddata', 'canplay']) m.fire(t);
        },
        tick(t) { m.currentTime = t; m.fire('timeupdate'); },
        end() { m.paused = true; m.ended = true; m.fire('pause'); m.fire('ended'); },
      };
      Object.defineProperty(el, '__media', { value: m });
    }
    return el.__media;
  };
  const prop = (name, { writable = true, onSet } = {}) =>
    Object.defineProperty(proto, name, {
      configurable: true,
      get() { return state(this)[name]; },
      set: writable ? function (v) { state(this)[name] = v; onSet?.(this, v); } : undefined,
    });
  prop('paused', { writable: false });
  prop('ended', { writable: false });
  prop('duration', { writable: false });
  prop('readyState', { writable: false });
  prop('currentTime', { onSet: (el) => { state(el).ended = false; state(el).fire('seeking'); state(el).fire('seeked'); } });
  prop('volume', { onSet: (el) => state(el).fire('volumechange') });
  prop('muted', { onSet: (el) => state(el).fire('volumechange') });
  prop('playbackRate', { onSet: (el) => state(el).fire('ratechange') });
  Object.defineProperties(proto, {
    play: { configurable: true, value() {
      const m = state(this); m.calls.push('play'); m.paused = false; m.ended = false;
      m.fire('play'); m.fire('playing');
      return Promise.resolve();
    } },
    pause: { configurable: true, value() {
      const m = state(this); m.calls.push('pause');
      if (!m.paused) { m.paused = true; m.fire('pause'); }
    } },
    load: { configurable: true, value() { const m = state(this); m.calls.push('load'); m.readyState = 0; m.duration = NaN; m.fire('emptied'); } },
    canPlayType: { configurable: true, value() { return 'maybe'; } },
  });
}

// ---------- globals ----------

// Names app.js and its modules use bare, taken from the jsdom window. Everything else on the window
// that Node lacks is copied too (below); Node's own timers, fetch, Response and structuredClone stay.
const ALWAYS = ['window', 'document', 'location', 'history', 'localStorage', 'sessionStorage', 'navigator',
  'addEventListener', 'removeEventListener', 'dispatchEvent', 'getComputedStyle', 'matchMedia',
  'confirm', 'alert', 'prompt', 'open', 'scrollTo', 'getSelection'];
const KEEP_NODE = new Set(['setTimeout', 'clearTimeout', 'setInterval', 'clearInterval', 'setImmediate',
  'clearImmediate', 'queueMicrotask', 'fetch', 'Response', 'Request', 'Headers', 'structuredClone',
  'URL', 'URLSearchParams', 'AbortController', 'AbortSignal', 'TextEncoder', 'TextDecoder', 'performance',
  'console', 'Event', 'EventTarget', 'CustomEvent', 'Blob', 'FormData', 'File', 'atob', 'btoa']);

function installGlobals(window, extra) {
  const saved = new Map(); // name -> original descriptor (or undefined)
  const set = (name, value) => {
    if (!saved.has(name)) saved.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
    Object.defineProperty(globalThis, name, { configurable: true, writable: true, enumerable: true, value });
  };
  // every window property Node lacks (innerWidth, Element, ...), as a live view of the window
  const names = new Set([...ALWAYS, ...Object.getOwnPropertyNames(window).filter((n) => !(n in globalThis))]);
  for (const n of names) {
    if (KEEP_NODE.has(n) || !(n in window)) continue;
    if (!saved.has(n)) saved.set(n, Object.getOwnPropertyDescriptor(globalThis, n));
    Object.defineProperty(globalThis, n, {
      configurable: true,
      enumerable: true,
      get() {
        const v = window[n];
        return typeof v === 'function' && !/^[A-Z]/.test(n) ? v.bind(window) : v;
      },
      set(v) { window[n] = v; },
    });
  }
  // DOM Event classes must be jsdom's, so dispatchEvent accepts them
  for (const n of ['Event', 'CustomEvent']) set(n, window[n]);
  for (const [n, v] of Object.entries(extra)) set(n, v);
  return () => {
    for (const [name, desc] of saved) {
      if (desc) Object.defineProperty(globalThis, name, desc);
      else delete globalThis[name];
    }
  };
}

// ---------- boot ----------

let bootN = 0;
let current = null; // the open page; boot() closes it before installing globals again

/**
 * Build the page and run app.js against the mocked backend. Options:
 *   routes   fetch routes, merged over ``bootRoutes(project)`` (see makeFetch above)
 *   project  the open project fixture, or null for "no projects yet" (default makeProject())
 *   hash     location hash before boot, e.g. '#t=abc' or '#p1' (default none)
 *   token    session token put in sessionStorage (default TOKEN); null for none
 *   storage  localStorage entries, e.g. { 'cutpoint.theme': 'dark' }
 *   settle   wait for the boot chain to finish (default true)
 *   allowUnhandled  do not fail close() when a request matched no route (default false)
 *   entry    the module to import (default 'app.js'; 'api.js' tests the client alone); its exports are ``page.mod``
 * Returns { window, document, routes, requests, unhandled, settle, assertNoUnhandled, close }.
 * close() throws if a request had no route, so a forgotten mock cannot pass silently.
 * app.js keeps module state, so each boot loads a fresh copy of the web directory; call close().
 */
export async function boot({ routes = {}, project = makeProject(), hash = '', token = TOKEN, storage = {}, settle = true, entry = 'app.js', allowUnhandled = false } = {}) {
  const html = readFileSync(join(WEB_DIR, 'index.html'), 'utf8');
  const dom = new JSDOM(html, { url: `${ORIGIN}/${hash}`, pretendToBeVisual: true, contentType: 'text/html' });
  const { window } = dom;
  if (token !== null) window.sessionStorage.setItem('cutpoint-token', token);
  for (const [k, v] of Object.entries(storage)) window.localStorage.setItem(k, v);
  installMediaStubs(window);
  // layout does not exist in jsdom; the observers only need to be constructible
  const Noop = class { observe() {} unobserve() {} disconnect() {} takeRecords() { return []; } };
  window.ResizeObserver ??= Noop;
  window.IntersectionObserver ??= Noop;
  window.matchMedia ??= (q) => ({ matches: false, media: q, addEventListener() {}, removeEventListener() {}, addListener() {}, removeListener() {} });
  window.CSS ??= { supports: () => false, escape: (s) => String(s).replace(/[^\w-]/g, '\\$&') };
  window.HTMLElement.prototype.scrollIntoView ??= function () {};
  window.HTMLElement.prototype.setPointerCapture ??= function () {};
  window.HTMLElement.prototype.releasePointerCapture ??= function () {};
  window.HTMLDialogElement?.prototype && (window.HTMLDialogElement.prototype.showModal ??= function () { this.setAttribute('open', ''); });
  window.HTMLDialogElement?.prototype && (window.HTMLDialogElement.prototype.close ??= function () { this.removeAttribute('open'); });

  const all = { ...bootRoutes(project), ...routes };
  const requests = [];
  const unhandled = [];
  const fetchMock = makeFetch(all, requests, unhandled);
  window.fetch = fetchMock;
  window.confirm = () => true;
  window.requestAnimationFrame = (fn) => setTimeout(() => fn(performance.now()), 16);
  window.cancelAnimationFrame = (id) => clearTimeout(id);

  // confirm, requestAnimationFrame and cancelAnimationFrame are live views of the window (see installGlobals),
  // so a test can override them on page.window and bare calls in app.js see the override
  current?.close({ check: false }); // a test that never closed its page must not leave globals behind
  const restore = installGlobals(window, { fetch: fetchMock });

  // fresh module graph per boot: a copy of the real files in a temp dir (api.js reads the token at load)
  const dir = await mkdtemp(join(tmpdir(), 'cutpoint-web-'));
  await cp(WEB_DIR, dir, { recursive: true });
  await writeFile(join(dir, 'package.json'), '{"type":"module"}');
  bootN += 1;

  const page = {
    window, document: window.document, routes: all, requests, unhandled,
    /** Let pending promise chains (fetch -> render) finish; with mock timers it never waits on a timer. */
    async settle(rounds = 25) {
      for (let i = 0; i < rounds; i++) await new Promise((r) => setImmediate(r));
    },
    assertNoUnhandled() {
      if (unhandled.length) throw new Error(`unhandled fetch routes:\n  ${unhandled.join('\n  ')}`);
    },
    /** Teardown happens once; throws every time it is asked if a request had no route, unless ``allowUnhandled`` or ``check: false``. */
    close({ check = true } = {}) {
      if (!closed) {
        closed = true;
        if (current === page) current = null;
        dom.window.close();
        restore();
        rm(dir, { recursive: true, force: true }).catch(() => {});
      }
      if (check && !allowUnhandled) page.assertNoUnhandled();
    },
  };
  let closed = false;
  current = page;
  try {
    page.mod = await import(`${pathToFileURL(join(dir, entry)).href}?boot=${bootN}`);
    if (settle) await page.settle();
  } catch (e) {
    page.close({ check: false });
    throw e;
  }
  return page;
}

/**
 * Import a pure web module (util.js) without booting the page: a copy of the web directory is made
 * in a temp dir, because the source tree has no package.json marking it as ES modules.
 */
export async function importPure(name) {
  const dir = await mkdtemp(join(tmpdir(), 'cutpoint-web-pure-'));
  try {
    await cp(WEB_DIR, dir, { recursive: true });
    await writeFile(join(dir, 'package.json'), '{"type":"module"}');
    return await import(pathToFileURL(join(dir, name)).href);
  } finally {
    rm(dir, { recursive: true, force: true }).catch(() => {});
  }
}
