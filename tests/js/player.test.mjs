// The stage player with window proxies (#323): which file loads, where playback resumes after a
// wait, one retry then a failed state, and moving on at a window's end.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject, reply } from './harness.mjs';

const prepareRoute = 'POST /api/projects/p1/sources/s1/prepare';

/** A project whose only source still builds its proxy; ``src`` overrides the source's listing. */
function proxyProject(src = {}) {
  const p = makeProject();
  p.sources[0] = { ...p.sources[0], status: 'needs-proxy', windows: [], pending: [], failed: [], ...src };
  return p;
}
const win = (lo, hi) => ({ lo, hi, url: `/api/projects/p1/sources/s1/window/${lo}-${hi}.mp4?t=x` });

/**
 * Boot with mock timers (setTimeout only, which also drives requestAnimationFrame). ``listing`` is
 * the source the project route returns now; assign ``player.listing = {...}`` to change it.
 * ``prepares`` records the window requests (``{lo, hi}``).
 */
async function player(t, src = {}, extra = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const state = { src };
  const project = () => ({ ...proxyProject(state.src), ...extra.project });
  const prepares = [];
  const page = await boot({
    project: project(),
    routes: {
      'GET /api/projects/p1': () => project(),
      [prepareRoute]: (req) => {
        if (!req.body) return {};
        prepares.push(req.body);
        // like the server: the whole proxy is queued too, and the requested window is listed as
        // encoding (unless the test says it vanished)
        state.src = { ...state.src, status: state.src.status === 'ready' ? 'ready' : 'preparing' };
        if (!state.vanish) state.src.pending = [{ ...req.body, state: 'encoding' }];
        return extra.prepare ? extra.prepare(req) : {};
      },
    },
  });
  t.after(() => page.close());
  const d = page.document;
  const v = d.querySelector('#video');
  const api = {
    page, d, v, prepares, state,
    msg: () => d.querySelector('#screenMsg').textContent,
    playLabel: () => d.querySelector('#playBtn').getAttribute('aria-label'),
    retryShown: () => !d.querySelector('#retryBtn').hidden,
    /** Advance mock time, letting promise chains finish before and after. */
    async wait(ms = 0) { await page.settle(); t.mock.timers.tick(ms); await page.settle(); },
    /** The server lists ``src`` now; the page's poll (every 2 s) picks it up. */
    async list(src) { state.src = src; await api.wait(2000); },
    /** Load the window file the stage was given: metadata arrives. */
    loadWindow(duration = 600) { v.__media.loaded(duration); },
    async scrub(to) {
      const seek = d.querySelector('#seek');
      // the rail shows a window around the moment; a drag outside it needs a pan, which jsdom
      // cannot lay out, so widen the slider to the whole source instead
      Object.assign(seek, { min: '0', max: '600', step: 'any' });
      seek.value = String(to);
      seek.dispatchEvent(new page.window.Event('change'));
      await api.wait();
    },
    async play() { d.querySelector('#playBtn').click(); await api.wait(); },
    /** Playback reaches source time ``t`` (video time is ``t`` minus the file's offset), a frame passes. */
    async playTo(t) {
      v.__media.tick(t - Number(v.dataset.offset || 0));
      await api.wait(16);
    },
  };
  await api.wait();
  return api;
}

test('a moment with no ready file asks for a window around it, then plays the window at the moment start', async (t) => {
  const p = await player(t);
  assert.deepEqual(p.prepares, [{ lo: 10, hi: 18 }]);
  assert.equal(p.v.dataset.url, undefined, 'nothing to play yet');
  assert.match(p.msg(), /Preparing this moment/);

  await p.list({ status: 'preparing', windows: [win(4, 40)] });
  assert.equal(p.v.dataset.url, win(4, 40).url);
  assert.equal(p.v.dataset.offset, '4');
  assert.equal(p.v.dataset.hi, '40');
  p.loadWindow(36);
  await p.wait();
  assert.equal(p.v.__media.currentTime, 6, 'source 10 s is 6 s into a file that starts at 4 s');
  assert.equal(p.msg(), '');
});

test('the whole proxy replaces the window at the next pause, keeping the position', async (t) => {
  const p = await player(t, { windows: [win(4, 40)] });
  p.loadWindow(36);
  await p.wait();
  await p.list({ status: 'ready' });
  const src = '/api/projects/p1/sources/s1?t=x';
  // a window of a ready source is not replaced while it plays; at the pause the whole file loads
  p.v.__media.paused = true;
  p.v.__media.fire('pause');
  await p.wait();
  assert.equal(p.v.dataset.url, src);
  assert.equal(p.v.dataset.hi, undefined);
  assert.equal(p.v.dataset.offset, '0');
});

test('scrubbing: inside the loaded window seeks it, into another ready window loads that one, outside any asks for a window', async (t) => {
  const p = await player(t, { windows: [win(0, 60), win(100, 160)] });
  p.loadWindow(60);
  await p.wait();
  const first = p.v.dataset.url;

  await p.scrub(30);
  assert.equal(p.v.dataset.url, first);
  assert.equal(p.v.__media.currentTime, 30);

  await p.scrub(120);
  assert.equal(p.v.dataset.url, win(100, 160).url);
  assert.equal(p.v.dataset.offset, '100');
  p.loadWindow(60);
  await p.wait();
  assert.equal(p.v.__media.currentTime, 20);

  p.prepares.length = 0;
  await p.scrub(300);
  assert.deepEqual(p.prepares, [], 'a scrub is debounced');
  await p.wait(300);
  assert.deepEqual(p.prepares, [{ lo: 300, hi: 300 }]);
  assert.match(p.msg(), /Preparing this part/);
  assert.ok(p.v.__media.paused);
});

test('at the end of a window proxy playback waits, asks for the next window, and plays on once it loads', async (t) => {
  const p = await player(t, { windows: [win(0, 60)] });
  p.loadWindow(60);
  await p.wait();
  await p.play();
  assert.equal(p.playLabel(), 'Pause');
  p.prepares.length = 0;

  await p.playTo(59.99);
  assert.ok(p.v.__media.paused, 'stops at the window end');
  assert.equal(p.playLabel(), 'Pause', 'still "playing": a click pauses the wait');
  assert.deepEqual(p.prepares, [{ lo: 59.99, hi: 59.99 }]);

  await p.list({ status: 'preparing', windows: [win(0, 60), win(55, 120)] });
  assert.equal(p.v.dataset.url, win(55, 120).url);
  p.v.__media.calls.length = 0;
  p.loadWindow(65);
  await p.wait();
  assert.equal(p.v.__media.currentTime, 59.99 - 55, 'resumes where it stopped');
  assert.deepEqual(p.v.__media.calls.filter((c) => c === 'play'), ['play']);
  assert.ok(!p.v.__media.paused);
});

test('pausing during that wait cancels the resume: the next window loads but does not play', async (t) => {
  const p = await player(t, { windows: [win(0, 60)] });
  p.loadWindow(60);
  await p.wait();
  await p.play();
  await p.playTo(59.99);
  await p.play(); // click: Pause
  assert.equal(p.playLabel(), 'Play');

  await p.list({ status: 'preparing', windows: [win(0, 60), win(55, 120)] });
  p.v.__media.calls.length = 0;
  p.loadWindow(65);
  await p.wait();
  assert.equal(p.v.dataset.url, win(55, 120).url);
  assert.ok(!p.v.__media.calls.includes('play'));
  assert.ok(p.v.__media.paused);
});

test('entering Cut mode during that wait cancels the resume: the next window does not start the hidden player', async (t) => {
  const segment = (id, start) => ({ id, source: '/videos/a.mp4', start, end: start + 8, label: id });
  const p = await player(t, { windows: [win(0, 60)] }, { project: { review: { segments: [segment('h001', 10), segment('h002', 100)] } } });
  p.loadWindow(60);
  await p.wait();
  await p.play();
  await p.playTo(59.99);
  assert.equal(p.playLabel(), 'Pause', 'waiting to play on');
  p.d.querySelector('#stageMode [data-mode=cut]').click();
  await p.wait();
  assert.equal(p.playLabel(), 'Play');

  await p.list({ status: 'preparing', windows: [win(0, 60), win(55, 120)] });
  p.v.__media.calls.length = 0;
  p.loadWindow(65);
  await p.wait();
  assert.ok(!p.v.__media.calls.includes('play'));
  assert.ok(p.v.__media.paused);
});

test('a ready next window takes over at the end without waiting', async (t) => {
  const p = await player(t, { windows: [win(0, 60), win(55, 120)] });
  p.loadWindow(60);
  await p.wait();
  await p.play();
  p.prepares.length = 0;
  await p.playTo(59.99);
  assert.equal(p.v.dataset.url, win(55, 120).url);
  assert.deepEqual(p.prepares, []);
  assert.ok(p.v.__media.calls.includes('play'));
});

test('the next window is asked for once, within 20 s of the end of a playing window', async (t) => {
  const p = await player(t, { windows: [win(0, 100)] });
  p.loadWindow(100);
  await p.wait();
  await p.play();
  p.prepares.length = 0;
  await p.playTo(70); // 30 s left
  assert.deepEqual(p.prepares, []);
  await p.playTo(85); // 15 s left
  assert.deepEqual(p.prepares, [{ lo: 100, hi: 100 }]);
  await p.playTo(90);
  assert.equal(p.prepares.length, 1);
});

test('a window the server never lists is asked for once more, then fails with Retry; Retry asks again', async (t) => {
  const p = await player(t, { windows: [win(0, 60)] });
  p.loadWindow(60);
  await p.wait();
  p.state.vanish = true; // the server accepts the request but never lists a window for it
  const asks = () => p.prepares.filter((r) => r.lo === 300).length;
  await p.scrub(300);
  await p.wait(300);
  // the ack's refresh lists no window for [300, 300]: asked once more
  assert.equal(asks(), 2);
  for (let i = 0; i < 3; i++) await p.wait(2000);
  assert.equal(asks(), 2, 'once per wait, so a failing encode is not retried forever');
  assert.equal(p.retryShown(), false);

  // the server reports the failure
  await p.list({ windows: [win(0, 60)], failed: [{ lo: 300, hi: 300, reason: 'encoder died' }] });
  assert.equal(p.retryShown(), true);
  assert.match(p.msg(), /could not be prepared.*encoder died/);

  // the server now accepts the encode again (a new request replaces the failure)
  p.state.vanish = false;
  p.state.src = { windows: [win(0, 60)] };
  p.prepares.length = 0;
  p.d.querySelector('#retryBtn').click();
  await p.wait();
  assert.deepEqual(p.prepares.slice(0, 1), [{ lo: 300, hi: 300 }]);
  assert.equal(p.retryShown(), false);
  assert.match(p.msg(), /Preparing this part/);
});

test('a listing fetched before the re-ask was taken still shows the old failure: it is not reported as the re-ask failing', async (t) => {
  let release;
  const gate = new Promise((r) => { release = r; });
  let asked = 0;
  const p = await player(t, { windows: [win(0, 60)] }, { prepare: () => (++asked === 2 ? gate.then(() => ({})) : {}) });
  p.state.vanish = true;
  p.state.src = { windows: [win(0, 60)], failed: [{ lo: 300, hi: 300, reason: 'old failure' }] };
  p.loadWindow(60);
  await p.wait();
  await p.scrub(300);
  await p.wait(300);
  assert.equal(asked, 2, 'asked, then asked once more; the server has not taken the second yet');

  await p.wait(2000); // a poll lists the old failure
  assert.equal(p.retryShown(), false);

  release();
  await p.wait();
  assert.equal(p.retryShown(), true);
  assert.match(p.msg(), /old failure/);
});

test('the stage and the Export tab\'s final video never sound together; leaving the Export tab pauses its video', async (t) => {
  const p = await player(t, { windows: [win(0, 60)] });
  p.loadWindow(60);
  await p.wait();
  const fin = p.d.querySelector('#finalVideo');
  const ev = (el, type) => el.dispatchEvent(new p.page.window.Event(type));

  await p.play();
  assert.ok(!p.v.__media.paused);
  fin.__media.paused = false; // the user starts the final video
  ev(fin, 'play');
  assert.ok(p.v.__media.paused, 'the stage pauses when the final video plays');

  await p.play(); // the stage again
  assert.ok(fin.__media.calls.includes('pause'), 'the final video pauses when the stage plays');

  fin.__media.paused = false;
  fin.__media.calls.length = 0;
  p.d.querySelector('#tabSequence').click(); // leave the Export panel for a tab
  p.d.querySelector('#exportBtn').click();
  p.d.querySelector('#tabSequence').click();
  await p.wait();
  assert.ok(fin.__media.calls.includes('pause'));
});
