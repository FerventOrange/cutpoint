// Play sequence on clips of one source file (#1020): the next clip of the file the visible player
// has open is sought to there, not opened again in the spare player (seconds per clip on a network
// share); a clip of another file still waits in the spare player and plays by a swap.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

const A = '/api/projects/p1/sources/s1?t=x';
const B = '/api/projects/p1/sources/s2?t=x';

/** h001 and h002 of a.mp4 (10-18, 100-108), then h003 of b.mp4 (50-58), in that order. */
function project() {
  const p = makeProject();
  p.sources.push({ id: 's2', path: '/videos/b.mp4', name: 'b.mp4', status: 'ready', url: B, duration: 300 });
  const c3 = p.highlights.candidates[2];
  Object.assign(c3, { source: '/videos/b.mp4', start: 50, end: 58 });
  p.review = { segments: p.highlights.candidates.map((c) => ({ id: c.id, source: c.source, start: c.start, end: c.end, label: c.label })) };
  return p;
}

async function playing(t) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ project: project() });
  const d = page.document;
  t.after(() => {
    const stop = d.querySelector('#seqPlayBtn');
    if (stop.textContent === 'Stop') stop.click(); // nothing runs on after the test
    page.close();
  });
  const [one, two] = ['#video', '#video2'].map((s) => d.querySelector(s));
  const s = {
    one, two, d, page,
    active: () => d.querySelector('.screen video.active'),
    row: () => d.querySelector('#seqList li.playing')?.dataset.id,
    /** The visible player reaches source time ``at``; one frame of playback runs. */
    async at(at) {
      s.active().__media.currentTime = at;
      t.mock.timers.tick(20);
      await page.settle();
    },
  };
  d.querySelector('#seqPlayBtn').click();
  await page.settle();
  for (const el of [one, two]) if (el.dataset.url && !(el.__media.readyState > 0)) el.__media.loaded(600);
  await page.settle();
  return s;
}

test('the next clip of the open file is sought to in the visible player, not opened in the spare', async (t) => {
  const s = await playing(t);
  assert.equal(s.active(), s.one);
  assert.equal(s.one.dataset.url, A);
  assert.equal(s.one.__media.currentTime, 10, 'the first clip starts at its start');
  assert.equal(s.row(), 'h001');
  assert.equal(s.two.dataset.url, undefined, 'the spare player does not open a.mp4 again');
  assert.equal(s.two.getAttribute('src'), null);

  await s.at(18);
  assert.equal(s.active(), s.one, 'no swap');
  assert.equal(s.one.dataset.url, A, 'the same file stays open');
  assert.equal(s.one.__media.currentTime, 100, 'sought to the second clip');
  assert.equal(s.one.paused, false, 'and plays on');
  assert.equal(s.row(), 'h002');
  assert.match(s.d.querySelector('#clock').textContent, /^Clip 2 of 3/);
  assert.ok(!s.one.__media.calls.includes('load'), 'never reloaded');
});

test('a stale `ended` of the first clip does not skip the clip sought to', async (t) => {
  const s = await playing(t);
  await s.at(18);
  assert.equal(s.row(), 'h002');
  s.one.__media.fire('ended'); // queued before the seek; the element is not at its end any more
  await Promise.resolve();
  assert.equal(s.row(), 'h002');
  assert.equal(s.one.__media.currentTime, 100);
});

test('a source that ends before its clip does moves on to the next clip', async (t) => {
  const s = await playing(t);
  assert.equal(s.row(), 'h001');
  s.one.__media.currentTime = 14; // below the clip end at 18
  s.one.__media.end(); // the file is shorter than its clip
  await s.page.settle();
  assert.equal(s.row(), 'h002');
  assert.equal(s.active().__media.currentTime, 100, 'sought to the start of clip 2');
});

test('a clip of another file still waits in the spare player and plays by a swap', async (t) => {
  const s = await playing(t);
  await s.at(18);
  assert.equal(s.two.dataset.url, B, 'b.mp4 is cued in the spare once its clip is next');
  s.two.__media.loaded(300);
  assert.equal(s.two.__media.currentTime, 50, 'at the clip start');
  assert.equal(s.two.paused, true, 'waiting');

  await s.at(108); // the end of the second clip of a.mp4 advances to the third
  assert.equal(s.row(), 'h003');
  assert.equal(s.active(), s.two, 'a hard cut to the preloaded player');
  assert.equal(s.two.paused, false);
  assert.equal(s.one.paused, true, 'the old player stops');
});

test('after the swap the sequence still ends with its last clip', async (t) => {
  const s = await playing(t);
  await s.at(18);
  await s.at(108);
  s.two.__media.loaded(300);
  await s.at(58);
  assert.equal(s.row(), undefined);
  assert.equal(s.d.querySelector('#seqPlayBtn').textContent, 'Play sequence');
});
