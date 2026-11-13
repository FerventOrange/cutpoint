// A player that waits for data over 5 s names the likely reason (#1014): the project folder on a
// network drive, a fragmented MP4 played as is; the note goes when playback starts.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

/** Boot with mock timers on a project whose source plays as is; ``over`` changes the listing. */
async function stage(t, over = {}, src = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const project = makeProject(over);
  project.sources[0] = { ...project.sources[0], ...src };
  const page = await boot({ project });
  t.after(() => page.close());
  const d = page.document;
  const v = d.querySelector('#video');
  const note = d.querySelector('#stallNote');
  return {
    v, note,
    async wait(ms = 0) { await page.settle(); t.mock.timers.tick(ms); await page.settle(); },
    /** The browser asked to play and waits for data. */
    waiting() { Object.assign(v.__media, { paused: false, readyState: 1 }); v.__media.fire('waiting'); },
  };
}

test('a wait over 5 s shows the likely reasons, and playing removes them', async (t) => {
  const s = await stage(t, { network: true }, { fragmented: true, proxy: false });
  assert.equal(s.v.dataset.url, '/api/projects/p1/sources/s1?t=x', 'the stage plays the source as is');
  s.waiting();
  await s.wait(4900);
  assert.ok(s.note.hidden, 'not before 5 s');
  await s.wait(200);
  assert.ok(!s.note.hidden);
  assert.match(s.note.textContent, /project folder is on a network or external drive/);
  assert.match(s.note.textContent, /fragmented MP4/);
  s.v.__media.readyState = 4;
  s.v.__media.fire('playing');
  assert.ok(s.note.hidden);
  assert.equal(s.note.textContent, '');
});

test('without a known reason the note says so in general terms', async (t) => {
  const s = await stage(t, { network: false }, { fragmented: false });
  s.waiting();
  await s.wait(5100);
  assert.ok(!s.note.hidden);
  assert.match(s.note.textContent, /^Still loading\. The drive or the video may be slow to read\.$/);
});

test('a source played from its proxy is not called fragmented', async (t) => {
  const s = await stage(t, { network: true }, { fragmented: true, proxy: true });
  s.waiting();
  await s.wait(5100);
  assert.match(s.note.textContent, /network or external drive/);
  assert.doesNotMatch(s.note.textContent, /fragmented/);
});

test('a wait that ends in time, or a pause, shows no note', async (t) => {
  const s = await stage(t, { network: true });
  s.waiting();
  await s.wait(3000);
  s.v.__media.fire('playing');
  await s.wait(3000);
  assert.ok(s.note.hidden, 'played after 3 s');

  s.waiting();
  await s.wait(1000);
  s.v.__media.paused = true;
  s.v.__media.fire('pause');
  await s.wait(5000);
  assert.ok(s.note.hidden, 'paused while waiting');
});

test('another file loading removes the note', async (t) => {
  const s = await stage(t, { network: true });
  s.waiting();
  await s.wait(5100);
  assert.ok(!s.note.hidden);
  s.v.__media.fire('loadstart');
  assert.ok(s.note.hidden, 'loadstart');

  s.waiting();
  await s.wait(5100);
  assert.ok(!s.note.hidden);
  s.v.__media.paused = true;
  s.v.load(); // emptied, as closing the project does: no pause event
  s.v.__media.fire('loadstart');
  assert.ok(s.note.hidden, 'load()');
});

test('a wait started before another file loads shows no note for it', async (t) => {
  const s = await stage(t, { network: true });
  s.waiting();
  await s.wait(2000);
  s.v.__media.fire('loadstart');
  await s.wait(4000);
  assert.ok(s.note.hidden);
});
