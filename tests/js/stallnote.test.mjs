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

/** Boot a project whose only clip cannot play yet; ``play``: Play sequence waits for it. */
async function opening(t, play) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const project = makeProject({ network: true });
  project.sources[0] = { ...project.sources[0], status: 'needs-proxy', windows: [] };
  const c = project.highlights.candidates[0];
  project.review = { segments: [{ id: c.id, source: c.source, start: c.start, end: c.end, label: c.label }] };
  const page = await boot({ project, routes: { 'POST /api/projects/p1/sources/s1/prepare': {} } });
  t.after(() => page.close());
  const d = page.document;
  if (play) d.querySelector('#seqPlayBtn').click();
  await page.settle();
  // the open is slow and the element stays paused, so it never fires `waiting`
  const m = d.querySelector('.screen video.active').__media;
  Object.assign(m, { paused: true, readyState: 0 });
  m.fire('loadstart');
  t.mock.timers.tick(5100); // read before the project refreshes: that reloads the stage
  const shown = !d.querySelector('#stallNote').hidden;
  if (play) d.querySelector('#seqPlayBtn').click(); // Stop: nothing runs on after the test
  await page.settle();
  return shown;
}

test('a paused element opening a sequence clip that waits shows the note', async (t) => {
  assert.ok(await opening(t, true));
});

test('a paused element outside Play sequence shows no note', async (t) => {
  assert.ok(!(await opening(t, false)));
});

/** Play sequence on a clip that waits; Stop at ``stopAt`` ms, then run on to ``end`` ms. */
async function stopping(t, stopAt, end) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const project = makeProject({ network: true });
  project.sources[0] = { ...project.sources[0], status: 'needs-proxy', windows: [] };
  const c = project.highlights.candidates[0];
  const seg = { id: c.id, source: c.source, start: c.start, end: c.end, label: c.label };
  project.review = { segments: [seg] };
  const page = await boot({ project, routes: { 'POST /api/projects/p1/sources/s1/prepare': {} } });
  t.after(() => page.close());
  const d = page.document;
  const note = d.querySelector('#stallNote');
  d.querySelector('#seqPlayBtn').click();
  await page.settle();
  const m = d.querySelector('.screen video.active').__media;
  Object.assign(m, { paused: true, readyState: 0 });
  m.fire('loadstart');
  t.mock.timers.tick(stopAt);
  const before = { hidden: note.hidden, text: note.textContent };
  d.querySelector('#seqPlayBtn').click(); // Stop
  const after = { hidden: note.hidden, text: note.textContent };
  await page.settle();
  for (let at = stopAt; at < end; at += 1000) { // the poll reloads the stage: settle between
    t.mock.timers.tick(1000);
    await page.settle();
  }
  return { before, after, late: { hidden: note.hidden, text: note.textContent } };
}

test('Stop removes the note a waiting sequence clip showed', async (t) => {
  const r = await stopping(t, 5100, 5100);
  assert.ok(!r.before.hidden, 'the note showed after 5 s');
  assert.ok(r.after.hidden, 'Stop hides it');
  assert.equal(r.after.text, '', 'and empties it');
});

test('Stop before 5 s means the note never shows', async (t) => {
  const r = await stopping(t, 2000, 8000);
  assert.ok(r.before.hidden, 'not yet at 2 s');
  assert.ok(r.after.hidden);
  assert.ok(r.late.hidden, 'the cancelled wait does not show a note later');
  assert.equal(r.late.text, '');
});
