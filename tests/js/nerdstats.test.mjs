// Stats for Nerds: the moment's laughter is shown as two lines, said in the transcript
// (signals.laughter) and heard in the audio (signals.events.laughter), each with a plain label.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, makeProject } from './harness.mjs';

const KEY = 'cutpoint.nerdStats';

async function open(t, signals) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const project = makeProject();
  project.highlights.candidates[0] = { ...project.highlights.candidates[0], signals };
  const page = await boot({ project, storage: { [KEY]: '1' } });
  t.after(() => page.close());
  page.document.querySelector('#candList li.cand[data-id="h001"] .cand-main').click();
  await page.settle();
  return page.document.querySelector('#signals').textContent.split('\n');
}

test('transcript laughter and heard laughter are separate lines with plain labels', async (t) => {
  const lines = await open(t, { loudness: 0.5, laughter: 0.63, events: { boost: 3, laughter: 0.21 } });
  assert.ok(lines.includes('Laughter in the transcript: 0.63'), lines.join('|'));
  assert.ok(lines.includes('Laughter heard in the audio: 0.21'), lines.join('|'));
  const signals = lines.find((l) => l.startsWith('Signals:'));
  assert.equal(signals, 'Signals: loudness 0.50', 'laughter is not repeated in the signal list');
});

test('only the laughter that exists is shown', async (t) => {
  const heard = await open(t, { laughter: 0, events: { laughter: 0.3 } });
  assert.ok(heard.includes('Laughter heard in the audio: 0.30'));
  assert.ok(!heard.some((l) => l.startsWith('Laughter in the transcript')));
});

test('a moment with no laughter shows neither line', async (t) => {
  const lines = await open(t, { loudness: 0.4, events: { boost: 0 } });
  assert.ok(!lines.some((l) => /^Laughter/.test(l)), lines.join('|'));
});
