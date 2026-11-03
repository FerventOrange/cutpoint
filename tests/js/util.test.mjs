// Pure helpers behind the moments list, the trim fields and the model picker.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { importPure } from './harness.mjs';

// top-level await, not a before() hook: older Node 22 releases start tests before an async hook finishes
const u = await importPure('util.js');

test('timecodes: typed times parse in every accepted form and reject the rest; fmt is their inverse', () => {
  assert.equal(u.parseTime('1:23.5'), 83.5);
  assert.equal(u.parseTime('1:02:03'), 3723);
  assert.ok(Number.isNaN(u.parseTime('1:2:3:4')));
  assert.equal(u.parseTime(u.fmt(83.5)), 83.5);
});

test('video order sorts files naturally (clip2 before clip10) and moments by start inside a file', () => {
  const m = (source, start) => ({ source, start });
  // same-named files in two folders sort by full path, so each source's moments stay together
  const sorted = [m('/v/clip10.mp4', 1), m('/v/clip2.mp4', 5), m('/other/clip2.mp4', 9), m('/v/clip2.mp4', 2), m('/other/clip2.mp4', 1)].sort(u.positionCompare);
  assert.deepEqual(sorted.map((x) => `${x.source}@${x.start}`),
    ['/other/clip2.mp4@1', '/other/clip2.mp4@9', '/v/clip2.mp4@2', '/v/clip2.mp4@5', '/v/clip10.mp4@1']);
  const one = [m('/v/a.mp4', 30), m('/v/a.mp4', 4)].sort(u.positionCompare);
  assert.deepEqual(one.map((x) => x.start), [4, 30]);
});

test('momentMatches: the search text covers label, id, transcript and file name; quick filters combine with it', () => {
  const m = { label: 'Big laugh', cli_id: 'h001', transcript: 'we fell over', source: '/videos/Holiday.mp4', picked: false, ready: true };
  const none = u.NO_FILTERS;
  const cases = [
    ['LAUGH', none, true], // label, case-insensitive
    ['holiday', none, true], // file name
    ['videos', none, false], // the folder path is not searched
    ['laugh', { ...none, seq: 'in' }, false], // search and quick filter must both hold
  ];
  for (const [q, f, expected] of cases) assert.equal(u.momentMatches(m, q, f), expected, `${JSON.stringify(q)} ${JSON.stringify(f)}`);
});

test('the model picker keeps a model the list lacks (so opening never changes it) and offers the fallback when no list came', () => {
  const listed = [{ id: 'a', name: 'Alpha' }, { id: 'b' }];
  let r = u.aiModelChoices('a', listed);
  assert.deepEqual(r.options.map((o) => o.text), ['Default', 'Alpha (a)', 'b', 'Other model…']);
  assert.equal(r.value, 'a');

  r = u.aiModelChoices('old-model', listed);
  assert.equal(r.value, 'old-model');
  assert.ok(r.options.some((o) => o.value === 'old-model' && /not in the provider's list/.test(o.text)));

  r = u.aiModelChoices('', [], { fallback: ['x', 'y'], required: true });
  assert.deepEqual(r.options.slice(0, 3).map((o) => o.value), ['', 'x', 'y']);
  assert.equal(r.options[0].text, 'Choose a model');

  r = u.aiModelChoices('typed-id', listed, { typing: true });
  assert.equal(r.value, u.AI_OTHER, 'an id being typed stays under Other model…');

  r = u.aiModelChoices('m', [{ id: 'm', name: 'm (may not accept images)' }]);
  assert.equal(r.options[1].text, 'm (may not accept images)', 'a name that starts with the id is not repeated');
});

test('colorNote says how many clips were graded and why the others were not; a LUT or Strength 0 says nothing', () => {
  const clip = (color) => ({ color });
  const cases = [
    [[clip({ method: 'auto' }), clip({ method: 'auto' })], ' Color: all 2 clips graded.'],
    [[clip({ method: 'auto' }), clip({ method: 'auto', skip: 'good' })], ' Color: 1 of 2 clips graded; 1 left as is (already looked good).'],
    [[clip({ method: 'auto', skip: 'log' }), clip({ method: 'auto', skip: 'veto_cast' })], ' Color: no clips graded (log footage: 1, would add a color cast: 1).'],
    [[clip({ method: 'auto', strength: 0 })], ''],
    [[clip({ lut: 'x.cube' }), {}], ''],
  ];
  for (const [clips, expected] of cases) assert.equal(u.colorNote(clips), expected);
});
