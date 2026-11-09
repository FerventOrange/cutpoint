// Pure helpers behind the moments list and the model picker.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { importPure } from './harness.mjs';

// top-level await, not a before() hook: older Node 22 releases start tests before an async hook finishes
const u = await importPure('util.js');

test('parseTime reads seconds and m:ss, and refuses anything else', () => {
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

  r = u.aiModelChoices('gpt-4', [{ id: 'gpt-4', name: 'gpt-4o preview' }]);
  assert.equal(r.options[1].text, 'gpt-4o preview (gpt-4)', 'a name that merely begins with the id text still shows the id');
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

test('momentMatches: the Long bits filter keeps only moments the server marked long_bit', () => {
  const bit = { label: 'x', long_bit: true };
  assert.equal(u.momentMatches(bit, '', { ...u.NO_FILTERS, bits: true }), true);
  assert.equal(u.momentMatches({ label: 'x' }, '', { ...u.NO_FILTERS, bits: true }), false);
  assert.equal(u.momentMatches({ label: 'x' }, '', u.NO_FILTERS), true);
  assert.equal(u.filtersOn({ ...u.NO_FILTERS, bits: true }), true);
  assert.equal(u.filtersOn(u.NO_FILTERS), false);
});

test('kindOf: a moment made from a talk stretch is Dialogue whatever words its title holds', () => {
  const made = { label: 'fast-run-again', display_label: 'Fast Run Again', dialogue_title: 'Fast Run Again', signals: { events: { nominated: ['dialogue-funny'] } } };
  assert.equal(u.kindOf(made), 'dialogue');
  assert.equal(u.kindOf({ label: 'dialogue-funny', signals: { events: { nominated: ['dialogue-story'] } } }), 'dialogue', 'titled or not');
  assert.equal(u.kindOf({ label: 'fast-run-again', display_label: 'Fast Run Again' }), 'action');
});

test('kindOf: a moment a stretch only widened keeps its own type, from the label it had', () => {
  const widened = { label: 'fast-run-again', display_label: 'Fast Run Again', dialogue_title: 'Fast Run Again', widened_from: { label: 'laughter' }, signals: { events: {} } };
  assert.equal(u.kindOf(widened), 'funny');
  assert.equal(u.kindOf({ ...widened, widened_from: { label: 'ski-moment' } }), 'action', 'a generic label names nothing: the transcript decides');
});

test('kindOf: a crash comes first (its crash record, or kind crash or wipeout); KINDS ends with Crashes', () => {
  assert.deepEqual(Object.entries(u.KINDS), [['action', 'Action'], ['scenic', 'Scenic'], ['funny', 'Funny'], ['dialogue', 'Dialogue'], ['crash', 'Crashes']]);
  assert.equal(u.kindOf({ label: 'laugh', crash: { confidence: 0.4 }, crash_forced: false }), 'funny', 'a moment a weaker crash marked keeps its kind');
  assert.equal(u.kindOf({ label: 'crash-reaction', kind: 'crash', crash: { confidence: 0.4 }, crash_forced: false }), 'crash', 'a weak crash moment');
  assert.equal(u.kindOf({ label: 'crash-reaction', must_include: true, crash: { confidence: 0.9 } }), 'crash', 'a forced reaction, older record without kind');
  assert.equal(u.kindOf({ label: 'wipeout', kind: 'wipeout' }), 'crash');
  assert.equal(u.kindOf({ label: 'x', signals: { events: { nominated: ['dialogue-funny'] } }, crash: { confidence: 0.9 }, crash_forced: true }), 'crash');
  assert.equal(u.kindOf({ label: 'ski-moment', signals: { crash: 0.9 } }), 'action', 'a crash cue alone stays action, as in AI Chat');
});

test('crashOf and crashText: the confidence with two decimals, forced from crash_forced or must_include', () => {
  assert.equal(u.crashOf({}), null);
  assert.deepEqual(u.crashOf({ crash: { confidence: 0.5 }, crash_confidence: 0.912, crash_forced: true }), { confidence: 0.912, forced: true });
  assert.deepEqual(u.crashOf({ crash: { confidence: 0.72 }, must_include: true }), { confidence: 0.72, forced: true }, 'an older action-cam record');
  assert.deepEqual(u.crashOf({ crash: {}, crash_forced: false }), { confidence: null, forced: false });
  assert.equal(u.crashText({ confidence: 0.912 }), 'CRASH 0.91');
  assert.equal(u.crashText({ confidence: null }), 'CRASH');
});
