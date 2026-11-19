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

test('the model picker: default and placeholder, option order, saved model with no list, a typed id', () => {
  const listed = [{ id: 'claude-a', name: 'Claude A' }, { id: 'claude-b', name: 'claude-b' }];
  const ids = (r) => r.options.map((o) => (o.value === u.AI_OTHER ? 'OTHER' : o.value));
  let r = u.aiModelChoices('claude-b', listed, { placeholder: 'claude-sonnet-5-5' });
  assert.deepEqual(r.options.map((o) => o.text),
    ['Default (claude-sonnet-5-5)', 'Claude A (claude-a)', 'claude-b', 'Other model…']);
  assert.equal(r.value, 'claude-b');
  assert.equal(u.aiModelChoices('', listed).value, '', 'an empty saved model is the default');
  assert.deepEqual(u.aiModelChoices('', [], { required: true }).options.map((o) => o.text),
    ['Choose a model', 'Other model…']);

  r = u.aiModelChoices('claude-old', listed, { fallback: ['claude-x'] });
  assert.deepEqual(ids(r), ['', 'claude-old', 'claude-a', 'claude-b', 'OTHER'], 'the list wins over the fallback');

  const presets = ['claude-sonnet-5-5', 'claude-opus-5-5'];
  r = u.aiModelChoices('my-model:7b', null, { fallback: presets });
  assert.equal(r.value, 'my-model:7b');
  assert.deepEqual(r.options.map((o) => o.text),
    ['Default', 'my-model:7b', 'claude-sonnet-5-5', 'claude-opus-5-5', 'Other model…']);
  r = u.aiModelChoices('claude-opus-5-5', null, { fallback: presets });
  assert.equal(r.value, 'claude-opus-5-5');
  assert.deepEqual(ids(r), ['', ...presets, 'OTHER'], 'a preset model adds no extra option');

  for (const typed of ['new-model', '']) {
    r = u.aiModelChoices(typed, listed, { typing: true });
    assert.equal(r.value, u.AI_OTHER);
    assert.deepEqual(ids(r), ['', 'claude-a', 'claude-b', 'OTHER'], 'no extra option while typing');
  }
  assert.equal(u.aiModelChoices('claude-a', listed, { typing: true }).value, 'claude-a', 'an id the list has is chosen in the list');
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

test('namesFile: a project file counts only whole and as written; a folder also as "<name> folder"', () => {
  const cases = [
    ['The transcript is in transcript.txt.', 'transcript.txt', true],
    ['See `final.mp4`', 'final.mp4', true],
    ['It is at /home/x/projects/demo/final.mp4', 'final.mp4', true],
    ['It is at C:\\Projects\\demo\\final.srt, next to it', 'final.srt', true],
    ['an old myfinal.mp4.bak', 'final.mp4', false],
    ['an old final.mp4.bak', 'final.mp4', false],
    ['clip-final.mp4 is the copy', 'final.mp4', false],
    ['Final.mp4', 'final.mp4', false], // case-sensitive, as the real file
    ['the transcript', 'transcript.txt', false],
    ['open clips/001_a.mp4', 'clips/', true],
    ['the clips folder of the export', 'clips/', true],
    ['the `clips` folder', 'clips/', true],
    ['I added 3 clips to the sequence', 'clips/', false],
    ['the clips_vertical folder', 'clips/', false],
    ['the clips_vertical folder', 'clips_vertical/', true],
    ['in D:\\Videos\\demo\\clips\\002_b.mp4', 'clips/', true],
  ];
  for (const [text, name, want] of cases) assert.equal(u.namesFile(text, name), want, `${name} in ${text}`);
});

test('namedFolders: the folders of the files a reply names, project first, each once', () => {
  const files = [
    { name: 'transcript.txt', target: 'project' },
    { name: 'final.mp4', target: 'project' },
    { name: 'clips/', target: 'clips' },
  ];
  assert.deepEqual(u.namedFolders('Your clips folder and final.mp4 and transcript.txt', files), ['project', 'clips']);
  assert.deepEqual(u.namedFolders('Only the clips/ folder', files), ['clips']);
  assert.deepEqual(u.namedFolders('Nothing named here', files), []);
  assert.deepEqual(u.namedFolders('final.mp4', undefined), []);
});

test('spanText: seconds under a minute, minutes under an hour, then hours and minutes (#459)', () => {
  for (const [s, text] of [[45, '45 s'], [59.5, '1 min'], [3570, '1 h'], [8040, '2 h 14 min']]) assert.equal(u.spanText(s), text, String(s));
});

test('rowInPlace: focus goes to the row now where one left the list, the last past the end, none in an empty list', () => {
  // Add under "Not in sequence" took the card out of the list; preserveFocus (tests/js/workspace.test.mjs) uses this
  for (const [at, n, row] of [[2, 5, 2], [4, 4, 3], [0, 0, -1]]) assert.equal(u.rowInPlace(at, n), row, `${at} of ${n}`);
});

test('video order: A-Z by file, equal starts keep their order, case variants and same-named files sort the same whatever the input order (#469)', () => {
  const m = (id, source, start) => ({ id, source, start });
  const ids = (list) => [...list].sort(u.positionCompare).map((x) => x.id);
  assert.deepEqual(ids([m('c10b', '/v/clip10.mp4', 5), m('c2late', '/v/clip2.mp4', 90), m('a', '/v/Alpha.mp4', 1), m('c10a', '/v/clip10.mp4', 2), m('c2early', '/v/clip2.mp4', 3)]),
    ['a', 'c2early', 'c2late', 'c10a', 'c10b']);
  assert.deepEqual(ids([m('first', '/v/a.mp4', 4), m('second', '/v/a.mp4', 4), m('third', '/v/a.mp4', 4)]), ['first', 'second', 'third'], 'stable');
  const mixed = [m('x1', '/two/clip.mp4', 1), m('y1', '/one/clip.mp4', 50), m('x2', '/two/clip.mp4', 9), m('y0', '/one/clip.mp4', 2)];
  assert.deepEqual(ids(mixed), ['y0', 'y1', 'x1', 'x2']);
  assert.deepEqual(ids([...mixed].reverse()), ['y0', 'y1', 'x1', 'x2']);
  const cased = [m('up', '/v/Clip.mp4', 1), m('low', '/v/clip.mp4', 1), m('z', '/v/zed.mp4', 0)];
  const got = ids(cased);
  assert.deepEqual(ids([...cased].reverse()), got);
  assert.equal(got.at(-1), 'z');
});

test('momentMatches: every quick filter and the search text must all hold; filtersOn names a set quick filter', () => {
  const ms = [
    { id: 'a', cli_id: 'h001', label: 'Jump', transcript: 'nice one', source: '/v/a.mp4', picked: true, ready: true, must_include: true },
    { id: 'b', label: 'Crash', transcript: null, source: '/v/b.mp4', picked: false, ready: true },
    { id: 'c', label: 'jump two', source: '/v/b.mp4', picked: false, ready: false, must_include: true },
  ];
  const cases = [
    ['', {}, ['a', 'b', 'c']],
    ['  JUMP ', {}, ['a', 'c']], // trimmed, case ignored
    ['NICE', {}, ['a']], // transcript
    ['b.mp4', {}, ['b', 'c']], // file name
    ['H001', {}, ['a']], // the CLI id on the card (#487)
    ['', { seq: 'in' }, ['a']],
    ['', { seq: 'out' }, ['b', 'c']],
    ['', { ready: true }, ['a', 'b']],
    ['', { must: true }, ['a', 'b', 'c']], // Must include was removed: not a filter
    ['', { source: '/v/b.mp4' }, ['b', 'c']],
    ['jump', { seq: 'out' }, ['c']],
    ['jump', { ready: true, source: '/v/b.mp4' }, []],
  ];
  for (const [q, f, want] of cases) {
    const all = { ...u.NO_FILTERS, ...f };
    assert.deepEqual(ms.filter((c) => u.momentMatches(c, q, all)).map((c) => c.id), want, `${JSON.stringify(q)} ${JSON.stringify(f)}`);
    assert.equal(u.filtersOn(all), ['seq', 'ready', 'source'].some((k) => k in f), JSON.stringify(f));
  }
});

test('colorNote groups the reasons clips were left as is, counts each when there are several, and counts Strength 0 as not graded (#461)', () => {
  const graded = (method, skip, strength = 1) => ({ file: '001_a.mp4', color: { method, strength, skip } });
  const auto = (...skips) => skips.map((s) => graded('auto', s));
  const cases = [
    [Array.from({ length: 4 }, () => graded('underwater', 'veto_clipped')), ' Color: no clips graded (too much highlight clipping).'],
    [auto(null, 'good', null, null), ' Color: 3 of 4 clips graded; 1 left as is (already looked good).'],
    [['veto_m3_drop', 'veto_clipped', 'veto_m3_drop', null].map((s) => graded('underwater', s)),
      ' Color: 1 of 4 clips graded; 3 left as is (would wash out the color: 2, too much highlight clipping: 1).'],
    [auto('360', 'log', 'no_frames', 'veto_cast', 'veto_m3', 'veto_new', 'not_underwater', 'veto_faded', 'good'),
      ' Color: no clips graded (360 footage: 1, log footage: 1, frames could not be read: 1, would add a color cast: 1,'
      + ' would oversaturate: 1, veto_new: 1, did not look underwater: 1, would wash out the color: 1, already looked good: 1).'],
    [auto('veto_faded', 'veto_m3_drop', null), ' Color: 1 of 3 clips graded; 2 left as is (would wash out the color).'], // PR #495: one reason
    [auto(null, null), ' Color: all 2 clips graded.'],
    [auto(null), ' Color: 1 clip graded.'],
    [[{ file: 'a.mp4', color: null }, { file: 'b.mp4' }], ''], // Color Off, or an export from before --color
    [[{ file: 'a.mp4', lut: { file: 'x.cube' } }], ''],
    [[], ''],
    [[graded('auto', null, 0), graded('auto', null, 0)], ''], // Strength 0% grades nothing
    [[graded('auto', null, 0.5), graded('auto', null, 0)], ' Color: 1 of 2 clips graded; 1 left as is (strength 0%).'],
    [[graded('auto', 'good', 0.5), graded('auto', null, 0), graded('auto', null, 0)], ' Color: no clips graded (already looked good: 1, strength 0%: 2).'],
    [[{ file: 'a.mp4', color: { method: 'auto' } }], ' Color: 1 clip graded.'], // no strength: full strength
  ];
  for (const [clips, want] of cases) assert.equal(u.colorNote(clips), want, JSON.stringify(clips));
});
