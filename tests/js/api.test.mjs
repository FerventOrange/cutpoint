// The JSON client: token handling, error messages and the expired-session switch.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, reply } from './harness.mjs';

const client = async (t, opts = {}) => {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({ entry: 'api.js', project: null, ...opts });
  t.after(() => page.close());
  return page;
};

test('the session token comes from the startup URL fragment, is stripped from it, and goes with every call', async (t) => {
  const page = await client(t, { hash: '#t=abc-123', token: null, routes: { 'GET /api/x': {} } });
  assert.equal(page.window.location.hash, '', 'the token is not left in the address bar');
  assert.equal(page.window.sessionStorage.getItem('cutpoint-token'), 'abc-123');
  await page.mod.api('/api/x');
  assert.equal(page.requests.at(-1).headers['X-Cutpoint-Token'], 'abc-123');
});

test('a page without a token fails at once, without a request', async (t) => {
  const page = await client(t, { token: null });
  await assert.rejects(page.mod.api('/api/x'), (e) => e.status === 401 && /no session token/.test(e.message));
  assert.equal(page.requests.length, 0);
});

test('error responses become readable messages (string, validation list, object, plain text)', async (t) => {
  const cases = [
    [reply(400, { detail: 'Bad thing' }), 'Bad thing'],
    [reply(422, { detail: [{ msg: 'field required' }, { msg: 'too long' }] }), 'field required; too long'],
    [reply(409, { detail: { a: 1 } }), '{"a":1}'],
    [reply(500, { error: 'boom' }), 'boom'],
  ];
  const routes = Object.fromEntries(cases.map(([r], i) => [`GET /api/e${i}`, r]));
  const page = await client(t, { routes });
  for (const [i, [r, message]] of cases.entries()) {
    await assert.rejects(page.mod.api(`/api/e${i}`), (e) => e.status === r.status && e.message === message);
  }
});

test('an unreachable server is reported as such, not as a JSON error', async (t) => {
  const page = await client(t);
  page.window.fetch = undefined;
  globalThis.fetch = async () => { throw new TypeError('network'); };
  await assert.rejects(page.mod.api('/api/x'), (e) => e.status === 0 && /Can't reach the cutpoint server/.test(e.message));
});

test('a rejected token ends the session once: the callback runs once and later calls make no request', async (t) => {
  const page = await client(t, { routes: { 'GET /api/x': reply(403, { detail: 'missing or wrong session token' }) } });
  const seen = [];
  page.mod.whenExpired((e) => seen.push(e.message));
  await assert.rejects(page.mod.api('/api/x'), /server restarted or this link expired/);
  const n = page.requests.length;
  await assert.rejects(page.mod.api('/api/x'), /server restarted or this link expired/);
  assert.equal(page.requests.length, n);
  assert.equal(seen.length, 1);
});

test('a 403 for another reason (host or origin) does not expire the session', async (t) => {
  const page = await client(t, { routes: { 'GET /api/x': reply(403, { detail: 'bad host' }), 'GET /api/y': {} } });
  const seen = [];
  page.mod.whenExpired((e) => seen.push(e));
  await assert.rejects(page.mod.api('/api/x'), /rejected this session token/);
  await page.mod.api('/api/y');
  assert.equal(seen.length, 0);
});
