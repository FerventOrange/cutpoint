// The Models tab starts the subscription proxy on demand (#821): a saved login starts it when the tab
// loads, a login click says "Starting" while the server starts it, and a failure shows on the row.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, reply } from './harness.mjs';

const settings = () => ({
  assistant: { provider: 'ollama', model: '', url: '', custom: false, local: true, providers: [], keys: {}, key_hosts: {} },
  vision_assistant: { provider: 'ollama', model: '', url: '', custom: false, providers: [], keys: {}, key_hosts: {} },
  vision: false, vision_model: 'qwen3.5:9b', vision_consent: null,
});

const CODEX = { provider: 'codex', title: 'Codex', logged_in: false, email: null, org: null, error: null };

/**
 * Boot and open the Models tab. ``logins`` is a function returning the body of GET /api/assistant/logins
 * (so a test can change it after the proxy started); ``routes`` adds or replaces routes.
 */
async function modelsTab(t, { logins, routes = {} }) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const page = await boot({
    routes: {
      'GET /api/settings': settings,
      'GET /api/assistant/logins': logins,
      'GET /api/assistant/models': { state: 'ok', models: [] },
      ...routes,
    },
  });
  t.after(() => page.close());
  const d = page.document;
  d.querySelector('#tabModels').click();
  await page.settle();
  const posts = (path) => page.requests.filter((r) => r.method === 'POST' && r.path === path);
  return { page, d, posts, row: () => d.querySelector('#modelsSubsList').textContent };
}

test('a saved login with no proxy running starts the proxy once and lists the logins again', async (t) => {
  let running = false;
  const m = await modelsTab(t, {
    logins: () => ({ logins: [{ ...CODEX, logged_in: running }], proxy_running: running, saved_login: !running }),
    routes: { 'POST /api/assistant/proxy': () => { running = true; return { running: true }; } },
  });
  assert.equal(m.posts('/api/assistant/proxy').length, 1);
  assert.match(m.row(), /Connected/);
  assert.equal(m.page.requests.filter((r) => r.path === '/api/assistant/logins').length, 2, 'listed again after the start');
  m.page.assertNoUnhandled();
});

for (const body of [
  { proxy_running: false, saved_login: false },
  { proxy_running: true, saved_login: false },
  { proxy_running: true, saved_login: true },
]) {
  test(`no proxy start for ${JSON.stringify(body)}`, async (t) => {
    const m = await modelsTab(t, { logins: () => ({ logins: [CODEX], ...body }) });
    assert.equal(m.posts('/api/assistant/proxy').length, 0);
    m.page.assertNoUnhandled();
  });
}

test('a proxy that does not start shows why on the row and is not asked again in a loop', async (t) => {
  const why = 'the subscription proxy did not start: port 18317 is already in use';
  const m = await modelsTab(t, {
    logins: () => ({ logins: [CODEX], proxy_running: false, saved_login: true }),
    routes: { 'POST /api/assistant/proxy': () => reply(503, { detail: why }) },
  });
  assert.equal(m.posts('/api/assistant/proxy').length, 1);
  assert.ok(m.row().includes(why), m.row());
  assert.match(m.row(), /Not connected/);
});

test('clicking Log in with no proxy running says the proxy is starting until the server answers', async (t) => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const m = await modelsTab(t, {
    logins: () => ({ logins: [CODEX], proxy_running: false, saved_login: false }),
    routes: { 'POST /api/assistant/login': async () => { await gate; return { url: 'https://auth.example/login' }; } },
  });
  m.page.window.confirm = () => true;
  m.page.window.open = () => {};
  m.d.querySelector('#modelsSubsList button').click();
  await m.page.settle();
  assert.match(m.row(), /Starting the proxy/);
  const button = m.d.querySelector('#modelsSubsList button');
  assert.equal(button.textContent, 'Starting…');
  assert.equal(button.disabled, true);

  release();
  await m.page.settle();
  assert.match(m.row(), /Waiting for browser/);
  assert.ok(!/Starting/.test(m.row()));
});

test('a login that fails because the proxy did not start shows the reason on the row', async (t) => {
  const why = 'the subscription proxy did not start: no binary';
  const m = await modelsTab(t, {
    logins: () => ({ logins: [CODEX], proxy_running: false, saved_login: false }),
    routes: { 'POST /api/assistant/login': () => reply(503, { detail: why }) },
  });
  m.page.window.confirm = () => true;
  m.d.querySelector('#modelsSubsList button').click();
  await m.page.settle();
  assert.ok(m.row().includes(why), m.row());
  assert.ok(!/Starting/.test(m.row()));
  assert.equal(m.d.querySelector('#modelsSubsList button').disabled, false, 'can try again');
});

test('once the proxy runs, reopening clears the "did not start" errors but keeps other login errors', async (t) => {
  const state = { running: false };
  const CLAUDE = { ...CODEX, provider: 'claude-code', title: 'Claude Code' };
  const why = 'the subscription proxy did not start: no binary';
  const m = await modelsTab(t, {
    logins: () => ({ logins: [CODEX, CLAUDE], proxy_running: state.running, saved_login: false }),
    routes: {
      'POST /api/assistant/login': (req) => (req.body.provider === 'codex'
        ? reply(503, { detail: why })
        : reply(500, { detail: 'login denied' })),
    },
  });
  m.page.window.confirm = () => true;
  for (const button of m.d.querySelectorAll('#modelsSubsList button')) {
    button.click();
    await m.page.settle();
  }
  assert.ok(m.row().includes(why), m.row());
  assert.ok(m.row().includes('login denied'), m.row());

  state.running = true;
  m.d.querySelector('#tabSequence').click(); // leave and reopen the Models tab
  await m.page.settle();
  m.d.querySelector('#tabModels').click();
  await m.page.settle();
  assert.ok(!m.row().includes(why), m.row());
  assert.ok(m.row().includes('login denied'), 'a login error of its own stays');
});
