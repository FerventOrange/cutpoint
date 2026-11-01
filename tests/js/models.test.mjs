// The Models tab (#575): per-role provider and model choice, subscription accounts, saving.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { boot, reply } from './harness.mjs';

const provider = (name, over = {}) => ({ name, title: name, model_required: false, placeholder: '', models: [], ...over });
const PROVIDERS = [
  provider('ollama', { placeholder: 'qwen3.5:9b', models: ['qwen3.5:9b'] }),
  provider('codex', { subscription: true, model_required: true }),
  provider('openai', { model_required: true, key_env: ['OPENAI_API_KEY'] }),
];

function settings(over = {}) {
  return {
    assistant: {
      provider: 'ollama', model: '', url: '', custom: false, local: true, providers: PROVIDERS,
      keys: { openai: 'none' }, key_hosts: { openai: 'api.openai.com' },
    },
    vision_assistant: { provider: 'ollama', model: '', url: '', custom: false, providers: PROVIDERS.filter((p) => p.name !== 'openai') },
    vision: false, vision_model: 'qwen3.5:9b', vision_consent: null,
    ...over,
  };
}

/**
 * Boot, open the Models tab. ``logins`` and ``catalog`` ({provider: models | {state, message}}) are what the
 * server answers; ``calls`` lists the requests the tab made after opening, as "METHOD path?query".
 */
async function modelsTab(t, { logins = [], catalog = {}, st = settings(), routes = {} } = {}) {
  t.mock.timers.enable({ apis: ['setTimeout'] });
  const state = { logins, st };
  const page = await boot({
    routes: {
      'GET /api/settings': () => state.st,
      'GET /api/assistant/logins': () => ({ logins: state.logins }),
      'GET /api/assistant/models': (req) => {
        const prov = new URLSearchParams(req.query).get('provider');
        const c = catalog[prov] ?? [];
        return Array.isArray(c) ? { state: 'ok', models: c } : c;
      },
      ...routes,
    },
  });
  t.after(() => page.close());
  const d = page.document;
  const start = page.requests.length;
  d.querySelector('#tabModels').click();
  await page.settle();
  return {
    page, d, state,
    $: (s) => d.querySelector(s),
    calls: () => page.requests.slice(start).map((r) => `${r.method} ${r.url}`),
    optionValues: (sel) => [...d.querySelector(sel).options].map((o) => o.value),
    async choose(sel, value) {
      const el = d.querySelector(sel);
      el.value = value;
      el.dispatchEvent(new page.window.Event('change'));
      await page.settle();
    },
    async click(sel) { d.querySelector(sel).click(); await page.settle(); },
  };
}

test('opening the tab shows the saved providers; a subscription not logged in loads no model list', async (t) => {
  const m = await modelsTab(t, { logins: [{ provider: 'codex', title: 'Codex', logged_in: false }] });
  assert.equal(m.$('#chatRoleProvider').value, 'ollama');
  assert.deepEqual(m.optionValues('#chatRoleProvider'), ['ollama', 'codex', 'openai']);
  assert.deepEqual(m.optionValues('#visionRoleProvider'), ['ollama', 'codex'], 'API-key providers cannot rate frames');

  await m.choose('#chatRoleProvider', 'codex');
  assert.match(m.$('#chatRoleModelsStatus').textContent, /Account not connected/);
  assert.match(m.$('#chatRoleProviderHint').textContent, /Not logged in yet/);
  assert.ok(!m.calls().some((c) => c.includes('provider=codex')), 'no catalog request without a login');
  m.page.assertNoUnhandled();
});

test('a logged-in subscription lists its account models for both roles and saves the chosen one', async (t) => {
  const m = await modelsTab(t, {
    logins: [{ provider: 'codex', title: 'Codex', logged_in: true, email: 'jane.doe@example.com' }],
    catalog: { codex: [{ id: 'gpt-5-codex', name: 'GPT-5 Codex' }, { id: 'gpt-5-mini' }] },
    routes: { 'PUT /api/settings': (req) => ({ ...settings(), assistant: { ...settings().assistant, provider: 'codex', model: req.body.assistant.model } }) },
  });
  await m.choose('#chatRoleProvider', 'codex');
  assert.deepEqual(m.optionValues('#chatRoleModel').slice(0, 3), ['', 'gpt-5-codex', 'gpt-5-mini']);
  assert.equal(m.$('#chatRoleModel').options[1].text, 'GPT-5 Codex (gpt-5-codex)');
  assert.ok(m.calls().includes('GET /api/assistant/models?provider=codex&role=assistant'));

  await m.choose('#chatRoleModel', 'gpt-5-mini');
  await m.click('#btnSaveModels');
  const put = m.page.requests.find((r) => r.method === 'PUT');
  assert.deepEqual(put.body, { assistant: { provider: 'codex', model: 'gpt-5-mini', url: '' } });
  assert.equal(m.$('#modelsSaveStatus').textContent, 'Models saved successfully.');
});

test('save: nothing changed sends nothing; a required model left empty is refused; a server error is shown', async (t) => {
  const m = await modelsTab(t, {
    logins: [{ provider: 'openai', title: 'OpenAI', logged_in: false }],
    routes: { 'PUT /api/settings': reply(400, { detail: 'model not allowed' }) },
  });
  const puts = () => m.page.requests.filter((r) => r.method === 'PUT').length;
  await m.click('#btnSaveModels');
  assert.equal(m.$('#modelsSaveStatus').textContent, 'No changes to save.');
  assert.equal(puts(), 0);

  await m.choose('#chatRoleProvider', 'openai'); // a model is required, none chosen
  await m.click('#btnSaveModels');
  assert.match(m.$('#modelsError').textContent, /Choose a model for/);
  assert.equal(puts(), 0);

  await m.choose('#chatRoleProvider', 'ollama');
  await m.choose('#chatRoleModel', m.optionValues('#chatRoleModel').at(-1)); // Other model…
  m.$('#chatRoleModelOther').value = 'llama-x';
  await m.click('#btnSaveModels');
  assert.equal(puts(), 1);
  assert.equal(m.$('#modelsError').textContent, 'model not allowed');
  assert.equal(m.$('#btnSaveModels').disabled, false);
});

test('a model list that arrives after the provider changed again is ignored', async (t) => {
  let release;
  const gate = new Promise((r) => { release = r; });
  const m = await modelsTab(t, {
    logins: [],
    routes: {
      'GET /api/assistant/models': async (req) => {
        const prov = new URLSearchParams(req.query).get('provider');
        if (prov === 'openai') { await gate; return { state: 'error', models: [{ id: 'late-model' }], message: 'late failure' }; }
        return { state: 'ok', models: [{ id: 'qwen3.5:9b' }] };
      },
    },
  });
  await m.choose('#chatRoleProvider', 'openai'); // its list is held back
  await m.choose('#chatRoleProvider', 'ollama');
  release();
  await m.page.settle();
  assert.ok(!m.optionValues('#chatRoleModel').includes('late-model'));
  assert.ok(m.optionValues('#chatRoleModel').includes('qwen3.5:9b'));
  assert.ok(!m.$('#chatRoleModelsStatus').textContent.includes('late failure'), 'its status does not land on the other provider');
});

test('a failing model list says why and still allows typing a model id', async (t) => {
  const m = await modelsTab(t, {
    catalog: { ollama: { state: 'error', models: [], message: 'connection refused.' } },
  });
  assert.equal(m.$('#chatRoleModelsStatus').textContent, 'Could not load model list: connection refused. Choose Other model… to enter an id.');
  assert.ok(m.optionValues('#chatRoleModel').includes('qwen3.5:9b'), 'the provider\'s built-in suggestions remain');
});

test('Connected accounts masks the email, also inside an organisation name', async (t) => {
  const m = await modelsTab(t, {
    logins: [
      { provider: 'codex', title: 'Codex', logged_in: true, email: 'jane.doe@example.com', org: 'team bob@corp.io' },
      { provider: 'openai', title: 'OpenAI', logged_in: false },
    ],
    catalog: { codex: [{ id: 'm' }] },
  });
  const items = [...m.d.querySelectorAll('#modelsSubsList li')];
  assert.equal(items.length, 2);
  assert.ok(!items[0].textContent.includes('jane.doe'), items[0].textContent);
  assert.ok(!items[0].textContent.includes('bob@'), items[0].textContent);
  assert.match(items[0].textContent, /j\*\*\*e@example\.com/);
  assert.match(items[0].textContent, /Connected/);
  assert.match(items[1].textContent, /Not connected/);
});

test('an API key: empty is refused locally, a pasted one is sent once and cleared from the field', async (t) => {
  const m = await modelsTab(t, {
    routes: { 'POST /api/assistant/key': (req) => ({ keys: { openai: 'keyring' }, key_hosts: { openai: 'api.openai.com' }, got: req.body }) },
  });
  await m.choose('#chatRoleProvider', 'openai');
  assert.equal(m.$('#chatRoleKey').hidden, false);
  await m.click('#chatRoleKeySave');
  assert.equal(m.$('#modelsError').textContent, 'Paste the API key first.');
  assert.ok(!m.page.requests.some((r) => r.path === '/api/assistant/key'));

  m.$('#chatRoleKeyInput').value = ' sk-test ';
  await m.click('#chatRoleKeySave');
  const post = m.page.requests.find((r) => r.path === '/api/assistant/key');
  assert.deepEqual(post.body, { provider: 'openai', key: 'sk-test' });
  assert.equal(m.$('#chatRoleKeyInput').value, '');
  assert.match(m.$('#chatRoleKeyStatus').textContent, /set \(keyring\).*Saved in the system keyring/);
});

test('vision consent: a hosted provider needs permission before frames are sent, granting posts the consent key', async (t) => {
  const st = settings({ vision_assistant: { ...settings().vision_assistant, provider: 'codex' } });
  st.vision_consent = { key: 'codex@host', needed: true, host: 'chatgpt.com' };
  const m = await modelsTab(t, {
    st, logins: [{ provider: 'codex', title: 'Codex', logged_in: true }], catalog: { codex: [{ id: 'm' }] },
    routes: { 'POST /api/assistant/consent': () => ({ consent: { key: 'codex@host', needed: false, host: 'chatgpt.com' } }) },
  });
  assert.equal(m.$('#visionConsent').hidden, false);
  assert.match(m.$('#visionConsentTitle').textContent, /Send video frames to chatgpt\.com\?/);

  await m.click('#visionConsentOk');
  assert.deepEqual(m.page.requests.find((r) => r.path === '/api/assistant/consent').body, { role: 'vision', key: 'codex@host' });
  assert.equal(m.$('#visionConsent').hidden, true);
  assert.match(m.$('#visionConsentStatus').textContent, /Permission granted/);

  // an unsaved provider choice never offers consent (it belongs to the saved one)
  await m.choose('#visionRoleProvider', 'ollama');
  assert.equal(m.$('#visionConsent').hidden, true);
});

// ---- subscription login ----

function loginSetup(t, { provider: name = 'codex', statuses = [{ state: 'done' }], confirms = [true] } = {}) {
  const opened = [];
  const state = { logins: [{ provider: name, title: name, logged_in: false }] };
  let polls = 0;
  const prompts = [];
  return modelsTab(t, {
    logins: state.logins,
    routes: {
      'POST /api/assistant/login': { url: 'https://auth.example/login' },
      'GET /api/assistant/login/status': () => {
        const s = statuses[Math.min(polls++, statuses.length - 1)];
        if (s.state === 'done') state.logins[0].logged_in = true;
        return s;
      },
    },
  }).then((m) => {
    m.page.window.open = (...a) => opened.push(a);
    let i = 0;
    m.page.window.confirm = (msg) => { prompts.push(msg); return confirms[Math.min(i++, confirms.length - 1)]; };
    return Object.assign(m, { opened, prompts, polls: () => polls, state });
  });
}

test('login: confirms, opens the sign-in page, polls until done, then shows the account as connected', async (t) => {
  const m = await loginSetup(t, { statuses: [{ state: 'pending' }, { state: 'pending' }, { state: 'done' }] });
  await m.click('#modelsSubsList button');
  assert.deepEqual(m.opened[0].slice(0, 2), ['https://auth.example/login', '_blank']);
  assert.match(m.$('#modelsSubsList').textContent, /Waiting for browser/);
  assert.equal(m.polls(), 0, 'first poll after 1.5 s');

  for (let i = 0; i < 3; i++) { t.mock.timers.tick(1500); await m.page.settle(); }
  assert.equal(m.polls(), 3);
  assert.equal(m.$('#modelsSubsStatus').textContent, 'Logged in.');
  assert.match(m.$('#modelsSubsList').textContent, /Connected/);
  assert.ok(!/Waiting/.test(m.$('#modelsSubsList').textContent));
});

test('login: a failed sign-in reports the reason and stops polling', async (t) => {
  const m = await loginSetup(t, { statuses: [{ state: 'failed', error: 'denied' }] });
  await m.click('#modelsSubsList button');
  t.mock.timers.tick(1500); await m.page.settle();
  assert.equal(m.$('#modelsSubsStatus').textContent, 'Login failed: denied');
  t.mock.timers.tick(6000); await m.page.settle();
  assert.equal(m.polls(), 1);
  assert.match(m.$('#modelsSubsList').textContent, /Not connected/);
});

test('login: declining the confirmation asks the server nothing', async (t) => {
  const m = await loginSetup(t, { confirms: [false] });
  await m.click('#modelsSubsList button');
  assert.ok(!m.page.requests.some((r) => r.path === '/api/assistant/login'));
  assert.equal(m.opened.length, 0);
});

test('login: Antigravity asks a second confirmation about provisioning a project', async (t) => {
  const m = await loginSetup(t, { provider: 'antigravity', confirms: [true, false] });
  await m.click('#modelsSubsList button');
  assert.equal(m.prompts.length, 2);
  assert.match(m.prompts[1], /provision/);
  assert.ok(!m.page.requests.some((r) => r.path === '/api/assistant/login'));
});

test('logout drops that provider\'s cached models and refreshes the accounts', async (t) => {
  const logins = [{ provider: 'codex', title: 'Codex', logged_in: true }];
  const m = await modelsTab(t, {
    logins, catalog: { codex: [{ id: 'gpt-5-mini' }] },
    routes: { 'POST /api/assistant/logout': () => { logins[0].logged_in = false; return {}; } },
  });
  await m.choose('#chatRoleProvider', 'codex');
  assert.ok(m.optionValues('#chatRoleModel').includes('gpt-5-mini'));
  const logout = [...m.d.querySelectorAll('#modelsSubsList button')].find((b) => b.textContent === 'Log out');
  logout.click();
  await m.page.settle();
  assert.deepEqual(m.page.requests.find((r) => r.path === '/api/assistant/logout').body, { provider: 'codex' });
  assert.ok(!m.optionValues('#chatRoleModel').includes('gpt-5-mini'));
  assert.match(m.$('#chatRoleModelsStatus').textContent, /Account not connected/);
});
