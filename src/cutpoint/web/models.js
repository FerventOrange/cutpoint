// cutpoint editing workspace: the Models tab (chat and vision roles, keys) and subscription logins.
import { api } from './api.js';
import { $, AI_OTHER, aiModelChoices, h } from './util.js';
import { confirmAction, S, sentence } from './app.js';
import { loadAssistant } from './chat.js';

// ---------- Models tab: role management and subscription accounts ----------
const KEY_WHERE = { env: 'set (env)', keyring: 'set (keyring)' };
// jobs started from the UI get no key variables (keys.child_env), so a Vision key from the environment is CLI only
const VISION_KEY_WHERE = { ...KEY_WHERE, env: 'env (CLI only)' };
const AI_CUSTOM = 'custom';
const PROVIDER_TITLES = {
  ollama: 'Ollama (local)',
  anthropic: 'Anthropic API',
  'claude-code': 'Claude Code subscription',
  openai: 'OpenAI API',
  codex: 'Codex subscription',
  antigravity: 'Antigravity subscription',
  xai: 'xAI API',
  custom: 'Custom (from config.json)',
};
// in the order the server offers them: local, the API keys, then the subscription logins
const VISION_PROVIDERS = ['ollama', 'anthropic', 'openai', 'xai', 'claude-code', 'codex', 'antigravity'];

const CHAT_HINTS = {
  ollama: 'Runs with Ollama on this machine; nothing leaves it. An Ollama on another machine counts as hosted.',
  anthropic: 'Anthropic API (api.anthropic.com): your API key. Project data goes to Anthropic; it asks once first. Use at your own risk.',
  'claude-code': 'Claude Code subscription: your Claude login (see Connected accounts). Project data goes to Anthropic; it asks once first. Use at your own risk.',
  openai: 'OpenAI API (api.openai.com): your API key. Project data goes to OpenAI; it asks once first. Use at your own risk.',
  codex: 'Codex subscription: your ChatGPT login (see Connected accounts). Project data goes to OpenAI; it asks once first. Use at your own risk.',
  antigravity: 'Antigravity subscription: your Google login (see Connected accounts). Project data goes to Google; it asks once first. Use at your own risk.',
  xai: 'xAI API (api.x.ai): your API key. Project data goes to xAI; it asks once first. Use at your own risk.',
  custom: 'Set in config.json; no preset has it. Its model can be changed here; choose a preset and save to use that instead.',
};

const VISION_HINTS = {
  ollama: 'Rates video frames on this machine with an Ollama vision model (default qwen3.5:9b).',
  anthropic: 'Anthropic API (api.anthropic.com): rates video frames; billed to your API key. Frames and the words said in each moment go to Anthropic. Pick a model that accepts images. Use at your own risk.',
  'claude-code': 'Claude Code subscription: rates video frames with your Claude login. Frames and the words said in each moment go to Anthropic. Use at your own risk.',
  openai: 'OpenAI API (api.openai.com): rates video frames; billed to your API key. Frames and the words said in each moment go to OpenAI. Pick a model that accepts images. Use at your own risk.',
  codex: 'Codex subscription: rates video frames with your ChatGPT login. Frames and the words said in each moment go to OpenAI. Use at your own risk.',
  antigravity: 'Antigravity subscription: rates video frames with your Google login. Frames and the words said in each moment go to Google. Use at your own risk.',
  xai: 'xAI API (api.x.ai): rates video frames; billed to your API key. Frames and the words said in each moment go to xAI. Pick a model that accepts images. Use at your own risk.',
};

const AI_MODEL_HINTS = {
  ollama: 'A model Ollama has pulled. Default: the first of qwen3.5:9b and the vision model it has, else qwen3.5:9b.',
  anthropic: 'Default: claude-sonnet-5-5. claude-opus-5-5 is stronger but slower and costs more.',
  'claude-code': 'Default: claude-sonnet-5-5. claude-opus-5-5 is stronger but slower and costs more.',
  openai: 'Needed: an OpenAI model your key can use (e.g. gpt-4o).',
  codex: 'Models from your ChatGPT account.',
  antigravity: 'Models from your Antigravity account.',
  xai: 'Needed: an xAI model your key can use.',
  custom: "The custom provider's model. Default: the server's model, when it serves only one.",
};

const AI_TYPE_ID = 'Choose Other model… to enter an id.';
const AI_LIST_STATES = {
  key_needed: `The model list loads once there is an API key. ${AI_TYPE_ID}`,
  consent_needed: AI_TYPE_ID,
  unsupported: AI_TYPE_ID,
};

function maskEmail(email) {
  if (!email || typeof email !== 'string') return '';
  const at = email.indexOf('@');
  if (at <= 1) return email;
  const name = email.slice(0, at);
  const domain = email.slice(at + 1);
  return `${name[0]}***${name.length > 2 ? name[name.length - 1] : ''}@${domain}`;
}

function aiKeyOrder(env) {
  if (!env || !env.length) return 'Read from the system keyring.';
  if (env[0] === 'CUTPOINT_ASSISTANT_API_KEY') {
    return `Read from the system keyring, else from ${env[0]} when cutpoint ui starts.`;
  }
  return `Read from ${env.join(' or ')} when cutpoint ui starts, else from the system keyring.`;
}

function getChatModel() {
  const sel = $('#chatRoleModel');
  return sel.value === AI_OTHER ? $('#chatRoleModelOther').value.trim() : sel.value;
}

function getVisionModel() {
  const sel = $('#visionRoleModel');
  return sel.value === AI_OTHER ? $('#visionRoleModelOther').value.trim() : sel.value;
}

function renderChatRoleHints() {
  const sel = $('#chatRoleProvider');
  if (!sel) return;
  const choice = sel.value;
  const custom = choice === AI_CUSTOM;
  const p = custom
    ? S.ai?.providers?.find((x) => x.name === S.ai?.provider)
    : S.ai?.providers?.find((x) => x.name === choice);
  let hint = CHAT_HINTS[custom ? AI_CUSTOM : (p?.name ?? choice)] ?? '';
  if (custom && !S.ai?.local) hint += ' Use at your own risk.';
  if (p?.subscription && !custom && !S.logins?.find((l) => l.provider === p.name)?.logged_in) {
    hint += ' Not logged in yet.';
  }
  $('#chatRoleProviderHint').textContent = hint;
}

function renderVisionRoleHints() {
  const sel = $('#visionRoleProvider');
  if (!sel) return;
  const choice = sel.value;
  const p = S.visionAi?.providers?.find((x) => x.name === choice);
  // a base URL no preset has (config.json): the preset's text would name the wrong host
  let hint = S.visionAi?.custom && choice === S.visionAi.provider
    ? `The provider and base URL set in config.json (${S.visionAi.url}). ${S.visionAi.local ? 'It runs on this machine; no frames leave it.' : 'Frames and the words said in the moment go to that host. Use at your own risk.'}`
    : VISION_HINTS[choice] ?? '';
  if (p?.subscription && !S.logins?.find((l) => l.provider === p.name)?.logged_in) {
    hint += ' Not logged in yet.';
  }
  $('#visionRoleProviderHint').textContent = hint;
}

function renderChatRoleFields(model) {
  const sel = $('#chatRoleProvider');
  const choice = sel.value;
  const custom = choice === AI_CUSTOM;
  const p = custom
    ? S.ai?.providers?.find((x) => x.name === S.ai?.provider)
    : S.ai?.providers?.find((x) => x.name === choice);

  renderChatRoleHints();
  $('#chatRoleModel').setAttribute('aria-required', String(!custom && (p?.model_required ?? false)));
  renderRoleModelList('chat', model);
  $('#chatRoleModelHint').textContent = AI_MODEL_HINTS[custom ? AI_CUSTOM : (p?.name ?? choice)] ?? '';
  $('#chatRoleUrlField').hidden = $('#chatRoleUrlHint').hidden = !custom;
  $('#chatRoleUrlHint').textContent = custom ? 'Set in config.json (assistant.url); edit that file to change it.' : '';

  const keyed = p?.name in (S.ai?.keys ?? {});
  $('#chatRoleKey').hidden = !keyed;
  if (!keyed) return;
  const where = S.ai.keys[p.name];
  const host = S.ai.key_hosts?.[p.name] ?? '';
  const unsaved = !custom && S.ai.custom && p.name === S.ai.provider;
  $('#chatRoleKeyStatus').textContent = unsaved
    ? 'API key: save the models first to see or set the key for this provider.'
    : `API key${host ? ` for ${host}` : ''}: ${KEY_WHERE[where] ?? 'not set'}`;
  $('#chatRoleKeySave').disabled = $('#chatRoleKeyClear').disabled = unsaved;
  $('#chatRoleKeyHint').textContent = unsaved
    ? "A key belongs to the base URL's host; the provider's own variable counts only for its official API host."
    : `${aiKeyOrder(p.key_env || [])} Sent only to ${host || 'the base URL'}; never saved in config.json or shown.`;
  $('#chatRoleKeyMore').textContent = 'A key goes only over https:// unless the server is on this machine.';
}

function renderVisionRoleFields(model) {
  const sel = $('#visionRoleProvider');
  const choice = sel.value;
  const custom = Boolean(S.visionAi?.custom && choice === (S.visionAi?.provider || 'ollama'));
  const p = S.visionAi?.providers?.find((x) => x.name === choice);

  renderVisionRoleHints();
  $('#visionRoleModel').setAttribute('aria-required', String(!custom && (p?.model_required ?? false)));
  renderRoleModelList('vision', model);
  $('#visionRoleModelHint').textContent = AI_MODEL_HINTS[choice] ?? '';
  $('#visionRoleUrlField').hidden = $('#visionRoleUrlHint').hidden = !custom;
  $('#visionRoleUrlHint').textContent = custom ? 'Set in config.json (vision_assistant.url); edit that file to change it.' : '';
  renderVisionRoleKey(p);
  renderVisionConsent();
}

function renderVisionRoleKey(p) {
  const keyed = p?.name in (S.visionAi?.keys ?? {});
  $('#visionRoleKey').hidden = !keyed;
  if (!keyed) return;
  const where = S.visionAi.keys[p.name];
  const host = S.visionAi.key_hosts?.[p.name] ?? '';
  const keyStatus = $('#visionRoleKeyStatus');
  keyStatus.textContent = `API key${host ? ` for ${host}` : ''}: ${VISION_KEY_WHERE[where] ?? 'not set'}`;
  keyStatus.title = where === 'env' ? 'Analyses started from cutpoint ui use the key saved in the Models tab, not this environment variable.' : '';
  // jobs started here get no key variables (keys.child_env): only the keyring reaches them
  const env = (p.key_env || []).join(' or ');
  $('#visionRoleKeyHint').textContent = `Analyses started here read it from the system keyring${env ? `; ${env} counts only in a terminal` : ''}. Sent only to ${host || 'the base URL'}; never saved in config.json or shown.`;
  $('#visionRoleKeyMore').textContent = 'A key goes only over https:// unless the server is on this machine. A key for the same host serves both roles.';
}

function renderVisionConsent() {
  const box = $('#visionConsent');
  const status = $('#visionConsentStatus');
  if (!box || !status) return;

  const sel = $('#visionRoleProvider');
  const unsaved = sel.value !== sel.dataset.saved;
  const consent = S.visionConsent;

  status.textContent = '';
  box.hidden = true;

  if (unsaved) {
    if (sel.value !== 'ollama') {
      status.textContent = 'Save models first to review and allow this provider.';
    }
    return;
  }

  if (!consent || consent.key === null) {
    if (sel.value === 'ollama') {
      status.textContent = 'Local model: no frame data leaves this machine.';
    }
    return;
  }

  // A missing key or login comes first: no allowing a provider that cannot be asked yet.
  const noKey = consent.missing_key || '';
  if (consent.needed || noKey) {
    box.hidden = false;
    const host = consent.host || 'the remote service';
    $('#visionConsentTitle').textContent = noKey ? (consent.missing_kind === 'login' ? 'Login needed' : 'API key needed') : `Send video frames to ${host}?`;
    $('#visionConsentText').textContent = `Rating moments sends video frames, transcript excerpts, file names and times, on-screen text and telemetry to ${host}. Nothing is sent until you allow it.`;
    $('#visionConsentMore').textContent = 'Describing event windows sends the same. The AI Chat gets other data (analysis results and whole transcripts; frames only when you allow them for a project); allowing one never covers the other. Permission is saved in config.json.';
    if (noKey) $('#visionConsentText').textContent = sentence(noKey);
    $('#visionConsentMore').parentElement.hidden = !!noKey;
    $('#visionConsentOk').hidden = !!noKey;
    $('#visionConsentOk').disabled = false;
  } else {
    status.textContent = `Permission granted: frame ratings allowed with ${consent.host || 'this provider'}.`;
  }
}

async function giveVisionConsent() {
  if (!S.visionConsent?.key) return;
  const btn = $('#visionConsentOk');
  btn.disabled = true;
  try {
    const r = await api('/api/assistant/consent', {
      method: 'POST',
      body: { role: 'vision', key: S.visionConsent.key },
    });
    S.visionConsent = r.consent;
    renderVisionConsent();
    const saveStatus = $('#modelsSaveStatus');
    saveStatus.className = 'hint ok';
    saveStatus.textContent = 'Vision permission granted.';
  } catch (e) {
    btn.disabled = false;
    $('#modelsError').textContent = `Permission grant failed: ${e.message}`;
  }
}

function renderRoleModelList(role, model) {
  const isChat = role === 'chat';
  const sel = $(isChat ? '#chatRoleModel' : '#visionRoleModel');
  const input = $(isChat ? '#chatRoleModelOther' : '#visionRoleModelOther');
  const otherField = $(isChat ? '#chatRoleModelOtherField' : '#visionRoleModelOtherField');
  const providerSel = $(isChat ? '#chatRoleProvider' : '#visionRoleProvider');
  if (!sel || !providerSel) return;
  const provider = providerSel.value;
  const custom = provider === AI_CUSTOM;
  const p = isChat
    ? (custom ? S.ai?.providers?.find((x) => x.name === S.ai?.provider) : S.ai?.providers?.find((x) => x.name === provider))
    : S.visionAi?.providers?.find((x) => x.name === provider);

  const typing = model === undefined && sel.value === AI_OTHER;
  const current = model ?? (typing ? input.value.trim() : (sel.value === AI_OTHER ? input.value.trim() : sel.value));

  const catalog = isChat ? S.aiModels?.[provider] : S.visionModels?.[provider];
  const { options, value } = aiModelChoices(current, catalog, {
    typing,
    required: !custom && (p?.model_required ?? false),
    placeholder: custom ? '' : (p?.placeholder ?? ''),
    fallback: custom ? [] : (p?.models ?? []),
  });
  sel.replaceChildren(...options.map((o) => h('option', { value: o.value, text: o.text })));
  sel.value = value;
  if (!typing && value !== AI_OTHER) input.value = '';
  otherField.hidden = value !== AI_OTHER;
}

async function loadRoleModels(role, refresh) {
  const isChat = role === 'chat';
  const sel = $(isChat ? '#chatRoleProvider' : '#visionRoleProvider');
  const status = $(isChat ? '#chatRoleModelsStatus' : '#visionRoleModelsStatus');
  if (!sel || !status) return;
  const choice = sel.value;
  const seqKey = isChat ? 'chatModelsSeq' : 'visionModelsSeq';
  const seq = (S[seqKey] = (S[seqKey] ?? 0) + 1);
  status.textContent = '';

  const meta = isChat ? S.ai : S.visionAi;
  if (!meta) return;
  const p = isChat
    ? (choice === AI_CUSTOM ? S.ai?.providers?.find((x) => x.name === S.ai?.provider) : S.ai?.providers?.find((x) => x.name === choice))
    : S.visionAi?.providers?.find((x) => x.name === choice);

  if (p?.subscription) {
    const login = S.logins?.find((l) => l.provider === choice);
    if (!login?.logged_in) {
      status.textContent = 'Account not connected. Log in under Connected accounts to view and use models.';
      return;
    }
  }

  status.textContent = 'Loading the model list…';
  let r;
  try {
    const isCustomVision = !isChat && Boolean(S.visionAi?.custom && choice === (S.visionAi?.provider || 'ollama'));
    const queryProvider = isCustomVision ? AI_CUSTOM : choice;
    const roleParam = isChat ? 'assistant' : 'vision';
    r = await api(`/api/assistant/models?provider=${encodeURIComponent(queryProvider)}&role=${roleParam}${refresh ? '&refresh=1' : ''}`);
  } catch (e) {
    r = { state: 'error', models: [], message: e.message };
  }

  if (seq !== S[seqKey] || sel.value !== choice) {
    if (status.textContent === 'Loading the model list…') status.textContent = '';
    return;
  }

  const catalog = isChat ? (S.aiModels ??= {}) : (S.visionModels ??= {});
  catalog[choice] = r.models ?? [];

  renderRoleModelList(role);

  const n = catalog[choice].length;
  if (r.state === 'ok') {
    status.textContent = n ? '' : `The provider listed no models. ${AI_TYPE_ID}`;
  } else if (r.state === 'error') {
    const cleanMsg = String(r.message ?? '').trim().replace(/\.+$/, '');
    status.textContent = cleanMsg
      ? `Could not load model list: ${cleanMsg}. ${AI_TYPE_ID}`
      : `Could not load model list. ${AI_TYPE_ID}`;
  } else {
    const msg = r.message ? String(r.message).trim().replace(/\.+$/, '') + '.' : '';
    status.textContent = [msg, AI_LIST_STATES[r.state] ?? AI_TYPE_ID].filter(Boolean).join(' ');
  }
}

const SUB_POLL_MS = 1500;
// ms a subscription login is polled at most: the server's own limit (ui/settings.py LOGIN_TIMEOUT_S, 300 s,
// after which it answers "login timed out") plus a few polls, so the server's answer normally comes first
const SUB_LOGIN_MS = 310_000;
const PROXY_START_FAILED = 'the subscription proxy did not start'; // ui/routes/shared.py proxy_error and ui/routes/settings.py Log in
const subNote = (...parts) => $('#modelsSubsStatus')?.replaceChildren(...parts);

async function loadLogins({ started = false } = {}) {
  let r;
  try {
    r = await api('/api/assistant/logins');
  } catch (e) {
    return subNote(e.message);
  }
  S.logins = r.logins;
  S.proxyRunning = r.proxy_running;
  if (r.proxy_running) { // it runs now: drop the errors of an earlier start, keep login errors
    S.proxyError = '';
    for (const [p, msg] of Object.entries(S.loginErrors ?? {})) {
      if (msg.startsWith(PROXY_START_FAILED)) delete S.loginErrors[p];
    }
  }
  // a login kept from an earlier run needs the proxy to show: the server starts it (#821)
  if (!started && !S.proxyStarting && !r.proxy_running && r.saved_login) return startSubProxy();
  renderLogins();
  await enumerateSubCatalogs(false);

  const chatProv = $('#chatRoleProvider')?.value;
  const chatP = chatProv === AI_CUSTOM
    ? S.ai?.providers?.find((x) => x.name === S.ai?.provider)
    : S.ai?.providers?.find((x) => x.name === chatProv);
  if (chatP?.subscription) loadRoleModels('chat', false);

  const visProv = $('#visionRoleProvider')?.value;
  const visP = S.visionAi?.providers?.find((x) => x.name === visProv);
  if (visP?.subscription) loadRoleModels('vision', false);
}

async function enumerateSubCatalogs(refresh) {
  if (!S.logins) return;
  S.subCatalogs ??= {};
  const subs = S.logins.filter((l) => l.logged_in);
  if (!subs.length) {
    renderLogins();
    return;
  }

  const chatSeq = S.chatModelsSeq;
  const seq = (S.subCatalogSeq = (S.subCatalogSeq ?? 0) + 1);
  await Promise.all(subs.map(async (l) => {
    try {
      const r = await api(`/api/assistant/models?provider=${encodeURIComponent(l.provider)}${refresh ? '&refresh=1' : ''}`);
      if (seq !== S.subCatalogSeq) return;
      if (r.state === 'ok') {
        S.aiModels ??= {};
        S.aiModels[l.provider] = r.models ?? [];
        S.subCatalogs[l.provider] = { state: 'ok', count: (r.models ?? []).length };
      } else {
        S.subCatalogs[l.provider] = { state: r.state, error: r.message, count: 0 };
      }
    } catch (e) {
      if (seq !== S.subCatalogSeq) return;
      S.subCatalogs[l.provider] = { state: 'error', error: e.message, count: 0 };
    }
  }));

  if (seq !== S.subCatalogSeq) return;
  renderLogins();

  const chatProv = $('#chatRoleProvider')?.value;
  if (subs.some((s) => s.provider === chatProv) && chatSeq === S.chatModelsSeq) {
    renderRoleModelList('chat');
    if ($('#chatRoleModelsStatus')?.textContent === 'Loading the model list…') {
      $('#chatRoleModelsStatus').textContent = '';
    }
  }
  const visionProv = $('#visionRoleProvider')?.value;
  if (subs.some((s) => s.provider === visionProv)) loadRoleModels('vision', refresh);
}


/** Start the proxy for a login kept from an earlier run, then list the logins again. */
async function startSubProxy() {
  S.proxyStarting = true;
  S.proxyError = '';
  renderLogins();
  try {
    await api('/api/assistant/proxy', { method: 'POST' });
  } catch (e) {
    S.proxyError = e.message;
  }
  S.proxyStarting = false;
  await loadLogins({ started: true });
}

function renderLogins() {
  S.loginPending ??= new Set();
  S.loginStarting ??= new Set();
  S.loginErrors ??= {};
  const list = $('#modelsSubsList');
  if (!list || !S.logins) return;

  list.replaceChildren(...S.logins.map((l) => {
    const starting = S.proxyStarting || S.loginStarting.has(l.provider);
    const wait = starting || S.loginPending.has(l.provider);
    const title = PROVIDER_TITLES[l.provider] ?? l.title;
    const cat = S.subCatalogs?.[l.provider];
    const failed = S.loginErrors[l.provider] || S.proxyError;

    let badgeClass = 'disconnected';
    let badgeText = 'Not connected';
    if (starting) {
      badgeText = 'Starting the proxy…';
    } else if (wait) {
      badgeText = 'Waiting for browser…';
    } else if (l.logged_in) {
      if (l.expired) {
        badgeClass = 'error';
        badgeText = 'Expired';
      } else {
        badgeClass = 'connected';
        badgeText = 'Connected';
      }
    }

    let detailText = '';
    if (starting) {
      detailText = 'Starting the subscription proxy…';
    } else if (wait) {
      detailText = 'Sign in in the browser window that opened.';
    } else if (l.logged_in) {
      const emailMasked = maskEmail(l.email);
      const org = (l.org ?? '').replace(/[^\s@]+@[^\s@]+/g, maskEmail); // an org may be named after the email
      const who = [emailMasked, org].filter(Boolean).join(' · ');
      detailText = who ? `${who}` : 'Subscription active';
    } else if (failed || l.error) {
      detailText = failed || l.error;
    } else {
      detailText = 'Not logged in. Log in to use this subscription without an API key.';
    }

    // shown only when it fails: Connected accounts never counts the models
    const countNode = l.logged_in && !wait && cat?.state === 'error'
      ? h('span', { class: 'error', text: `Catalog error: ${cat.error || 'Failed to load'}` })
      : null;

    const acts = [];
    if (!l.logged_in || wait) {
      acts.push(h('button', {
        type: 'button',
        class: 'btn small',
        disabled: wait,
        text: starting ? 'Starting…' : wait ? 'Waiting…' : 'Log in',
        onclick: () => subLogin(l.provider),
      }));
    } else {
      if (cat?.state === 'error') {
        acts.push(h('button', {
          type: 'button',
          class: 'btn small',
          text: 'Retry catalog',
          onclick: () => enumerateSubCatalogs(true),
        }));
      } else {
        acts.push(h('button', {
          type: 'button',
          class: 'btn small',
          text: 'Refresh catalog',
          onclick: () => enumerateSubCatalogs(true),
        }));
      }
      acts.push(h('button', {
        type: 'button',
        class: 'btn small',
        text: 'Log out',
        onclick: () => subLogout(l.provider),
      }));
    }

    return h('li', { class: 'subs-item' },
      h('div', { class: 'subs-item-head' },
        h('span', { class: 'subs-title', text: title }),
        h('span', { class: `subs-badge ${badgeClass}`, text: badgeText }),
      ),
      h('div', { class: 'subs-detail' },
        h('span', { class: 'hint', text: detailText }),
        countNode,
      ),
      h('div', { class: 'row' }, ...acts),
    );
  }));

  renderChatRoleHints();
  renderVisionRoleHints();
}

/** ``url`` when it is a sign-in page to open: https, or http on this machine; else null (a ``javascript:`` URL would run in this page). */
function signInUrl(url) {
  try {
    const u = new URL(url);
    if (u.protocol === 'https:' || (u.protocol === 'http:' && ['127.0.0.1', 'localhost', '[::1]'].includes(u.hostname))) return url;
  } catch { /* not a URL */ }
  return null;
}

async function subLogin(provider) {
  if (S.loginPending?.has(provider)) return; // already waiting for this sign-in: one poll per provider
  const title = PROVIDER_TITLES[provider] ?? provider;
  if (!(await confirmAction(`Log in with ${title}?`, 'Use at your own risk.', 'Log in', false))) return;
  if (provider === 'antigravity') {
    const ok = await confirmAction('Continue the Antigravity sign-in?', 'Antigravity sign-in may provision a free-tier project if your account has none.', 'Continue', false);
    if (!ok) return;
  }
  subNote();
  S.loginStarting ??= new Set();
  S.loginErrors ??= {};
  delete S.loginErrors[provider];
  S.proxyError = '';
  // with no proxy running, the server starts it first (#821): say so on the row
  if (S.proxyRunning === false) S.loginStarting.add(provider);
  renderLogins();
  let r;
  try {
    r = await api('/api/assistant/login', { method: 'POST', body: { provider } });
  } catch (e) {
    S.loginStarting.delete(provider);
    S.loginErrors[provider] = e.message;
    renderLogins();
    return subNote(e.message);
  }
  S.loginStarting.delete(provider);
  S.proxyRunning = true;
  const url = signInUrl(r.url);
  if (!url) {
    S.loginErrors[provider] = 'the sign-in address is not a web page';
    renderLogins();
    return subNote(`Login failed: ${S.loginErrors[provider]}.`);
  }
  if (S.loginPending.has(provider)) return; // another click got here first and polls already
  S.loginPending.add(provider);
  window.open(url, '_blank', 'noopener,noreferrer');
  subNote('Finish signing in in the browser tab that opened. ',
    h('a', { href: url, target: '_blank', rel: 'noopener noreferrer', text: 'Open the sign-in page again' }));
  renderLogins();
  let late = false;
  const deadline = setTimeout(() => (late = true), SUB_LOGIN_MS);
  try {
    while (S.loginPending.has(provider)) {
      await new Promise((resolve) => setTimeout(resolve, SUB_POLL_MS));
      if (late) {
        subNote('Login failed: login timed out'); // the server's words for its own deadline
        break;
      }
      let s;
      try {
        s = await api(`/api/assistant/login/status?provider=${encodeURIComponent(provider)}`);
      } catch (e) {
        subNote(e.message);
        break;
      }
      if (s.state === 'pending') continue;
      subNote(s.state === 'done' ? 'Logged in.' : `Login failed: ${s.error}`);
      break;
    }
  } finally {
    // also when a reply is not what the page expects: the provider must not stay waiting
    clearTimeout(deadline);
    S.loginPending.delete(provider);
    renderLogins();
  }
  await loadLogins();
  await refreshVisionConsent();
  for (const statusId of ['#chatRoleModelsStatus', '#visionRoleModelsStatus']) {
    const el = $(statusId);
    if (el && el.textContent.includes('Account not connected')) {
      el.textContent = '';
    }
  }
  if (S.id) loadAssistant();
}

async function subLogout(provider) {
  subNote();
  try {
    await api('/api/assistant/logout', { method: 'POST', body: { provider } });
    subNote('Logged out.');
  } catch (e) {
    subNote(e.message);
  }
  delete S.aiModels?.[provider];
  delete S.visionModels?.[provider];
  delete S.subCatalogs?.[provider];
  await loadLogins();
  await refreshVisionConsent();

  const chatProv = $('#chatRoleProvider')?.value;
  if (chatProv === provider) {
    renderRoleModelList('chat');
    renderChatRoleFields();
    loadRoleModels('chat', false);
  }
  const visionProv = $('#visionRoleProvider')?.value;
  if (visionProv === provider) {
    renderRoleModelList('vision');
    renderVisionRoleFields();
    loadRoleModels('vision', false);
  }
  if (S.id) loadAssistant();
}


async function saveAiKey(clear, role = 'chat') {
  if (role === 'vision') return saveVisionKey(clear);
  const custom = $('#chatRoleProvider').value === AI_CUSTOM;
  const name = custom ? S.ai?.provider : $('#chatRoleProvider').value;
  const p = S.ai?.providers?.find((x) => x.name === name);
  const input = $('#chatRoleKeyInput');
  const err = $('#modelsError');
  err.textContent = '';
  const key = input.value.trim();
  if (!p) return;
  if (!clear && !key) {
    err.textContent = 'Paste the API key first.';
    return input.focus();
  }
  input.value = '';
  try {
    const r = await api(clear ? '/api/assistant/key/clear' : '/api/assistant/key',
      { method: 'POST', body: clear ? { provider: p.name } : { provider: p.name, key } });
    S.ai.keys = r.keys;
    S.ai.key_hosts = r.key_hosts;
    shareKeyStatus(S.visionAi, r, p.name);
  } catch (e) {
    err.textContent = e.message;
    return;
  }
  renderChatRoleFields();
  if (S.visionAi && $('#visionRoleProvider')?.value === p.name) {
    renderVisionRoleFields();
    await refreshVisionConsent();
  }
  $('#chatRoleKeyStatus').textContent += clear
    ? '. Removed from the keyring.'
    : '. Saved in the system keyring; it is checked when you send the first message.';
  if (!clear) loadRoleModels('chat', false);
  if (S.id) loadAssistant();
}

/** A key is kept per provider and host: copy where `provider`'s key comes from into the other
 * role's view when that role uses the same host. */
function shareKeyStatus(view, r, provider) {
  if (view?.keys && view.key_hosts?.[provider] === r.key_hosts?.[provider]) {
    view.keys[provider] = r.keys[provider];
  }
}

async function saveVisionKey(clear) {
  const name = $('#visionRoleProvider').value;
  const p = S.visionAi?.providers?.find((x) => x.name === name);
  const input = $('#visionRoleKeyInput');
  const err = $('#modelsError');
  err.textContent = '';
  const key = input.value.trim();
  if (!p) return;
  if (!clear && !key) {
    err.textContent = 'Paste the API key first.';
    return input.focus();
  }
  input.value = '';
  try {
    const r = await api(clear ? '/api/assistant/key/clear' : '/api/assistant/key',
      { method: 'POST', body: clear ? { provider: p.name, role: 'vision' } : { provider: p.name, key, role: 'vision' } });
    S.visionAi.keys = r.keys;
    S.visionAi.key_hosts = r.key_hosts;
    shareKeyStatus(S.ai, r, p.name);
  } catch (e) {
    err.textContent = e.message;
    return;
  }
  renderVisionRoleFields();
  $('#visionRoleKeyStatus').textContent += clear ? '. Removed from the keyring.' : '. Saved in the system keyring.';
  if (S.ai && $('#chatRoleProvider')?.value === p.name) renderChatRoleFields();
  await refreshVisionConsent();
  if (!clear) loadRoleModels('vision', false);
}

/** The consent box depends on keys and logins: re-read it after they change. */
async function refreshVisionConsent() {
  try {
    S.visionConsent = (await api('/api/settings')).vision_consent ?? null;
  } catch {
    // keep the last state
  }
  renderVisionConsent();
}

function fillChatRoleSettings(a) {
  if (!a) return;
  S.ai = a;
  S.aiModels ??= {};
  const sel = $('#chatRoleProvider');
  const options = a.providers.map((p) => h('option', { value: p.name, text: PROVIDER_TITLES[p.name] ?? p.title }));
  if (a.custom) options.push(h('option', { value: AI_CUSTOM, text: 'Custom (from config.json)' }));
  sel.replaceChildren(...options);
  sel.value = sel.dataset.saved = a.custom ? AI_CUSTOM : a.provider;
  $('#chatRoleModel').dataset.saved = a.model;
  $('#chatRoleUrl').value = a.url;
  $('#chatRoleKeyInput').value = '';
  renderChatRoleFields(a.model);
  loadRoleModels('chat', false);
}

function fillVisionRoleSettings(v, enabled, defaultLocalModel, consent) {
  S.visionConsent = consent ?? null;
  const box = $('#visionRoleEnabled');
  box.checked = enabled === true;
  box.dataset.saved = String(box.checked);
  $('#visionRoleEnabledHint').textContent = `Rates moments using video frames (the frame rating only). Off by default.${defaultLocalModel ? ` Local model: ${defaultLocalModel}.` : ''}${S.project?.kind === '360' ? ' Vision rating is not used for 360 footage.' : ''}`;

  S.visionAi = v ?? {
    provider: 'ollama',
    model: defaultLocalModel || '',
    url: '',
    providers: (S.ai?.providers ?? []).filter((p) => VISION_PROVIDERS.includes(p.name)),
  };
  S.visionModels ??= {};
  const sel = $('#visionRoleProvider');
  const providers = (S.visionAi.providers || []).filter((p) => VISION_PROVIDERS.includes(p.name));
  const list = providers.length ? providers : VISION_PROVIDERS.map((name) => ({ name, title: PROVIDER_TITLES[name] }));
  sel.replaceChildren(...list.map((p) => h('option', { value: p.name, text: PROVIDER_TITLES[p.name] ?? p.title })));
  sel.value = sel.dataset.saved = S.visionAi.provider || 'ollama';
  $('#visionRoleModel').dataset.saved = S.visionAi.model || '';
  $('#visionRoleUrl').dataset.saved = S.visionAi.url || '';
  $('#visionRoleUrl').value = S.visionAi.url || '';
  renderVisionRoleFields(S.visionAi.model);
  renderVisionConsent();
  loadRoleModels('vision', false);
}

async function loadModelsSettings() {
  let st;
  try {
    st = await api('/api/settings');
  } catch (e) {
    $('#modelsError').textContent = e.message;
    return false;
  }
  fillChatRoleSettings(st.assistant);
  fillVisionRoleSettings(st.vision_assistant, st.vision, st.vision_model, st.vision_consent);
  return true;
}

async function saveModels() {
  const err = $('#modelsError');
  const status = $('#modelsSaveStatus');
  err.textContent = status.textContent = '';
  status.className = 'hint';

  const chatProvider = $('#chatRoleProvider').value;
  const chatModel = getChatModel();
  const savedChatProvider = $('#chatRoleProvider').dataset.saved;
  const savedChatModel = $('#chatRoleModel').dataset.saved ?? '';
  const customChat = chatProvider === AI_CUSTOM;
  const chatOffer = customChat ? S.ai?.providers?.find((x) => x.name === S.ai?.provider) : S.ai?.providers?.find((x) => x.name === chatProvider);
  const chatChanged = (chatProvider !== savedChatProvider) || (chatModel !== savedChatModel);

  if (chatChanged && !customChat && chatOffer?.model_required && !chatModel) {
    err.textContent = `Choose a model for ${PROVIDER_TITLES[chatProvider] ?? chatOffer?.title ?? chatProvider}.`;
    return $('#chatRoleModel').focus();
  }

  const visionEnabled = $('#visionRoleEnabled').checked;
  const savedVisionEnabled = $('#visionRoleEnabled').dataset.saved === 'true';
  const visionEnabledChanged = visionEnabled !== savedVisionEnabled;

  const visionProvider = $('#visionRoleProvider').value;
  const visionModel = getVisionModel();
  const savedVisionProvider = $('#visionRoleProvider').dataset.saved;
  const savedVisionModel = $('#visionRoleModel').dataset.saved ?? '';
  const customVision = Boolean(S.visionAi?.custom && visionProvider === (S.visionAi?.provider || 'ollama'));
  const visionChanged = (visionProvider !== savedVisionProvider) || (visionModel !== savedVisionModel);
  const visionOffer = S.visionAi?.providers?.find((x) => x.name === visionProvider);

  if ((visionChanged || (visionEnabledChanged && visionEnabled)) && visionEnabled && visionOffer?.model_required && !visionModel) {
    err.textContent = `Choose a model for ${PROVIDER_TITLES[visionProvider] ?? visionOffer?.title ?? visionProvider} vision.`;
    return $('#visionRoleModel').focus();
  }

  if (!chatChanged && !visionChanged && !visionEnabledChanged) {
    status.className = 'hint';
    status.textContent = 'No changes to save.';
    return;
  }

  const body = {};
  if (chatChanged) {
    body.assistant = customChat ? { model: chatModel } : { provider: chatProvider, model: chatModel, url: '' };
  }
  if (visionChanged) {
    body.vision_assistant = customVision ? { model: visionModel } : { provider: visionProvider, model: visionModel, url: '' };
  }
  if (visionEnabledChanged) {
    body.vision = visionEnabled;
  }

  const btn = $('#btnSaveModels');
  btn.disabled = true;
  status.textContent = 'Saving models…';

  let st;
  try {
    st = await api('/api/settings', { method: 'PUT', body });
  } catch (e) {
    err.textContent = e.message;
    status.textContent = '';
    return;
  } finally {
    btn.disabled = false;
  }

  if (st.assistant) fillChatRoleSettings(st.assistant);
  if (st.vision_assistant) {
    fillVisionRoleSettings(st.vision_assistant, st.vision, st.vision_model, st.vision_consent);
  } else if (st.vision !== undefined) {
    $('#visionRoleEnabled').checked = st.vision === true;
    $('#visionRoleEnabled').dataset.saved = String($('#visionRoleEnabled').checked);
  }
  status.className = 'hint ok';
  status.textContent = 'Models saved successfully.';
  if (st.assistant_note) status.textContent += ` ${st.assistant_note}`;
  if (st.assistant_check) err.textContent = `Assistant notice: ${st.assistant_check}`;

  if (S.id) loadAssistant();
}

export async function openModelsTab() {
  if (S.modelsTabLoading) return;
  $('#modelsError').textContent = '';
  $('#modelsSaveStatus').textContent = '';

  if (!S.modelsTabInitialized) {
    S.modelsTabLoading = true;
    try {
      const ok = await loadModelsSettings();
      if (ok) S.modelsTabInitialized = true;
      await loadLogins();
    } finally {
      S.modelsTabLoading = false;
    }
  } else {
    await loadLogins();
  }
}

export function bindModelsTab() {
  $('#chatRoleProvider').addEventListener('change', (e) => {
    const back = e.target.value === e.target.dataset.saved;
    renderChatRoleFields(back ? $('#chatRoleModel').dataset.saved : '');
    loadRoleModels('chat', false);
  });
  $('#chatRoleModel').addEventListener('change', (e) => {
    S.chatModelsSeq = (S.chatModelsSeq ?? 0) + 1;
    $('#chatRoleModelOtherField').hidden = e.target.value !== AI_OTHER;
  });
  $('#chatRoleModelOther').addEventListener('input', () => {
    S.chatModelsSeq = (S.chatModelsSeq ?? 0) + 1;
  });
  $('#chatRoleModelsRefresh').addEventListener('click', () => loadRoleModels('chat', true));
  $('#chatRoleKeySave').addEventListener('click', () => saveAiKey(false));
  $('#chatRoleKeyClear').addEventListener('click', () => saveAiKey(true));
  $('#chatRoleKeyInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      saveAiKey(false);
    }
  });

  $('#visionRoleProvider').addEventListener('change', (e) => {
    const back = e.target.value === e.target.dataset.saved;
    renderVisionRoleFields(back ? $('#visionRoleModel').dataset.saved : '');
    loadRoleModels('vision', false);
  });
  $('#visionRoleModel').addEventListener('change', (e) => {
    S.visionModelsSeq = (S.visionModelsSeq ?? 0) + 1;
    $('#visionRoleModelOtherField').hidden = e.target.value !== AI_OTHER;
  });
  $('#visionRoleModelOther').addEventListener('input', () => {
    S.visionModelsSeq = (S.visionModelsSeq ?? 0) + 1;
  });
  $('#visionRoleModelsRefresh').addEventListener('click', () => loadRoleModels('vision', true));
  $('#visionConsentOk').addEventListener('click', giveVisionConsent);
  $('#visionRoleKeySave').addEventListener('click', () => saveAiKey(false, 'vision'));
  $('#visionRoleKeyClear').addEventListener('click', () => saveAiKey(true, 'vision'));
  $('#visionRoleKeyInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      saveAiKey(false, 'vision');
    }
  });

  $('#btnSaveModels').addEventListener('click', saveModels);
}
