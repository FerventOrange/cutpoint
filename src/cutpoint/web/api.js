// JSON client for the cutpoint UI server. Every call carries the per-run session token, which
// `cutpoint ui` prints inside the startup URL (#t=<token>); it is kept in sessionStorage only.

const KEY = 'cutpoint-token';
const NO_TOKEN = 'This page has no session token. Open the complete URL printed by `cutpoint ui`.';
const BAD_TOKEN = 'The server rejected this session token. Open the complete URL printed by `cutpoint ui` again.';

export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

/** Take the token from the URL fragment (then strip it), else from this tab's sessionStorage. */
function readToken() {
  const m = /^#t=([\w-]+)/.exec(location.hash);
  if (m) {
    sessionStorage.setItem(KEY, m[1]);
    history.replaceState(null, '', location.pathname + location.search);
  }
  return sessionStorage.getItem(KEY);
}

const token = readToken();

// Same-tab navigation to a fresh startup URL only changes the hash; reload so startup consumes it.
addEventListener('hashchange', () => {
  if (/^#t=[\w-]+/.test(location.hash)) location.reload();
});

async function detail(r) {
  try {
    const j = await r.json();
    const d = j.detail ?? j.error ?? j;
    if (typeof d === 'string') return d;
    if (Array.isArray(d)) return d.map((e) => e.msg ?? JSON.stringify(e)).join('; ');
    return JSON.stringify(d);
  } catch {
    return `${r.status} ${r.statusText}`.trim();
  }
}

export async function api(path, { method = 'GET', body } = {}) {
  if (!token) throw new ApiError(401, NO_TOKEN);
  let r;
  try {
    r = await fetch(path, {
      method,
      headers: {
        ...(body !== undefined && { 'Content-Type': 'application/json' }),
        'X-Cutpoint-Token': token,
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "Can't reach the cutpoint server. Is `cutpoint ui` still running?");
  }
  if (r.status === 403) throw new ApiError(403, BAD_TOKEN);
  if (!r.ok) throw new ApiError(r.status, await detail(r));
  return r.status === 204 ? null : r.json();
}

/** Validate the token and fetch the style names; call once at startup. */
export const startSession = () => api('/api/session');
