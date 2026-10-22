// "New project" dialog with a read-only local path browser (the server lists, nothing uploads).
import { api } from './api.js';
import { $, baseName, h } from './util.js';

const VIDEO = /\.(mp4|m4v|mov|mkv|avi|webm|mts|m2ts|ts|wmv|flv|mpe?g)$/i;

export function initNewProject(onCreated) {
  const dlg = $('#newDialog');
  const source = $('#newSource');
  const name = $('#newName');
  const style = $('#newStyle');
  const error = $('#newError');
  const browser = $('#browser');
  const browseBtn = $('#browseBtn');
  let current = null; // {path, parent}
  let nameTouched = false;

  const setError = (msg) => (error.textContent = msg || '');
  const stem = (p) => baseName(p.replace(/[\\/]+$/, '')).replace(/\.[^.]+$/, '');

  function useSource(path) {
    source.value = path;
    if (!nameTouched) name.value = stem(path);
  }

  async function load(path) {
    const msg = $('#browseMsg');
    msg.textContent = 'Loading…';
    try {
      const r = await api(`/api/browse${path ? `?path=${encodeURIComponent(path)}` : ''}`);
      current = r;
      $('#browsePath').textContent = r.path;
      $('#browseUp').disabled = !r.parent;
      const entries = r.entries
        .filter((e) => e.directory || VIDEO.test(e.name))
        .sort((a, b) => b.directory - a.directory || a.name.localeCompare(b.name, undefined, { numeric: true }));
      $('#browseList').replaceChildren(
        ...entries.map((e) =>
          h('li', {},
            h('button', {
              type: 'button',
              class: `entry ${e.directory ? 'dir' : 'file'}`,
              onclick: () => {
                if (e.directory) load(e.path);
                else {
                  useSource(e.path);
                  toggleBrowser(false);
                  source.focus();
                }
              },
            }, e.directory ? `${e.name}/` : e.name)),
        ),
      );
      msg.textContent = entries.length ? '' : 'No folders or videos here.';
    } catch (e) {
      msg.textContent = e.message;
      if (path) load(''); // the typed path was not browsable; fall back to the server default
    }
  }

  function toggleBrowser(show) {
    browser.hidden = !show;
    browseBtn.setAttribute('aria-expanded', String(show));
    if (show) load(source.value.trim());
  }

  browseBtn.addEventListener('click', () => toggleBrowser(browser.hidden));
  $('#browseUp').addEventListener('click', () => current?.parent && load(current.parent));
  $('#browseUse').addEventListener('click', () => {
    if (!current) return;
    useSource(current.path);
    toggleBrowser(false);
    source.focus();
  });
  name.addEventListener('input', () => (nameTouched = true));
  source.addEventListener('input', () => !nameTouched && (name.value = stem(source.value.trim())));
  $('#newCancel').addEventListener('click', () => dlg.close());

  $('#newForm').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const body = { name: name.value.trim(), source: source.value.trim(), style: style.value };
    if (!body.source) return setError('Choose a video file or a folder of footage.');
    if (!body.name) return setError('Give the project a name.');
    if (!body.style) return setError('No highlight style is available from the server.');
    setError('');
    $('#newCreate').disabled = true;
    try {
      const created = await api('/api/projects', { method: 'POST', body });
      dlg.close();
      onCreated(created);
    } catch (e) {
      setError(e.message);
    } finally {
      $('#newCreate').disabled = false;
    }
  });

  return function open(styles) {
    nameTouched = false;
    source.value = name.value = '';
    setError('');
    toggleBrowser(false);
    style.replaceChildren(...styles.map((s) => h('option', { value: s, text: s })));
    dlg.showModal();
    source.focus();
  };
}
