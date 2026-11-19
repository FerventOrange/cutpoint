// Module hooks for harness.mjs (registered there with the web folder's URL). They let each boot()
// import the real files in src/cutpoint/web with fresh module state, without copying them:
// - resolve: a relative import from a web module keeps its importer's query, so app.js?boot=3
//   imports api.js?boot=3, picker.js?boot=3 and util.js?boot=3 (one graph per boot, as one page
//   load has), never another boot's instances;
// - load: a .js file there is an ES module (the folder has no package.json saying so).
let web = null; // "file:///.../src/cutpoint/web/"

export function initialize(data) {
  web = data.web;
}

export async function resolve(specifier, context, next) {
  const out = await next(specifier, context);
  const parent = context.parentURL;
  if (!parent?.startsWith(web) || !out.url.startsWith(web) || !/^\.\.?\//.test(specifier)) return out;
  const { search } = new URL(parent);
  if (!search) return out;
  const url = new URL(out.url);
  url.search = search;
  return { ...out, url: url.href };
}

export async function load(url, context, next) {
  if (url.startsWith(web) && new URL(url).pathname.endsWith('.js')) return next(url, { ...context, format: 'module' });
  return next(url, context);
}
