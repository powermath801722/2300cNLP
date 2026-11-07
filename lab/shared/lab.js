// Shared page chrome for NLP Lab module pages.
// Usage in a module page:
//   <body data-slug="tokenize"> ... <main id="main">module content</main> ...
//   <script type="module" src="/lab/shared/lab.js"></script>   (renders header, hero, footer)
//   Module code: import { el, ready } from '/lab/shared/lab.js'; ... ready();  // marks window.__lab.ready
import { MODULES, TIERS, moduleBySlug, tierOf, hrefOf, isLive, LIVE } from '/lab/shared/modules.js';

export { MODULES, TIERS, moduleBySlug, tierOf, hrefOf, isLive, LIVE };

/** Tiny element helper: el('div', {class:'x', text:'hi', onclick:fn}, [children]) */
export function el(tag, attrs = {}, children = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'text') n.textContent = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k.startsWith('on') && typeof v === 'function') n.addEventListener(k.slice(2), v);
    else if (k === 'style' && typeof v === 'object') Object.assign(n.style, v);
    else n.setAttribute(k, v === true ? '' : v);
  }
  for (const c of [].concat(children)) if (c != null) n.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return n;
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
export const REDUCED = window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches;
export const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

/** Signal to tests (and people) that the module finished its initial load. */
export function ready(extra = {}) {
  window.__lab = Object.assign(window.__lab || {}, { ready: true }, extra);
  document.documentElement.dataset.labReady = 'true';
}

/** Report a fatal load error in the page (tests treat window.__lab.error as a failure). */
export function fail(msg) {
  window.__lab = Object.assign(window.__lab || {}, { error: String(msg) });
  const m = $('#main');
  if (m) m.prepend(el('div', { class: 'wrap' }, el('p', { class: 'msg', role: 'alert', text: String(msg) })));
}

const BRAID = '<svg width="30" height="20" viewBox="0 0 30 20" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" aria-hidden="true"><path d="M1 3c8 0 9 7 16 7h12"/><path d="M1 10c5 0 6-3 9-3s4 3 7 3"/><path d="M1 17c8 0 9-7 16-7"/></svg>';

export function header() {
  return el('header', {}, el('nav', { class: 'wrap nav', 'aria-label': 'Primary' }, [
    el('a', { class: 'brand', href: '/', html: BRAID + ' Braidwater <small>NLP Lab</small>' }),
    el('a', { class: 'l opt', href: '/lab/', text: 'All modules' }),
    el('a', { class: 'l opt', href: '/word2vec/', text: 'Word2Vec' }),
    el('a', { class: 'l', href: '/', text: '← Home' }),
  ]));
}

function hero(m) {
  const t = tierOf(m.tier);
  const idx = MODULES.indexOf(m);
  return el('section', { class: 'hero', 'aria-labelledby': 'mod-h' }, el('div', { class: 'wrap' }, [
    el('div', { class: 'crumbs' }, [el('a', { href: '/lab/', text: 'NLP Lab' }), '›', el('span', { text: `Tier ${t.n} · ${t.name}` }), '›', el('span', { text: `Module ${idx + 1} of ${MODULES.length}` })]),
    el('div', { class: 'row' }, [
      el('span', { class: 'tier', 'data-tier': t.n, text: `Tier ${t.n} · ${t.name}` }),
      m.kind === 'transformer' ? el('span', { class: 'badge', text: 'Pretrained transformer · downloads on first run' }) : el('span', { class: 'badge', text: 'Runs entirely in your browser' }),
    ]),
    el('h1', { id: 'mod-h', text: m.title }),
    el('p', { class: 'lede', text: m.summary }),
  ]));
}

function pager(m) {
  const i = MODULES.indexOf(m);
  const prev = MODULES.slice(0, i).reverse().find(isLive), next = MODULES.slice(i + 1).find(isLive);
  return el('nav', { class: 'wrap row', 'aria-label': 'Module navigation', style: { justifyContent: 'space-between', padding: '28px 0' } }, [
    prev ? el('a', { class: 'btn btn-secondary', href: hrefOf(prev), text: '← ' + prev.title }) : el('span'),
    next ? el('a', { class: 'btn btn-secondary', href: hrefOf(next), text: next.title + ' →' }) : el('a', { class: 'btn btn-secondary', href: '/lab/', text: 'All modules' }),
  ]);
}

export function footer() {
  return el('footer', {}, el('div', { class: 'wrap', html: '<span>Braidwater · NLP Lab</span><span><a href="/lab/">All modules</a> · <a href="/">Home</a></span>' }));
}

/** Renders header/hero before #main and pager/footer after it, based on <body data-slug>. */
function chrome() {
  const slug = document.body.dataset.slug;
  const main = document.getElementById('main');
  document.body.prepend(header());
  if (slug && main) {
    const m = moduleBySlug(slug);
    if (m) {
      document.title = `${m.title} — NLP Lab — Braidwater`;
      main.before(hero(m));
      main.after(pager(m));
    }
  }
  document.body.append(footer());
}

if (!window.__labChrome) { window.__labChrome = true; chrome(); }

/** Lazy-load Plotly (self-hosted) once. */
let plotlyP;
export function plotly() {
  if (window.Plotly) return Promise.resolve(window.Plotly);
  plotlyP ||= new Promise((res, rej) => {
    const s = el('script', { src: '/assets/vendor/plotly-2.35.2.min.js' });
    s.onload = () => res(window.Plotly); s.onerror = () => rej(new Error('Plotting library failed to load'));
    document.head.append(s);
  });
  return plotlyP;
}

/** Dark Plotly layout matching the site. */
export function plotLayout(extra = {}) {
  const grid = 'rgba(228,231,235,.08)', axis = { gridcolor: grid, zerolinecolor: 'rgba(228,231,235,.22)', color: '#9aa3ad', tickfont: { size: 11 } };
  return Object.assign({
    paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
    font: { family: 'Barlow, system-ui, sans-serif', color: '#e4e7eb' },
    margin: { l: 56, r: 16, t: 24, b: 48 },
    xaxis: { ...axis }, yaxis: { ...axis },
    hoverlabel: { bgcolor: '#1d2732', bordercolor: '#749dc4', font: { family: 'Barlow, sans-serif', color: '#e4e7eb' } },
    legend: { bgcolor: 'rgba(0,0,0,0)' },
  }, extra);
}
export const PLOT_CONFIG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'toImage'] };
export const COLORS = ['#7ea3c8', '#d8b66a', '#8fbf9a', '#d98c7a', '#b49bd6', '#6cc3c3'];

/** Fetch JSON with a friendly error. */
export async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Couldn’t load ${url} (${r.status})`);
  return r.json();
}
