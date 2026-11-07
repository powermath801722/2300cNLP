// NLP Lab smoke + module tests.
//   python3 -m http.server 8765        (from repo root)
//   node tests/lab.mjs [baseUrl] [slug ...]
// For every module in lab/shared/modules.js: load the page, wait for window.__lab.ready, fail on page errors,
// then run tests/lab/<slug>.mjs (default export async ({ page, base, check, warn })) if it exists.
import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import path from 'node:path';

const here = path.dirname(fileURLToPath(import.meta.url));
const { chromium } = createRequire(import.meta.url)('playwright');
const root = path.resolve(here, '..');
const base = (process.argv[2] && process.argv[2].startsWith('http') ? process.argv[2] : 'http://localhost:8765').replace(/\/$/, '');
const only = process.argv.slice(2).filter((a) => !a.startsWith('http'));
const { MODULES } = await import(pathToFileURL(path.join(root, 'lab/shared/modules.js')).href);

let failed = 0;
const log = (s, m) => { console.log(`${s} ${m}`); if (s === 'FAIL') failed++; };
// 'Failed to load resource' duplicates the HTTP check above (which has the URL and skips HEAD probes).
const IGNORE = [/favicon/i, /Failed to load resource/i, /Download the React DevTools/i, /GPU stall|GL Driver Message|WebGL|Automatic fallback to software/i, /onnxruntime|VerifyEachNodeIsAssignedToAnEp|Some nodes were not assigned/i];

const browser = await chromium.launch();
async function visit(url, slug, fn, viewport = { width: 1440, height: 900 }) {
  const page = await browser.newPage({ viewport });
  const errs = [];
  page.on('pageerror', (e) => errs.push('pageerror: ' + e.message));
  page.on('console', (m) => { if (m.type() === 'error' && !IGNORE.some((r) => r.test(m.text()))) errs.push('console: ' + m.text()); });
  // HEAD requests are the hub's existence probes for unbuilt modules; real load failures are GETs.
  page.on('response', (r) => { if (r.status() >= 400 && r.request().method() !== 'HEAD' && !IGNORE.some((x) => x.test(r.url()))) errs.push(`HTTP ${r.status()} ${r.url()}`); });
  const check = (ok, msg) => log(ok ? 'PASS' : 'FAIL', `[${slug}] ${msg}`);
  const warn = (ok, msg) => log(ok ? 'PASS' : 'WARN', `[${slug}] ${msg}`);
  try {
    await page.goto(url, { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(() => (window.__lab && (window.__lab.ready || window.__lab.error)) || (window.__w2v && window.__w2v.ready), null, { timeout: 120000 });
    const err = await page.evaluate(() => window.__lab && window.__lab.error);
    check(!err, 'loads' + (err ? ': ' + err : ''));
    if (fn) await fn({ page, base, check, warn });
    const sw = await page.evaluate(() => document.documentElement.scrollWidth);
    check(sw <= viewport.width, `no horizontal scroll at ${viewport.width}px (scrollWidth ${sw})`);
  } catch (e) { check(false, 'threw: ' + e.message.split('\n')[0]); }
  for (const e of errs) check(false, e);
  await page.close();
}

if (!only.length || only.includes('hub')) {
  await visit(base + '/lab/', 'hub', async ({ page, check }) => {
    const n = await page.locator('.mod').count();
    check(n === MODULES.length, `hub lists all ${MODULES.length} modules, live or in development (got ${n})`);
  });
  await visit(base + '/lab/', 'hub-mobile', null, { width: 390, height: 844 });
}
for (const m of MODULES) {
  if (only.length && !only.includes(m.slug)) continue;
  if (m.href) continue; // external module with its own test (word2vec → tests/e2e.mjs)
  const url = `${base}/lab/${m.slug}/`;
  const testFile = path.join(here, 'lab', `${m.slug}.mjs`);
  const fn = existsSync(testFile) ? (await import(pathToFileURL(testFile).href)).default : null;
  if (!existsSync(path.join(root, 'lab', m.slug, 'index.html'))) { log('FAIL', `[${m.slug}] lab/${m.slug}/index.html missing`); continue; }
  await visit(url, m.slug, fn);
  await visit(url, m.slug + '-mobile', null, { width: 390, height: 844 });
}
await browser.close();
console.log(failed ? `\n${failed} check(s) failed` : '\nAll checks passed');
process.exit(failed ? 1 : 0);
