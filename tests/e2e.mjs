// End-to-end check for the landing page + Word2Vec Lab.
// Usage: python3 -m http.server 8765   (from repo root)
//        node tests/e2e.mjs [baseUrl]
import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';
const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const BASE = (process.argv[2] || 'http://localhost:8765').replace(/\/$/, '');
const SHOT_DIR = process.env.SHOT_DIR || path.join('tests', 'screenshots');
fs.mkdirSync(SHOT_DIR, { recursive: true });

let failed = 0;
const pass = (m) => console.log('PASS', m);
const fail = (m) => { failed++; console.log('FAIL', m); };
const warn = (m) => console.log('WARN', m);
const check = (ok, m) => (ok ? pass(m) : fail(m));

const IGNORE = [/favicon/i, /image-slots\.state\.json/, /\{\{ /, /status of 404/];
const errors = [];
function watch(page, tag) {
  page.on('pageerror', (e) => errors.push(`${tag} pageerror: ${e.message}`));
  page.on('console', (m) => {
    if (m.type() !== 'error') return;
    const t = m.text();
    if (!IGNORE.some((r) => r.test(t))) errors.push(`${tag} console: ${t}`);
  });
  page.on('response', (r) => {
    if (r.status() >= 400 && !IGNORE.some((x) => x.test(r.url())))
      errors.push(`${tag} HTTP ${r.status()} ${r.url()}`);
  });
}

async function step(name, fn) {
  try { await fn(); } catch (e) { fail(`${name}: ${e.message.split('\n')[0]}`); }
}

const browser = await chromium.launch();
try {
  // ── Landing page ────────────────────────────────────────────
  await step('landing desktop', async () => {
    const p = await browser.newPage({ viewport: { width: 1440, height: 900 } });
    watch(p, 'landing-desktop');
    await p.goto(BASE + '/', { waitUntil: 'networkidle' });
    check((await p.title()).includes('Braidwater'), 'landing title contains Braidwater');
    const link = p.locator('header a[href="/lab/"]', { hasText: 'NLP Lab' });
    await link.first().waitFor({ state: 'visible', timeout: 15000 });
    pass('landing desktop nav shows NLP Lab link');
    await p.close();
  });
  await step('landing mobile', async () => {
    const p = await browser.newPage({ viewport: { width: 390, height: 844 } });
    watch(p, 'landing-mobile');
    await p.goto(BASE + '/', { waitUntil: 'networkidle' });
    await p.locator('button[aria-label="Open menu"]').click();
    await p.locator('[role=dialog] a[href="/lab/"]', { hasText: 'NLP Lab' }).waitFor({ state: 'visible', timeout: 5000 });
    pass('landing mobile menu shows NLP Lab link');
    await p.close();
  });

  // ── Word2Vec Lab (desktop) ──────────────────────────────────
  const p = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 2 });
  watch(p, 'w2v');
  await step('w2v load', async () => {
    await p.goto(BASE + '/word2vec/', { waitUntil: 'domcontentloaded' });
    await p.waitForFunction(() => window.__w2v && window.__w2v.ready, null, { timeout: 60000 });
    const st = (await p.locator('#statustext').textContent()).trim();
    check(st.startsWith('Ready'), `status text starts with "Ready" (got "${st}")`);
  });

  const query = async (q) => {
    await p.fill('#q', q);
    await p.locator('#qform').evaluate((f) => f.requestSubmit());
    await p.waitForTimeout(400);
  };

  await step('king neighbors', async () => {
    await query('king');
    const items = p.locator('#lists ol.nn li');
    await items.first().waitFor({ timeout: 5000 });
    const n = await items.count();
    check(n === 10, `"king" gives exactly 10 neighbors (got ${n})`);
    const words = (await p.locator('#lists ol.nn .w').allTextContents()).map((s) => s.trim());
    check(!words.includes('king'), '"king" not among its own neighbors');
    const sims = (await p.locator('#lists ol.nn .sim').allTextContents()).map((s) => parseFloat(s));
    const ok = sims.length === 10 && sims.every((s, i) => s > 0 && s <= 1 && (i === 0 || s <= sims[i - 1] + 1e-9));
    check(ok, `similarities descending in (0,1] (${sims.join(', ')})`);
  });

  await step('semantic france', async () => {
    const r = await p.evaluate(() => {
      const res = window.__w2v.nearest('france', 10) || [];
      const words = res.map((x) => (typeof x === 'string' ? x : x.word ?? x.w ?? x[0]));
      const size = window.__w2v.size ?? window.__w2v.vocabSize ?? (window.__w2v.words || []).length;
      return { words, size };
    });
    const hit = ['spain', 'italy', 'germany', 'belgium', 'paris'].some((w) => r.words.includes(w));
    if (hit) pass(`france neighbors are semantic (${r.words.join(', ')})`);
    else if (r.size && r.size < 1000) warn(`france neighbors not semantic, model looks like a dummy (vocab ${r.size})`);
    else fail(`france neighbors not semantic (${r.words.join(', ')})`);
  });

  await step('multi-word + plots', async () => {
    await query('paris, guitar');
    const cards = await p.locator('#lists .card').count();
    check(cards === 2, `"paris, guitar" renders 2 cards (got ${cards})`);
    await p.locator('#plot.js-plotly-plot, #plot .js-plotly-plot').first().waitFor({ timeout: 10000 });
    const t2 = await p.evaluate(() => {
      const gd = document.querySelector('#plot.js-plotly-plot') || document.querySelector('#plot .js-plotly-plot');
      return (gd.data || []).some((d) => d.type === 'scatter' || d.type === 'scattergl');
    });
    check(t2, '2D plot has scatter traces');
    await p.screenshot({ path: path.join(SHOT_DIR, 'w2v-desktop-2d.png'), fullPage: false });
    await p.click('#t3');
    await p.waitForTimeout(1500);
    const t3 = await p.evaluate(() => {
      const gd = document.querySelector('#plot.js-plotly-plot') || document.querySelector('#plot .js-plotly-plot');
      return !!(document.querySelector('#plot .gl-container') || (gd && gd._fullLayout && gd._fullLayout.scene)) &&
        (gd.data || []).some((d) => d.type === 'scatter3d');
    });
    check(t3, '3D tab renders scatter3d scene');
    await p.locator('#plot').scrollIntoViewIfNeeded();
    await p.screenshot({ path: path.join(SHOT_DIR, 'w2v-desktop-3d.png'), fullPage: false });
    await p.click('#t2');
  });

  await step('neighbor click', async () => {
    await query('river');
    const before = await p.inputValue('#q');
    const hashBefore = await p.evaluate(() => location.hash);
    await p.locator('#lists ol.nn button').first().click();
    await p.waitForTimeout(400);
    const after = await p.inputValue('#q');
    const hashAfter = await p.evaluate(() => location.hash);
    check(after && after !== before, `clicking neighbor updates #q ("${before}" → "${after}")`);
    check(hashAfter !== hashBefore && hashAfter.length > 1, `clicking neighbor updates hash (${hashBefore} → ${hashAfter})`);
  });

  await step('analogy', async () => {
    await p.fill('#a1', 'man'); await p.fill('#a2', 'king'); await p.fill('#a3', 'woman');
    await p.locator('#aform').evaluate((f) => f.requestSubmit());
    await p.locator('#aout .pill').first().waitFor({ timeout: 5000 });
    const n = await p.locator('#aout .pill').count();
    check(n >= 1, `analogy yields ${n} answers`);
    const first = (await p.locator('#aout .pill').first().textContent()).trim().toLowerCase();
    if (first.startsWith('queen')) pass('man:king :: woman:queen');
    else warn(`analogy top answer is "${first}", not queen`);
  });

  await step('unknown word', async () => {
    await query('qwzx');
    check(await p.locator('#msg').isVisible(), 'unknown word shows #msg');
  });
  await p.close();

  // ── Mobile ─────────────────────────────────────────────────
  await step('w2v mobile', async () => {
    const m = await browser.newPage({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2 });
    watch(m, 'w2v-mobile');
    await m.goto(BASE + '/word2vec/#king', { waitUntil: 'domcontentloaded' });
    await m.waitForFunction(() => window.__w2v && window.__w2v.ready, null, { timeout: 60000 });
    await m.fill('#q', 'king');
    await m.locator('#qform').evaluate((f) => f.requestSubmit());
    await m.waitForTimeout(800);
    const sw = await m.evaluate(() => document.documentElement.scrollWidth);
    check(sw <= 390, `mobile has no horizontal scroll (scrollWidth ${sw})`);
    await m.screenshot({ path: path.join(SHOT_DIR, 'w2v-mobile.png'), fullPage: true });
    await m.close();
  });
} finally {
  await browser.close();
}

if (errors.length) { errors.forEach((e) => fail(e)); }
else pass('no page or console errors');
console.log(failed ? `\n${failed} check(s) failed` : '\nAll checks passed');
process.exit(failed ? 1 : 0);
