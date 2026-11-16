// End-to-end checks for Braidwater Text Studio (/studio/).
// Usage: python3 -m http.server 8815   (from repo root)
//        node tests/studio.mjs [baseUrl]        (set SKIP_MODELS=1 to skip the NER / distilgpt2 downloads)
import { createRequire } from 'module';
import fs from 'fs';
const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const BASE = (process.argv[2] || 'http://localhost:8815').replace(/\/$/, '');
let failed = 0;
const check = (ok, m) => { console.log(ok ? 'PASS' : 'FAIL', m); if (!ok) failed++; };
const warn = (ok, m) => console.log(ok ? 'PASS' : 'WARN', m);

const IGNORE = [/favicon/i, /GPU stall|GL Driver|WebGL|software WebGL/i, /onnxruntime|VerifyEachNodeIsAssignedToAnEp|Some nodes were not assigned/i];
const errors = [];
function watch(page, tag) {
  page.on('pageerror', (e) => errors.push(`${tag} pageerror: ${e.message}`));
  page.on('console', (m) => { if (m.type() === 'error' && !IGNORE.some((r) => r.test(m.text()))) errors.push(`${tag} console: ${m.text()}`); });
  page.on('response', (r) => { if (r.status() >= 400 && !IGNORE.some((x) => x.test(r.url()))) errors.push(`${tag} HTTP ${r.status()} ${r.url()}`); });
}

const REVIEW_POS = 'I absolutely love this kettle. It is beautiful, fast and quiet, and the temperature presets are wonderful. Customer service was friendly and helpful. Best purchase I have made this year, highly recommended!';

const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, acceptDownloads: true });
  watch(page, 'desktop');
  await page.goto(BASE + '/studio/');
  await page.waitForFunction(() => window.__lab && window.__lab.ready, null, { timeout: 60000 });
  check(true, 'page loads and models are ready');
  check(/Text Studio/.test(await page.title()), 'title mentions Text Studio');
  // the first sample is analyzed automatically
  await page.waitForFunction(() => window.__lab.report() && window.__lab.report().style.spelling, null, { timeout: 30000 });
  check(await page.locator('#doc .sent').count() > 3, 'auto-loaded sample renders clickable sentences');

  // ── news sample: every classic section ──
  await page.locator('[data-sample=news]').click();
  await page.waitForFunction(() => { const r = window.__lab.report(); return r && r.style.spelling && r.headings.length; }, null, { timeout: 30000 });
  let r = await page.evaluate(() => window.__lab.report());
  check(r.timings.totalMs < 1000, `classic analysis under 1 s (${r.timings.totalMs} ms, spelling ${r.timings.spellingMs} ms)`);
  check(!r.sentences.some((s) => /^Riverside, Nov\.$/.test(s.text)), 'abbreviation "Nov." does not end a sentence');
  check(r.headings[0].text.startsWith('Council approves'), 'headline is detected as a heading, not a sentence');
  check(r.overview.words > 150 && r.overview.sentences >= 8, `overview counts (${r.overview.words} words, ${r.overview.sentences} sentences)`);
  check(r.overview.fkgl > 6 && r.overview.fkgl < 18, `Flesch–Kincaid grade plausible for news (${r.overview.fkgl.toFixed(1)})`);
  const posSum = Object.values(r.style.pos.ratios).reduce((a, b) => a + b, 0);
  check(Math.abs(posSum - 1) < 1e-6, `POS ratios sum to 1 (${posSum.toFixed(6)})`);
  check(r.style.pos.ratios.noun > r.style.pos.ratios.adv, 'news text has more nouns than adverbs');
  check(r.style.passives.some((p) => /designed|scheduled|expected/.test(p.phrase)), `passive voice found (${r.style.passives.map((p) => p.phrase).slice(0, 4).join(', ')})`);
  check(r.keyIdeas.summary.length === r.keyIdeas.summaryLength && r.keyIdeas.keywords.length >= 5, `summary has ${r.keyIdeas.summary.length} sentences and ${r.keyIdeas.keywords.length} keywords`);
  check(r.topics.mixture.length === 15 && Math.abs(r.topics.mixture.reduce((a, m) => a + m.p, 0) - 1) < 0.01, 'topic mixture over 15 topics sums to 1');
  check(await page.locator('#overview .statgrid > div').count() === 8, 'overview shows 8 statistics');
  await page.waitForSelector('#arc .main-svg', { timeout: 15000 }).catch(() => {});
  check(await page.locator('#arc .main-svg').count() > 0, 'sentiment arc chart renders');

  // summary length slider → requested count
  await page.evaluate(() => { const s = document.querySelector('#klen'); s.value = '3'; s.dispatchEvent(new Event('change', { bubbles: true })); });
  await page.waitForFunction(() => document.querySelectorAll('#summary li').length === 3, null, { timeout: 10000 });
  check(true, 'summary slider gives exactly 3 sentences');

  // clicking a sentence highlights it across sections + inspector
  const first = await page.locator('#summary li button').first().getAttribute('data-s');
  await page.locator('#summary li button').first().click();
  check(await page.locator(`#doc .sent.active[data-s="${first}"]`).count() === 1, 'clicking a summary sentence highlights it in the document');
  await page.waitForFunction(() => document.querySelectorAll('#inspector .tags span b').length > 3, null, { timeout: 10000 });
  check(await page.locator('#inspector').isVisible(), 'sentence inspector shows POS tags');

  // lenses
  for (const lens of ['Sentiment', 'Summary', 'Passive voice']) {
    await page.locator('#lenses button', { hasText: lens }).click();
    check(await page.locator('#lenses button[aria-pressed=true]').innerText() === lens, `lens “${lens}” toggles`);
  }

  // ── review sample: spelling + sentiment ──
  await page.evaluate((t) => window.__lab.analyze(t), REVIEW_POS);
  r = await page.evaluate(() => window.__lab.report());
  check(r.sentiment.overall.label === 'pos' && r.sentiment.overall.vader.label === 'pos', `clearly positive review is positive (${r.sentiment.overall.agreement} agree)`);
  const easy = await page.evaluate(() => window.__lab.analyze('The cat sat on the mat. The dog ran to the park. We had fun in the sun. It was a good day.'));
  check(easy.overview.fre > 90 && easy.overview.fkgl < 3, `very simple text scores as very easy (FRE ${easy.overview.fre.toFixed(0)}, grade ${easy.overview.fkgl.toFixed(1)})`);

  await page.locator('[data-sample=review]').click();
  await page.waitForFunction(() => { const r = window.__lab.report(); return r && r.style.spelling && r.style.spelling.length >= 3; }, null, { timeout: 30000 });
  r = await page.evaluate(() => window.__lab.report());
  const words = r.style.spelling.map((x) => x.word.toLowerCase());
  check(['recieve', 'definately', 'batery'].every((w) => words.includes(w)), `typos flagged (${words.join(', ')})`);
  const top = Object.fromEntries(r.style.spelling.map((x) => [x.word.toLowerCase(), x.suggestions[0] && x.suggestions[0].w]));
  check(top.recieve === 'receive' && top.definately === 'definitely', `top suggestions (recieve→${top.recieve}, definately→${top.definately}, batery→${top.batery})`);
  // apply a suggestion → text edited and re-analyzed
  await page.locator('#spelllist [data-word=recieve] .fix').first().click();
  await page.waitForFunction(() => { const r = window.__lab.report(); return r && r.style.spelling && !r.style.spelling.some((x) => x.word === 'recieve'); }, null, { timeout: 30000 });
  const txt = await page.inputValue('#txt');
  check(txt.includes('to receive a replacement') && !txt.includes('recieve'), 'clicking a suggestion edits the text and re-analyzes');
  await page.locator('#spelllist [data-word=batery] .ign').click();
  check(await page.locator('#spelllist [data-word=batery]').count() === 0, 'Ignore removes a flagged word');

  // ── exports ──
  const [md] = await Promise.all([page.waitForEvent('download'), page.click('#dl-md')]);
  const mdText = fs.readFileSync(await md.path(), 'utf8');
  check(/\.md$/.test(md.suggestedFilename()) && mdText.includes('# Text Studio report') && mdText.includes('## Summary (TextRank)'), `Markdown download (${md.suggestedFilename()}, ${mdText.length} chars)`);
  const [js] = await Promise.all([page.waitForEvent('download'), page.click('#dl-json')]);
  const obj = JSON.parse(fs.readFileSync(await js.path(), 'utf8'));
  check(obj.tool === 'Braidwater Text Studio' && obj.sentences.length > 0 && obj.overview.words > 0, `JSON download parses (${obj.sentences.length} sentences)`);

  // ── file upload (.md is stripped of markup) ──
  await page.setInputFiles('#file', { name: 'notes.md', mimeType: 'text/markdown', buffer: Buffer.from('# Field notes\n\nThe **river** rose two feet overnight. We moved the boats to [higher ground](http://example.com). Everyone stayed safe and dry.\n') });
  await page.waitForFunction(() => { const r = window.__lab.report(); return r && r.headings.some((h) => h.text === 'Field notes'); }, null, { timeout: 30000 });
  const up = await page.inputValue('#txt');
  check(!/\*\*|\]\(/.test(up) && up.includes('higher ground'), 'Markdown file upload strips markup and analyzes');

  // ── transformer sections ──
  if (!process.env.SKIP_MODELS) {
    await page.evaluate(() => window.__lab.analyze('Ada Lovelace worked with Charles Babbage in London. She wrote notes for the Royal Society about the Analytical Engine, and the University of London later honored her work.'));
    await page.click('#ner-go');
    await page.waitForSelector('#entities .ents', { timeout: 300000 });
    const per = await page.locator('#entities [data-type=PER] li').allInnerTexts();
    const loc = await page.locator('#entities [data-type=LOC] li').allInnerTexts();
    check(per.some((x) => /Ada Lovelace/.test(x)) && loc.some((x) => /London/.test(x)), `NER finds people and places (PER: ${per.join('; ')} | LOC: ${loc.join('; ')})`);
    check(await page.locator('#doc .ent').count() > 0, 'entities lens underlines names in the text');
    await page.click('#sig-go');
    await page.waitForSelector('#toks span', { timeout: 300000 });
    const sig = await page.evaluate(() => window.__lab.exportObject().aiSignals);
    check(sig && sig.perplexity > 1 && sig.analyzedTokens > 10 && sig.buckets.reduce((a, b) => a + b, 0) === sig.analyzedTokens, `AI-text signals (perplexity ${sig && sig.perplexity}, ${sig && sig.analyzedTokens} tokens)`);
    check(await page.locator('#s-signals .caveat').isVisible(), 'AI-text caveat is shown');
    const [md2] = await Promise.all([page.waitForEvent('download'), page.click('#dl-md')]);
    const t2 = fs.readFileSync(await md2.path(), 'utf8');
    check(t2.includes('## Entities') && t2.includes('## AI-text signals') && t2.includes('not proof of authorship'), 'Markdown export includes entities and signals with the caveat');
  } else console.log('SKIP transformer sections (SKIP_MODELS set)');

  // ── mobile ──
  const m = await browser.newPage({ viewport: { width: 390, height: 844 } });
  watch(m, 'mobile');
  await m.goto(BASE + '/studio/');
  await m.waitForFunction(() => window.__lab && window.__lab.ready && window.__lab.report() && window.__lab.report().style.spelling, null, { timeout: 60000 });
  await m.locator('[data-sample=lincoln]').click();
  await m.waitForFunction(() => { const r = window.__lab.report(); return r && r.overview.words > 600 && r.style.spelling; }, null, { timeout: 30000 });
  const sw = await m.evaluate(() => document.documentElement.scrollWidth);
  check(sw <= 390, `no horizontal scroll at 390px (scrollWidth ${sw})`);
  check(await m.locator('#snav').isVisible(), 'section nav visible on mobile');
} catch (e) {
  check(false, 'threw: ' + e.message.split('\n')[0]);
} finally {
  for (const e of errors) check(false, e);
  await browser.close();
}
console.log(failed ? `\n${failed} check(s) failed` : '\nAll checks passed');
process.exit(failed ? 1 : 0);
