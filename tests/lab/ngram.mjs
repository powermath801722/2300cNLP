export default async ({ page, check, warn }) => {
  const s = await page.evaluate(() => window.__lab.stats);
  check(s && s.trainTokens > 100000 && s.trigrams > 50000, `Austen trigram model trained (${s && s.trainTokens} tokens, ${s && s.trigrams} trigram types, ${s && s.ms} ms)`);
  const ppl = await page.evaluate(() => window.__lab.perplexity);
  const kn3 = ppl.kn[2].ppl, kn1 = ppl.kn[0].ppl, add1 = ppl.addk[2].ppl;
  check(ppl.mle[2].ppl === 'Infinity' && ppl.mle[2].zeros > 0, 'MLE trigram perplexity is infinite on held-out text');
  check(kn3 < kn1, `Kneser–Ney trigram beats unigram (${kn3.toFixed(1)} < ${kn1.toFixed(1)})`);
  check(kn3 < ppl.interp[2].ppl, `KN trigram beats fixed-λ interpolation (${kn3.toFixed(1)} vs ${ppl.interp[2].ppl.toFixed(1)})`);
  warn(kn3 < add1, `KN beats add-k trigram (${kn3.toFixed(1)} vs ${add1.toFixed(1)})`);
  // probabilities sum to 1 for normalized methods
  for (const m of ['kn', 'addk', 'interp', 'mle']) {
    const r = await page.evaluate((m) => window.__lab.call('next', { context: 'she was not', order: 3, method: m, p: { k: 0.01, D: 0.75, lambdas: { 1: [1, 0, 0], 2: [0.3, 0.7, 0], 3: [0.1, 0.3, 0.6] } } }), m);
    check(Math.abs(r.mass - 1) < 1e-6, `${m}: next-word distribution sums to 1 (${r.mass.toFixed(8)})`);
  }
  check(await page.locator('#next tbody tr').count() >= 10, 'next-word table filled');
  const gen = await page.evaluate(() => window.__lab.generated);
  check(gen.words.length >= 2, `generation produced text (${gen.words.join(' ')})`);
  await page.selectOption('#method', 'mle');
  await page.fill('#sent', 'she had never seen a purple elephant dancing .');
  await page.click('#sform button');
  await page.waitForFunction(() => window.__lab.score && !Number.isFinite(window.__lab.score.log2));
  check(true, 'MLE gives an odd sentence probability 0');
  await page.selectOption('#method', 'kn');
  await page.click('#sform button');
  await page.waitForFunction(() => window.__lab.score && Number.isFinite(window.__lab.score.log2));
  check(await page.locator('#stab tbody tr').count() === 10, 'KN scores every token (9 words + </s>)');
  // switch corpus retrains
  await page.selectOption('#corpus', 'carroll');
  await page.waitForFunction(() => document.querySelector('#status .t').textContent.startsWith('Trained'), null, { timeout: 30000 });
  await page.waitForTimeout(300);
  const st2 = await page.locator('#mstats').innerText();
  check(/2[0-9],[0-9]{3}\s*training tokens/.test(st2.replace(/\n/g, ' ')), 'switching to Carroll retrains on the smaller corpus');
  check(await page.locator('#pplot .main-svg').count() > 0, 'perplexity chart renders');
};
