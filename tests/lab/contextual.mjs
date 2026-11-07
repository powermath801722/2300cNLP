// Contextual module: fill-mask predicts sensible tokens; same-sense contexts are more similar than cross-sense ones.
export default async ({ page, check, warn }) => {
  await page.click('#mlmchips button:has-text("capital of France")');
  await page.waitForFunction(() => window.__lab.mlm && /France/.test(window.__lab.mlm.text), null, { timeout: 300000 });
  const m = await page.evaluate(() => window.__lab.mlm);
  check(m.preds[0].length === 10, 'fill-mask returns 10 predictions');
  warn(m.preds[0].slice(0, 3).some(([w]) => w === 'paris'), `"paris" in the top 3 for the capital of France (${m.preds[0].slice(0, 3).map(([w]) => w).join(', ')})`);
  await page.click('#mlmchips button:has-text("every [MASK]")');
  await page.waitForFunction(() => window.__lab.mlm && /every/.test(window.__lab.mlm.text), null, { timeout: 60000 });
  check(await page.locator('#preds .pred').count() === 2, 'two masks give two prediction cards');
  await page.fill('#mlmin', 'no mask here'); await page.click('#mlmgo');
  check(await page.locator('#mlmmsg').isVisible(), 'missing [MASK] shows a message');

  await page.click('#ctxgo');
  await page.waitForFunction(() => window.__lab.ctx && window.__lab.ctx.target === 'bank', null, { timeout: 300000 });
  const c = await page.evaluate(() => window.__lab.ctx);
  check(c.n === 5, `all five "bank" sentences used (${c.n})`);
  const M = c.M; // S1,S2 river; S3,S4 money
  const within = (M[0][1] + M[2][3]) / 2, across = (M[0][2] + M[0][3] + M[1][2] + M[1][3]) / 4;
  check(within > across, `same-sense contexts are more similar than cross-sense (${within.toFixed(3)} vs ${across.toFixed(3)})`);
  check(Array.isArray(c.w2v) && c.w2v.length === 10, `Word2Vec neighbors shown for the static vector (${c.w2v && c.w2v.slice(0, 4).join(', ')})`);
  check(await page.evaluate(() => document.getElementById('heat').data[0].z.length === 5), 'heatmap is 5×5');
};
