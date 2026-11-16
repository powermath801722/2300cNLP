// Speculative decoding: fewer GPT-2 calls than plain decoding, theory roughly matches, output distribution equals the target's.
export default async ({ page, check, warn }) => {
  await page.evaluate(() => { window.__lab.last = null; });
  await page.click('#run');
  await page.waitForFunction(() => window.__lab.last || window.__lab.runError, null, { timeout: 300000 });
  const r = await page.evaluate(() => window.__lab.last || { error: window.__lab.runError });
  check(!r.error, 'speculative run completes' + (r.error ? ': ' + r.error : ''));
  check(r.targetCalls < r.baselineCalls, `fewer GPT-2 calls than plain decoding (${r.targetCalls} vs ${r.baselineCalls})`);
  check(r.tokensPerCall > 1.3, `more than one token per GPT-2 call (${r.tokensPerCall.toFixed(2)})`);
  warn(Math.abs(r.tokensPerCall - r.predicted) < 1, `tokens per call close to (1−α^(γ+1))/(1−α) (${r.tokensPerCall.toFixed(2)} vs ${r.predicted.toFixed(2)})`);
  check(await page.locator('#rounds li').count() === r.targetCalls, 'one round listed per GPT-2 call');
  await page.evaluate(() => { window.__lab.verify = null; });
  await page.click('#verify');
  await page.waitForFunction(() => window.__lab.verify || window.__lab.runError, null, { timeout: 300000 });
  const v = await page.evaluate(() => window.__lab.verify);
  check(v && v.tvSpec <= 2 * v.tvNoise, `speculative distribution within sampling noise of p (TV ${v.tvSpec.toFixed(4)}, noise ≈ ${v.tvNoise.toFixed(4)})`);
  check(v && v.tvQ > 2 * v.tvSpec, `draft-only distribution is far from p (TV ${v.tvQ.toFixed(4)})`);
  check(v && Math.abs(v.acceptEmp - v.acceptTheory) < 0.02, `acceptance rate matches Σ min(p, q) (${v.acceptEmp.toFixed(3)} vs ${v.acceptTheory.toFixed(3)})`);
  // greedy: identical, or diverges only at a quantization near-tie
  await page.evaluate(() => { const t = document.getElementById('temp'); t.value = 0; t.dispatchEvent(new Event('input')); window.__lab.last = null; });
  await page.click('#run');
  await page.waitForFunction(() => window.__lab.last || window.__lab.runError, null, { timeout: 300000 });
  const g = await page.evaluate(() => window.__lab.last);
  check(g.greedyMatch === true || (g.tieGap != null && g.tieGap < 0.5), `greedy matches GPT-2 greedy or diverges only at a near-tie (match ${g.greedyMatch}, first diff ${g.firstDiff}, gap ${g.tieGap})`);
};
