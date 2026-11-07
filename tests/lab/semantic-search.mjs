// Semantic search: paraphrase query favors dense retrieval, rare-code query favors BM25, evaluation runs.
export default async ({ page, check, warn }) => {
  const info = await page.evaluate(() => ({ docs: window.__lab.docs, dim: window.__lab.dim }));
  check(info.docs === 160 && info.dim === 384, `corpus and embeddings loaded (${info.docs} × ${info.dim})`);
  check(await page.evaluate(() => document.getElementById('map').data.length) >= 16, 'embedding map drawn with one trace per topic');

  await page.click('#chips button:has-text("my pulse is racing")');
  await page.waitForFunction(() => window.__lab.last && /pulse/.test(window.__lab.last.q), null, { timeout: 300000 });
  const a = await page.evaluate(() => window.__lab.last);
  check(a.dense.slice(0, 5).includes('Health-0'), `dense finds the tachycardia passage for a paraphrase (top 5: ${a.dense.join(', ')})`);
  warn(!a.bm25.slice(0, 1).includes('Health-0'), `BM25 misses it at rank 1 (top: ${a.bm25[0] || 'none'})`);
  check(await page.locator('#results .col').count() === 3, 'three result columns (BM25, dense, hybrid)');

  await page.click('#chips button:has-text("HX-4471")');
  await page.waitForFunction(() => window.__lab.last && window.__lab.last.q === 'HX-4471', null, { timeout: 60000 });
  const b = await page.evaluate(() => window.__lab.last);
  check(b.bm25[0] === 'Transport-7', `BM25 finds the exact part number first (${b.bm25[0]})`);
  check(b.hybrid.slice(0, 3).includes('Transport-7'), 'hybrid keeps it in the top 3');

  await page.click('#evalgo');
  await page.waitForFunction(() => window.__lab.eval, null, { timeout: 120000 });
  const e = await page.evaluate(() => window.__lab.eval);
  check(e.Hybrid.r5 >= Math.max(e.BM25.r5, e.Dense.r5) - 0.15, `hybrid recall@5 is competitive (BM25 ${e.BM25.r5.toFixed(2)}, dense ${e.Dense.r5.toFixed(2)}, hybrid ${e.Hybrid.r5.toFixed(2)})`);
  check(await page.locator('#evaltab tbody tr').count() === 10, 'per-query table has 10 rows');
};
