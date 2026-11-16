// Calibration: metrics on the shipped SST-2 split, temperature scaling and selective prediction.
export default async ({ page, check, warn }) => {
  await page.waitForFunction(() => window.__lab.rendered >= 1, null, { timeout: 60000 });
  const R = await page.evaluate(() => window.__cal.results);
  check(R.bert.fullAcc > 0.88 && R.bert.fullAcc < 0.93, `DistilBERT accuracy on 872 SST-2 sentences ≈ 90% (${(R.bert.fullAcc * 100).toFixed(1)}%)`);
  check(R.bert.T > 1, `DistilBERT is overconfident: fitted T > 1 (${R.bert.T.toFixed(2)})`);
  check(R.bert.after.nll < R.bert.before.nll, 'temperature scaling lowers DistilBERT test NLL');
  check(Math.abs(R.bert.after.acc - R.bert.before.acc) < 1e-12, 'temperature scaling leaves accuracy unchanged');
  warn(R.bert.after.ece < R.bert.before.ece, 'temperature scaling lowers DistilBERT ECE');
  check(R.lr.fullAcc < R.bert.fullAcc && R.nb.fullAcc < R.bert.fullAcc, 'movie-review models are less accurate on SST-2 (domain shift)');
  check(R.bert.rc.aurc < R.lr.rc.aurc, 'DistilBERT has the lowest AURC');
  // metric sanity on a toy case
  const toy = await page.evaluate(() => { const P = window.__cal.preds([10, -10, 10], [1, 0, 0]); return window.__cal.metrics(P, 10, 'width'); });
  check(Math.abs(toy.acc - 2 / 3) < 1e-9 && Math.abs(toy.ece - (1 - 2 / 3)) < 1e-3, `ECE of a toy set is |accuracy − confidence| (${toy.ece.toFixed(3)})`);
  check(await page.locator('#rel0 .main-svg').count() > 0 && await page.locator('#rc .main-svg').count() > 0, 'reliability and risk–coverage plots render');
  await page.selectOption('#scheme', 'mass');
  await page.selectOption('#model', 'nb');
  await page.waitForFunction(() => window.__lab.rendered >= 3);
  check(await page.locator('#tbl tr.sel[data-model="nb"]').count() === 1, 'switching model updates the highlighted row');
  // live re-run reproduces precomputed log-odds
  await page.selectOption('#liven', '20');
  await page.click('#live');
  await page.waitForFunction(() => window.__lab.live, null, { timeout: 300000 });
  const L = await page.evaluate(() => window.__lab.live);
  check(L.agree === L.n, `live DistilBERT agrees with the precomputed predictions (${L.agree}/${L.n})`);
  check(L.diff < 1e-3, `live log-odds reproduce the precomputed ones (max |Δ| ${L.diff})`);
};
