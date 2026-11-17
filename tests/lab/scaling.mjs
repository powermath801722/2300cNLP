// Checks the scaling-law explorer's numbers against the closed form and the papers' headline results.
export default async ({ page, check, warn }) => {
  const b = await page.evaluate(() => window.__lab.budget);
  // default budget = Gopher/Chinchilla compute (6 × 280e9 × 300e9 ≈ 5.04e23)
  check(Math.abs(Math.log10(b.C) - Math.log10(5.04e23)) < 0.01, `default budget is Gopher's compute (${b.C.toExponential(2)} FLOPs)`);
  check(Math.abs(b.besiroglu.ratio - 20) < 4, `Besiroglu refit implies ~20 tokens/parameter at that budget (${b.besiroglu.ratio.toFixed(1)})`);
  check(b.besiroglu.N > 50e9 && b.besiroglu.N < 90e9, `…and N* near Chinchilla's 70B (${(b.besiroglu.N / 1e9).toFixed(1)}B)`);
  check(b.hoffmann.N < b.besiroglu.N && b.hoffmann.ratio > 40, `Hoffmann Approach 3 fit favours a smaller model with more tokens (${(b.hoffmann.N / 1e9).toFixed(1)}B, ${b.hoffmann.ratio.toFixed(0)} tokens/param)`);
  check(Math.abs(b.hoffmann.a - 0.2849 / (0.3392 + 0.2849)) < 1e-9 && Math.abs(b.besiroglu.a - 0.3658 / (0.3478 + 0.3658)) < 1e-9, `allocation exponents a = β/(α+β): ${b.hoffmann.a.toFixed(3)} and ${b.besiroglu.a.toFixed(3)}`);
  // the closed-form optimum really is the minimum along the iso-FLOP line
  const ok = await page.evaluate(() => {
    const { FITS, optimal, loss } = window.__lab.laws; const C = 1e23; let fine = true;
    for (const f of Object.values(FITS)) { const o = optimal(f, C); for (const m of [0.5, 0.8, 1.25, 2]) { const N = o.N * m; if (loss(f, N, C / (6 * N)) < o.L - 1e-12) fine = false; } }
    return fine;
  });
  check(ok, 'closed-form optimum beats nearby allocations on the same iso-FLOP line');
  const rows = b.rows;
  check(rows[3].lb > rows[1].lb, `N ∝ C^0.73 allocation through GPT-3 is worse than optimal at this budget (${rows[3].lb.toFixed(3)} vs ${rows[1].lb.toFixed(3)})`);
  check(Math.abs(rows[3].N / 175e9 - Math.pow(b.C / (6 * 175e9 * 300e9), 0.73)) < 1e-9, 'Kaplan-exponent line passes through GPT-3 with slope 0.73');
  // calculator: Chinchilla itself sits at the optimum of the Besiroglu fit within a few percent
  const c = await page.evaluate(() => window.__lab.setCalc(70, 1400, 2.0));
  check(c.efficiency > 0.97 && c.efficiency <= 1.0000001, `Chinchilla (70B, 1.4T) needs ${(c.efficiency * 100).toFixed(1)}% of its compute under the refit, i.e. near-optimal`);
  const g = await page.evaluate(() => window.__lab.setCalc(175, 300, 2.0));
  check(g.efficiency < 0.7, `GPT-3 (175B, 300B) could reach its predicted loss with ${(g.efficiency * 100).toFixed(0)}% of the compute`);
  const u = await page.evaluate(() => window.__lab.setCalc(7, 2000, 1.5));
  check(!isFinite(u.needTarget), 'a target below the irreducible loss is reported unreachable');
  await page.waitForSelector('#map .main-svg', { timeout: 30000 }).catch(() => {});
  check(await page.locator('#map .main-svg').count() > 0, 'contour map rendered');
  check(await page.locator('#opttable tbody tr').count() === 4, 'allocation table has 4 rows');
  const b2 = await page.evaluate(() => window.__lab.setLogC(26));
  check(b2.besiroglu.N > b.besiroglu.N * 10, `more compute → bigger optimal model (${(b2.besiroglu.N / 1e9).toFixed(0)}B at 10^26 FLOPs)`);
};
