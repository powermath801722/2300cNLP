// Checks the preference lab's measured outcomes with the default data (seed 1, 400 comparisons per prompt).
export default async ({ page, check, warn }) => {
  const rm = await page.evaluate(() => window.__lab.rm);
  check(rm.accTest > 0.85, `reward model ranks ${(rm.accTest * 100).toFixed(1)}% of held-out pairs like the annotators`);
  check(rm.w[1] > 1.2 && rm.w[2] > 0.8, `reward model learned the annotators' length and flattery bias (w = ${rm.w.map((x) => x.toFixed(2)).join(', ')})`);
  check(rm.goldAgree < rm.accTest, `it agrees less with the gold ranking (${(rm.goldAgree * 100).toFixed(1)}%)`);
  await page.waitForFunction(() => window.__lab.sweep, null, { timeout: 30000 });
  const sw = await page.evaluate(() => window.__lab.sweep);
  const pts = sw.points, last = pts[pts.length - 1];
  check(sw.best.gold > sw.refGold + 0.5, `mild optimization helps: peak gold ${sw.best.gold.toFixed(2)} vs π_ref ${sw.refGold.toFixed(2)} (at β ≈ ${sw.best.beta.toFixed(2)})`);
  check(last.gold < sw.best.gold - 1.5, `over-optimization: gold falls to ${last.gold.toFixed(2)} at β = ${last.beta.toFixed(2)}`);
  let mono = true; for (let i = 1; i < pts.length; i++) if (pts[i].proxy < pts[i - 1].proxy - 1e-9) mono = false;
  check(mono, `proxy reward rises monotonically as β falls (to ${last.proxy.toFixed(2)})`);
  // the slider computes the closed form
  const pol = await page.evaluate(() => window.__lab.setBeta(0.05));
  check(pol.gold < 1, `at β = 0.05 the optimal policy's gold utility is ${pol.gold.toFixed(2)}`);
  check(pol.pi.every((row) => Math.abs(row.reduce((a, b) => a + b, 0) - 1) < 1e-9), 'π* rows sum to 1');
  await page.evaluate(() => window.__lab.setBeta(1.5));
  const al = await page.evaluate(() => window.__lab.runAlign());
  const c = al.curve[al.curve.length - 1];
  check(c.klRL < 1e-6, `RLHF converged to π* ∝ π_ref·exp(r/β) (KL ${c.klRL.toExponential(1)})`);
  check(c.klDPOtab < 1e-6, `DPO equals the closed-form policy for the tabular BT reward on the same pairs (KL ${c.klDPOtab.toExponential(1)})`);
  check(al.gold.rl > al.gold.ref && al.gold.dpo > al.gold.ref, `both improve gold utility at β = 1.5 (π_ref ${al.gold.ref.toFixed(2)} → RLHF ${al.gold.rl.toFixed(2)}, DPO ${al.gold.dpo.toFixed(2)})`);
  warn(c.klDPOrm > 1e-3, `DPO differs from the reward-model policy (KL ${c.klDPOrm.toFixed(3)})`);
  check(await page.locator('#cands tbody tr').count() === 6, 'candidate table shows 6 answers');
};
