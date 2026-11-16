// Trains the default model (seed 1) in the page's worker and checks that the induction circuit forms.
export default async ({ page, check, warn }) => {
  await page.waitForFunction(() => window.__lab.last && window.__lab.last.step === 0, null, { timeout: 30000 });
  const cfg = await page.evaluate(() => window.__lab.cfg());
  check(cfg.task === 'induction' && cfg.layers === 2 && cfg.heads === 2, 'defaults: induction task, 2 layers × 2 heads');
  check(await page.locator('#heads canvas').count() === 4, 'one attention heatmap per head (4)');

  const t0 = Date.now();
  const done = await page.evaluate(() => window.__lab.trainUntil(1000), null);
  const secs = (Date.now() - t0) / 1000;
  check(done.step === 1000, `trained 1000 steps in the worker (${secs.toFixed(1)} s)`);
  const last = await page.evaluate(() => window.__lab.last);
  const H = cfg.heads;
  const ind = last.scores.induction, prev = last.scores.prevToken;
  const bestInd2 = Math.max(...ind.slice(H)), bestPrev1 = Math.max(...prev.slice(0, H)), bestInd1 = Math.max(...ind.slice(0, H));
  check(bestInd2 > 0.6, `an induction head formed in layer 2 (best score ${bestInd2.toFixed(2)})`);
  check(bestPrev1 > 0.4, `a previous-token head formed in layer 1 (best score ${bestPrev1.toFixed(2)})`);
  check(bestInd1 < 0.3, `layer-1 heads are not induction heads (best ${bestInd1.toFixed(2)})`);
  check(last.recallLoss < last.otherLoss - 1.5, `recallable tokens are predicted far better than others (${last.recallLoss.toFixed(2)} vs ${last.otherLoss.toFixed(2)} nats)`);
  check(last.otherLoss > 2.9, `non-recallable tokens stay near chance (ln 24 = 3.18; got ${last.otherLoss.toFixed(2)})`);

  // the phase change: the recall loss is still high early on
  const hist = await page.evaluate(() => window.__lab.hist());
  const early = hist.recall[hist.evalSteps.indexOf(300)];
  check(early > 2.0, `recall loss was still on the plateau at step 300 (${early.toFixed(2)})`);

  const abl = await page.evaluate(() => window.__lab.ablate());
  const k = ind.indexOf(bestInd2), row = abl.rows.find((r) => r.layer * H + r.head === k);
  check(row.loss - abl.full > 0.4, `ablating the strongest induction head (L${row.layer + 1}H${row.head + 1}) raises recall loss by ${(row.loss - abl.full).toFixed(2)} nats`);
  const kp = prev.indexOf(bestPrev1), rowp = abl.rows.find((r) => r.layer * H + r.head === kp);
  warn(rowp.loss - abl.full > 0.2, `ablating the previous-token head (L${rowp.layer + 1}H${rowp.head + 1}) also hurts (+${(rowp.loss - abl.full).toFixed(2)})`);
  check(await page.locator('#ablt tbody tr').count() === 5, 'ablation table lists the full model and all 4 heads');
  check(await page.locator('#strip .tok').count() === 32, 'example sequence strip shows 32 tokens');

  // reversal task trains quickly to full accuracy
  await page.evaluate(() => window.__lab.setTask('reverse'));
  await page.waitForFunction(() => window.__lab.last && window.__lab.last.step === 0 && window.__lab.cfg().task === 'reverse', null, { timeout: 30000 });
  await page.evaluate(() => window.__lab.trainUntil(200));
  const rev = await page.evaluate(() => window.__lab.last);
  check(rev.tokenAcc > 0.95, `reversal: ${(rev.tokenAcc * 100).toFixed(1)}% of output tokens correct after 200 steps`);
  check(await page.locator('#scorepanel').isHidden(), 'head-score chart is hidden for the reversal task');
};
