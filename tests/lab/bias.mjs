// WEAT module: presets run, statistics are sane, debiasing changes gender WEATs, clustering test runs.
export default async ({ page, check, warn }) => {
  await page.waitForFunction(() => window.__lab && window.__lab.lastWEAT && window.__lab.lastWEAT.p != null, null, { timeout: 120000 });
  const w1 = await page.evaluate(() => window.__lab.lastWEAT);
  check(Number.isFinite(w1.d) && Math.abs(w1.d) <= 2, `WEAT 1 effect size is finite and within [-2, 2] (d=${w1.d.toFixed(2)})`);
  check(w1.p > 0 && w1.p <= 1, `p-value in (0, 1] (p=${w1.p})`);
  warn(w1.d > 0.5, `flowers/insects shows a positive association with pleasant words (d=${w1.d.toFixed(2)})`);
  const bars = await page.evaluate(() => document.getElementById('weatplot').data.reduce((n, t) => n + t.y.length, 0));
  check(bars === w1.nx + w1.ny, `per-word chart has one bar per target word (${bars})`);

  const proj = await page.evaluate(() => window.__lab.projection);
  check(Array.isArray(proj) && proj.length > 20, `occupations projected onto the gender direction (${proj && proj.length})`);
  const p = Object.fromEntries(proj.map((r) => [r.w, r.p]));
  warn(p.nurse > p.engineer, `nurse projects more female-leaning than engineer (${p.nurse?.toFixed(3)} vs ${p.engineer?.toFixed(3)})`);

  // math vs arts × male/female, then with debiasing
  await page.selectOption('#preset', 'math');
  await page.waitForFunction(() => window.__lab.lastWEAT && window.__lab.lastWEAT.nx === 8 && window.__lab.lastWEAT.p != null && !window.__lab.lastWEAT.debias, null, { timeout: 60000 });
  const before = await page.evaluate(() => window.__lab.lastWEAT.d);
  await page.check('#wdebias');
  await page.waitForFunction(() => window.__lab.lastWEAT && window.__lab.lastWEAT.debias && window.__lab.lastWEAT.p != null, null, { timeout: 60000 });
  const after = await page.evaluate(() => window.__lab.lastWEAT.d);
  check(Math.abs(after) < Math.abs(before), `neutralizing shrinks the math/arts gender effect (|d| ${Math.abs(before).toFixed(2)} → ${Math.abs(after).toFixed(2)})`);

  // custom set with an unknown word reports it
  await page.uncheck('#wdebias');
  await page.fill('#setX', 'rose, tulip, daisy, qwzxv');
  await page.click('#run');
  await page.waitForFunction(() => !document.getElementById('missing').hidden && /qwzxv/.test(document.getElementById('missing').textContent), null, { timeout: 30000 });
  check(true, 'unknown words are reported as missing');
  check(await page.inputValue('#preset') === 'custom', 'editing a list switches the preset to Custom');

  await page.click('#gg');
  await page.waitForFunction(() => window.__lab.gg, null, { timeout: 120000 });
  const gg = await page.evaluate(() => window.__lab.gg);
  check(gg.before > 0.5 && gg.after >= 0.5 && gg.after <= 1, `clustering test reports accuracies (before ${(gg.before * 100).toFixed(1)}%, after ${(gg.after * 100).toFixed(1)}%)`);
};
