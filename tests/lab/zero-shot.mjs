export default async ({ page, check, warn }) => {
  await page.getByRole('button', { name: 'News: match' }).click();
  await page.waitForFunction(() => window.__lab.lastZS || window.__lab.runError, null, { timeout: 300000 });
  const err = await page.evaluate(() => window.__lab.runError);
  check(!err, 'model runs' + (err ? ': ' + err : ''));
  let r = await page.evaluate(() => window.__lab.lastZS);
  check(r[0].label === 'sports', `match report → sports (${r.map((x) => x.label + ' ' + x.score.toFixed(2)).join(', ')})`);
  const sum = r.reduce((a, x) => a + x.score, 0);
  check(Math.abs(sum - 1) < 1e-6, `single-label scores sum to 1 (${sum.toFixed(4)})`);
  check(await page.locator('#pairs tbody tr').count() === 5, 'shows one premise–hypothesis pair per label');
  check(await page.locator('#bars li').count() === 5, 'draws a bar per label');
  // multi-label
  await page.evaluate(() => { window.__lab.lastZS = null; });
  await page.getByRole('button', { name: 'Mixed review' }).click();
  await page.waitForFunction(() => window.__lab.lastZS, null, { timeout: 120000 });
  r = await page.evaluate(() => window.__lab.lastZS);
  const get = (l) => r.find((x) => x.label === l).score;
  warn(get('battery') > 0.5 && get('display') > 0.5, `multi-label: battery and display both high (battery ${get('battery').toFixed(2)}, display ${get('display').toFixed(2)}, price ${get('price').toFixed(2)})`);
  const msum = r.reduce((a, x) => a + x.score, 0);
  check(Math.abs(msum - 1) > 0.01, `multi-label scores are independent, not normalized to 1 (sum ${msum.toFixed(2)})`);
};
