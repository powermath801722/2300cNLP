export default async ({ page, check, warn }) => {
  await page.selectOption('#preset', '1'); // Rosetta Stone
  await page.getByRole('button', { name: 'Who deciphered the hieroglyphs?' }).click();
  await page.waitForFunction(() => window.__lab.lastAnswer || window.__lab.runError, null, { timeout: 300000 });
  const err = await page.evaluate(() => window.__lab.runError);
  check(!err, 'model runs' + (err ? ': ' + err : ''));
  const a = await page.evaluate(() => window.__lab.lastAnswer);
  check(/Champollion/.test(a.answer), `answers "Champollion" (got "${a.answer}", p=${a.prob.toFixed(3)})`);
  check(await page.locator('#passage-view mark').count() === 1, 'highlights the span in the passage');
  check(await page.locator('#cands tbody tr').count() >= 3, 'lists candidate spans');
  check((await page.textContent('#gold')).includes('Exact match'), 'shows EM/F1 against gold');
  const chart = await page.waitForSelector('#chart .main-svg', { timeout: 30000 }).then(() => true, () => false);
  check(chart, 'draws start/end probability chart');
  // unanswerable question with a high threshold should abstain
  await page.evaluate(() => { window.__lab.lastAnswer = null; });
  await page.fill('#thr', '0.9'); await page.dispatchEvent('#thr', 'input');
  await page.getByRole('button', { name: 'Who carved the inscription on the stone?' }).click();
  await page.waitForFunction(() => window.__lab.lastAnswer, null, { timeout: 120000 });
  const u = await page.evaluate(() => window.__lab.lastAnswer);
  warn(u.abstained, `abstains on an unanswerable question at threshold 0.9 (p=${u.prob.toFixed(3)})`);
  check((await page.textContent('#gold')).includes('unanswerable'), 'explains the unanswerable gold label');
};
