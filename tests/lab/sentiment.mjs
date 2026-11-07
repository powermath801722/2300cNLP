export default async ({ page, check, warn }) => {
  const label = (m) => page.locator(`#res [data-method="${m}"] .lab`).getAttribute('data-label');
  check(await page.locator('#res .card').count() === 4, 'renders four analyzer cards');
  check((await page.locator('#acc tbody tr').count()) === 4, 'held-out accuracy table filled');
  check((await page.locator('#agree tbody tr').count()) === 8, 'agreement table has all 8 examples');

  await page.fill('#txt', 'This was a wonderful, moving and brilliant film.');
  await page.click('#go');
  for (const m of ['vader', 'nb', 'lr']) check(await label(m) === 'pos', `${m} labels a clearly positive sentence positive`);
  await page.fill('#txt', 'A boring, awful, terrible waste of time.');
  await page.click('#go');
  for (const m of ['vader', 'nb', 'lr']) check(await label(m) === 'neg', `${m} labels a clearly negative sentence negative`);

  const neg = await page.evaluate(() => [window.__sent.vader('The food was good.').score, window.__sent.vader('The food was not good.').score]);
  check(neg[0] > 0 && neg[1] < 0, `VADER negation flips sign (${neg.map((x) => x.toFixed(2)).join(' → ')})`);
  check(await page.locator('#res [data-method="lr"] .words span[title]').count() > 0, 'logistic regression shows per-word weights');

  await page.locator('#samples .sample').first().click();
  check((await page.inputValue('#txt')).startsWith('The acting was not good'), 'clicking an example loads and analyzes it');

  await page.click('#tfbtn');
  await page.waitForFunction(() => window.__lab.tf, null, { timeout: 300000 });
  await page.waitForSelector('#res .panel.card[data-method="tf"] .lab', { timeout: 120000 });
  await page.fill('#txt', 'An absolutely delightful movie.');
  await page.click('#go');
  await page.waitForSelector('#res .panel.card[data-method="tf"] .lab', { timeout: 120000 });
  check(await label('tf') === 'pos', 'DistilBERT loads and labels a positive sentence positive');
  check((await page.locator('#agree thead th').count()) === 5, 'agreement table gains a DistilBERT column');
};
