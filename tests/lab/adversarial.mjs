// Adversarial: attack flips Naive Bayes, neighbours include antonyms, CheckList suite runs.
export default async ({ page, check, warn }) => {
  await page.waitForFunction(() => window.__lab.attacks >= 1, null, { timeout: 60000 });
  const last = await page.evaluate(() => ({ flipped: window.__adv.last.flipped, n: window.__adv.last.subs.length, q: window.__adv.last.queries }));
  check(last.flipped && last.n >= 1 && last.q > 0, `default attack flips Naive Bayes (${last.n} substitutions, ${last.q} queries)`);
  check(await page.locator('#subs tbody tr').count() === last.n, 'substitution table lists every change');
  const nb = await page.evaluate(() => window.__adv.neighbours('good', 10, 0.4).map((x) => [x.w, window.__adv.relation('good', x.w)]));
  check(nb.some(([w, r]) => w === 'bad' && r === 'antonym'), 'Word2Vec neighbours of “good” include its WordNet antonym “bad”');

  await page.selectOption('#bn', '10');
  await page.click('#batch');
  await page.waitForFunction(() => window.__lab.batch, null, { timeout: 120000 });
  const B = await page.evaluate(() => window.__lab.batch);
  check(B.n === 10 && B.success > 0.5, `batch attack on 10 SST-2 sentences succeeds on most (${(B.success * 100).toFixed(0)}%)`);

  await page.click('#cl');
  await page.waitForFunction(() => window.__lab.suite >= 1, null, { timeout: 120000 });
  const S = await page.evaluate(() => window.__adv.suite);
  check(S.nb.length === 11 && S.lr.length === 11, 'CheckList suite runs 11 tests on both instant models');
  const neg = S.nb.find((t) => t.name === 'Negated positive').rate;
  check(neg > 0.5, `bag-of-words Naive Bayes fails most negated-positive cases (${(neg * 100).toFixed(0)}%)`);
  await page.locator('#cltbl tbody tr').nth(3).click();
  check(await page.locator('#clfail h3').count() === 2, 'clicking a test shows failures per model');

  await page.selectOption('#model', 'bert');
  await page.locator('#samples .chip').nth(2).click();
  await page.waitForFunction(() => window.__lab.attacks >= 2, null, { timeout: 300000 });
  const b = await page.evaluate(() => window.__adv.last);
  check(b.queries > 0 && b.subs.length >= 0, `DistilBERT attack runs (${b.queries} queries, flipped: ${b.flipped})`);
  warn(b.flipped, 'attack flips DistilBERT on this example');
};
