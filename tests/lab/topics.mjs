export default async ({ page, check }) => {
  check(await page.locator('.tcard').count() === 15, 'shows 15 topic cards');
  await page.waitForSelector('#map .main-svg', { timeout: 30000 });
  await page.waitForSelector('#sweep .main-svg', { timeout: 30000 });
  check(true, 'topic map and coherence sweep render');

  const top = async (text) => page.evaluate((t) => { const r = window.__lab.infer(t, 300); return r.theta.indexOf(Math.max(...r.theta)); }, text);
  const oil = await top('Crude oil prices rose after OPEC agreed to cut output; refiners paid more per barrel of crude and gasoline.');
  check(oil === 11, `oil text → Oil & energy topic (got ${oil + 1})`);
  const grain = await top('Wheat and corn exports grew as the USDA raised its grain crop estimate and farmers planted more soybeans.');
  check(grain === 12, `grain text → Grain & agriculture topic (got ${grain + 1})`);
  const sum = await page.evaluate(() => window.__lab.infer('dollar yen exchange rates', 200).theta.reduce((a, b) => a + b, 0));
  check(Math.abs(sum - 1) < 1e-6, `θ sums to 1 (${sum.toFixed(6)})`);

  await page.locator('#chips .chip', { hasText: 'Merger' }).click();
  check((await page.locator('#infwords span').count()) > 5, 'example chip runs inference and colors words');
  check(/Mergers/.test(await page.locator('#inflist div').first().innerText()), 'merger example tops out on the mergers topic');

  await page.locator('.tcard').nth(11).click();
  check((await page.locator('.tcard').nth(11).getAttribute('aria-pressed')) === 'true', 'selecting a topic card marks it pressed');
  check(/Oil/.test(await page.locator('#seltitle').textContent()) && (await page.locator('#topdocs .doc').count()) === 4, 'selected topic lists its top documents');

  await page.selectOption('#docsel', '5');
  check((await page.locator('#doclist > div').count()) === 5, 'document explorer shows the top-5 topic mixture');
  await page.fill('#txt', 'zzzz qqqq');
  await page.click('#infer');
  check(/vocabulary/.test(await page.locator('#infnote').innerText()), 'out-of-vocabulary text gets a helpful message');
};
