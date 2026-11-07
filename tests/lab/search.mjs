export default async ({ page, check, warn }) => {
  const N = await page.evaluate(() => window.__lab.search.N);
  check(N > 1000, `collection loaded (${N} paragraphs)`);
  await page.fill('#q', 'slavery'); await page.click('#bSearch');
  check((await page.locator('#hitsTf li button').count()) === 10 && (await page.locator('#hitsBm li button').count()) === 10, 'both rankers return 10 results');
  const years = await page.$$eval('#hitsBm .hmeta > span:first-child', (s) => s.map((x) => +x.textContent.match(/\d{4}/)[0]));
  warn(years.filter((y) => y >= 1837 && y <= 1900).length >= 5, `slavery results cluster in 1837–1900 (${years.join(', ')})`);
  check((await page.locator('#hitsBm mark').count()) > 0, 'matched terms highlighted');
  // descending scores
  const r = await page.evaluate(() => window.__lab.search.search('foreign entanglements'));
  const desc = (a) => a.every((x, i) => i === 0 || a[i - 1].score >= x.score);
  check(desc(r.tf) && desc(r.bm), 'scores sorted descending');
  // explanation panel
  await page.locator('#hitsTf li button').nth(1).click();
  check(!(await page.isHidden('#explainPanel')) && (await page.locator('#exTable tbody tr').count()) >= 2, 'explain panel shows per-term contributions');
  check((await page.locator('#exKw span').count()) > 3, 'keywords extracted for selected paragraph');
  // BM25 b slider changes scores
  const before = await page.textContent('#hitsBm li:first-child .hmeta');
  await page.fill('#q', 'liberty and union'); await page.click('#bSearch');
  const s0 = await page.evaluate(() => window.__lab.search.search('liberty and union', 1.2, 0.75).bm[0].score);
  const s1 = await page.evaluate(() => window.__lab.search.search('liberty and union', 1.2, 0).bm[0].score);
  check(Math.abs(s0 - s1) > 1e-6, 'BM25 b changes scores');
  await page.locator('#b').fill('0');
  check(/0/.test(await page.textContent('#bo')), 'b slider updates');
  // stopwords
  await page.fill('#q', 'the'); await page.click('#bSearch');
  check(/stopword/i.test(await page.textContent('#qterms')), 'stopword-only query explained');
  await page.uncheck('#stop');
  check((await page.locator('#hitsBm li button').count()) === 10, 'searching for "the" works without stopword removal');
  await page.waitForSelector('#timeline .main-svg', { timeout: 15000 });
  check(true, 'timeline chart renders');
  void before;
};
