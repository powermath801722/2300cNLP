export default async ({ page, check, warn }) => {
  await page.waitForSelector('#sumTr li');
  check((await page.locator('#sumTr li').count()) === 3, 'default summary has 3 sentences');
  check((await page.locator('#sumLead li').count()) === 3, 'lead-3 baseline shown');
  check((await page.locator('#kw span').count()) >= 5, 'keywords extracted');
  const kw = await page.textContent('#kw');
  warn(/river|delta|sediment/.test(kw), 'delta article keywords mention river/delta/sediment');
  await page.waitForSelector('#graph .main-svg', { timeout: 15000 });
  check(true, 'sentence graph renders');
  // pagerank sanity: converges and in original order
  const r = await page.evaluate(() => { const t = window.__lab.textrank; const x = t.summarize(document.querySelector('#text').value, { k: 4 }); return { picked: x.picked, it: x.trace.length, last: x.trace.at(-1) }; });
  check(r.picked.every((v, i) => i === 0 || r.picked[i - 1] < v), 'summary keeps original sentence order');
  check(r.last < 1e-4 && r.it < 200, `PageRank converged (${r.it} iterations)`);
  // controls
  await page.locator('#k').fill('5');
  await page.waitForFunction(() => document.querySelectorAll('#sumTr li').length === 5);
  check(true, 'summary length slider works');
  await page.selectOption('#sim', 'cosine');
  check((await page.locator('#sumTr li').count()) === 5, 'TF-IDF cosine similarity works');
  await page.selectOption('#sample', 'kennedy');
  await page.waitForFunction(() => /5\d sentences|4\d sentences/.test(document.querySelector('#stat').textContent));
  check(true, 'switching sample re-summarizes');
  await page.locator('#doc .sent').nth(2).click();
  check(/S3:/.test(await page.textContent('#stat')), 'clicking a sentence shows its links');
  // sentence splitter handles abbreviations
  const n = await page.evaluate(() => window.__lab.textrank.splitSentences('Mr. Smith went to Washington. He met Dr. Jones at 5 p.m. on Friday. It rained.').filter((s) => !s.heading).length);
  warn(n === 3, `abbreviation-aware splitting (${n} sentences, expected 3)`);
};
