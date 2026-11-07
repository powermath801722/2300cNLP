export default async ({ page, check, warn }) => {
  // default sample: abbreviation-aware segmentation and Penn-style clitics
  const last = await page.evaluate(() => window.__lab.last);
  check(last.sentences.length === 8, `abbreviation-aware segmentation finds 8 sentences in the default text (got ${last.sentences.length})`);
  check(last.sentences.some((s) => s.startsWith('Dr. Ada Okafor arrived in St. Louis at 7:45 a.m. on Jan. 3.')), 'Dr./St./a.m./Jan. do not end sentences');
  const toks = await page.evaluate(() => window.__lab.wordTokenize("It isn't $1,299.99 at https://example.com/x?y=1, U.S. 😍 @ada #nlp").map((t) => t.text));
  const want = ['It', 'is', "n't", '$1,299.99', 'at', 'https://example.com/x?y=1', ',', 'U.S.', '😍', '@ada', '#nlp'];
  check(JSON.stringify(toks) === JSON.stringify(want), `word tokenizer output ${JSON.stringify(toks)}`);
  check(await page.locator('#rx .tok').count() > 50, 'word tokens rendered');
  // naive segmentation splits at every period
  await page.check('input[name=sseg][value=naive]');
  const naive = await page.locator('#sents li').count();
  check(naive > 8, `naive segmentation over-splits (${naive} sentences)`);
  // switch sample and stopword toggle
  await page.getByRole('button', { name: 'Social media' }).click();
  await page.waitForTimeout(200);
  const hasEmoji = await page.locator('#rx .tok.t-emoji').count();
  check(hasEmoji >= 3, `emoji kept as tokens (${hasEmoji})`);
  check(await page.locator('#rx .tok.t-url').count() === 1, 'URL kept as one token');
  await page.getByRole('button', { name: /Austen/ }).click();
  await page.waitForTimeout(200);
  const topBefore = await page.locator('#freq tbody tr td:nth-child(2)').first().innerText();
  await page.check('#stop');
  const topAfter = await page.locator('#freq tbody tr td:nth-child(2)').first().innerText();
  check(['and', 'the', 'of', 'her', 'a'].includes(topBefore) && !['and', 'the', 'of', 'her', 'a'].includes(topAfter), `stopword removal changes top word (${topBefore} → ${topAfter})`);
  await page.waitForTimeout(400);
  check(await page.locator('#zipf .main-svg').count() > 0 && await page.locator('#heaps .main-svg').count() > 0, 'Zipf and Heaps plots render');
  const note = await page.locator('#zipfnote').innerText();
  const m = note.match(/r\^(-?\d+(?:\.\d+)?)/);
  warn(m && +m[1] < -0.8 && +m[1] > -1.3, `text8 Zipf slope near −1 (${m && m[1]})`);
};
