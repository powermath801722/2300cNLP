export default async ({ page, check, warn }) => {
  await page.selectOption('#beams', '1');
  await page.click('#run'); // default sample: "The river carries sediment to the sea." → French
  await page.waitForFunction(() => window.__lab.lastMT || window.__lab.runError, null, { timeout: 300000 });
  const err = await page.evaluate(() => window.__lab.runError);
  check(!err, 'model runs' + (err ? ': ' + err : ''));
  const g = await page.evaluate(() => window.__lab.lastMT);
  check(/rivière|fleuve/i.test(g.text) && /mer/i.test(g.text), `greedy French translation is sensible ("${g.text}")`);
  check(typeof g.bleu === 'number' && g.bleu > 0, `computes BLEU against the reference (${g.bleu && g.bleu.toFixed(1)})`);
  check(await page.locator('#outtoks .tok').count() >= 5, 'shows output pieces with probabilities');
  check(await page.locator('#srctoks .tok').count() >= 5, 'shows source SentencePiece pieces');
  // beam search
  await page.evaluate(() => { window.__lab.lastMT = null; });
  await page.selectOption('#beams', '3');
  await page.click('#run');
  await page.waitForFunction(() => window.__lab.lastMT, null, { timeout: 120000 });
  const b = await page.evaluate(() => window.__lab.lastMT);
  check(b.hyps.length === 3 && new Set(b.hyps).size >= 2, `beam 3 returns 3 hypotheses (${JSON.stringify(b.hyps)})`);
  check(!(await page.isVisible('#backbox')), 'round-trip panel stays hidden until back-translation');
  // back-translation
  await page.click('#back');
  await page.waitForFunction(() => window.__lab.lastBack || window.__lab.runError, null, { timeout: 300000 });
  const back = await page.evaluate(() => window.__lab.lastBack || '');
  check(await page.isVisible('#backbox'), 'round-trip panel appears');
  check(/river|stream/i.test(back) && /sea/i.test(back), `back-translation returns to English ("${back}")`);
};
