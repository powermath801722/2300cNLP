// Expected values from sacreBLEU 2.x sentence_bleu / sentence_chrf (checked offline) for case 1.
const SB = [[6.27, 15.82], [75.06, 88.11], [51.93, 83.11], [100, 100]];

export default async ({ page, check, warn }) => {
  await page.waitForFunction(() => window.__lab.results && window.__lab.results.length === 4);
  const r = await page.evaluate(() => window.__lab.results);
  SB.forEach(([b, c], i) => {
    check(Math.abs(r[i].bleu * 100 - b) < 0.01, `case 1 candidate ${i + 1}: BLEU ${(r[i].bleu * 100).toFixed(2)} matches sacreBLEU ${b}`);
    check(Math.abs(r[i].chrf * 100 - c) < 0.01, `case 1 candidate ${i + 1}: chrF ${(r[i].chrf * 100).toFixed(2)} matches sacreBLEU ${c}`);
  });
  check(r[3].r1 === 1 && r[3].r2 === 1 && r[3].rl === 1, 'exact copy gets ROUGE 100');
  check(r[1].bleu > r[0].bleu && r[2].bleu > r[0].bleu, 'BLEU ranks the wrong date and the negation above the paraphrase');
  await page.selectOption('#smooth', 'none');
  await page.waitForTimeout(300);
  check(await page.evaluate(() => window.__lab.results[0].bleu) === 0, 'unsmoothed BLEU of the paraphrase is 0 (no 4-gram match)');
  await page.selectOption('#smooth', 'exp');
  check(await page.locator('#bleudet tbody tr').count() === 4, 'BLEU breakdown lists n = 1..4');
  // multi-reference: removing the second reference lowers the second candidate
  await page.locator('#cases .chip').nth(2).click(); await page.waitForTimeout(300);
  const both = await page.evaluate(() => window.__lab.results[1].bleu);
  await page.fill('#refs', 'He was born in a small village near the coast.'); await page.click('#score'); await page.waitForTimeout(300);
  const one = await page.evaluate(() => window.__lab.results[1].bleu);
  const c0 = await page.evaluate(() => window.__lab.results[0].bleu);
  check(one < both, `dropping a reference lowers a candidate that matched it (${(both * 100).toFixed(1)} → ${(one * 100).toFixed(1)})`);
  check(c0 < 0.5, `with one reference, the other correct candidate scores low (${(c0 * 100).toFixed(1)})`);
  // BERTScore
  await page.locator('#cases .chip').first().click();
  await page.click('#bert');
  await page.waitForFunction(() => window.__lab.results && window.__lab.results[0].bert != null, null, { timeout: 240000 });
  const rb = await page.evaluate(() => window.__lab.results);
  check(Math.abs(rb[3].bert - 1) < 1e-3, `BERTScore of an exact copy is 1 (${rb[3].bert.toFixed(4)})`);
  check(rb.every((x) => x.bert > 0 && x.bert <= 1.0001), 'BERTScore values in (0, 1]');
  warn(rb[0].bert > rb[0].bleu, 'BERTScore gives the paraphrase more credit than BLEU');
  check(await page.locator('#heat .main-svg').count() > 0, 'token similarity heatmap rendered');
  await page.click('#agreebtn');
  await page.waitForFunction(() => window.__lab.agreement, null, { timeout: 120000 });
  const ag = await page.evaluate(() => window.__lab.agreement);
  check(ag.cols.length === 6 && ag.n === 15, `agreement table over 15 candidates and 6 metrics (${ag.cols.length}, ${ag.n})`);
};
