// Decoding strategies module: drives real distilgpt2 decoding (≈85 MB download on first run).
export default async ({ page, check, warn }) => {
  page.setDefaultTimeout(300000);
  // Greedy is deterministic; seeded nucleus sampling must reproduce exactly.
  const g1 = await page.evaluate(() => window.__lab.decode({ strategy: 'greedy', maxnew: 12, reppen: 1, prompt: 'The old lighthouse keeper opened the door and' }));
  check(g1.tokens.length === 12 || g1.tokens.includes(50256), `greedy produced ${g1.tokens.length} tokens`);
  const g2 = await page.evaluate(() => window.__lab.decode({ strategy: 'greedy', maxnew: 12, reppen: 1, prompt: 'The old lighthouse keeper opened the door and' }));
  check(g1.text === g2.text, 'greedy decoding is deterministic');
  const s1 = await page.evaluate(() => window.__lab.decode({ strategy: 'topp', topp: 0.9, temp: 1, seed: 7, maxnew: 12 }));
  const s2 = await page.evaluate(() => window.__lab.decode({ strategy: 'topp', topp: 0.9, temp: 1, seed: 7, maxnew: 12 }));
  const s3 = await page.evaluate(() => window.__lab.decode({ strategy: 'topp', topp: 0.9, temp: 1, seed: 8, maxnew: 12 }));
  check(s1.text === s2.text, 'same seed reproduces nucleus sample');
  warn(s1.text !== s3.text, 'different seed gives a different sample');
  // Greedy always picks the argmax, so its chosen-token probabilities should beat sampled ones on average.
  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  warn(mean(g1.probs) > mean(s1.probs), `greedy mean P(chosen) ${mean(g1.probs).toFixed(3)} > nucleus ${mean(s1.probs).toFixed(3)}`);
  const b = await page.evaluate(() => window.__lab.decode({ strategy: 'beam', beams: 3, maxnew: 8 }));
  check(b.tokens.length > 0, `beam search produced "${b.text.replace(/\n/g, '↵')}"`);

  // UI: top-k run, then inspect a step: candidates listed, and with k=3 truncation must cut candidates.
  await page.selectOption('#strategy', 'topk');
  await page.locator('#topk').fill('3');
  await page.locator('#maxnew').fill('6');
  await page.click('#gen');
  await page.waitForFunction(() => window.__lab.last && window.__lab.last.strategy === 'topk');
  const n = await page.locator('#out button.tk').count();
  check(n >= 1, `UI rendered ${n} generated tokens`);
  await page.locator('#out button.tk').first().click();
  const cands = await page.locator('#step .cands li:not([aria-hidden])').count();
  const cut = await page.locator('#step .cands li.cut').count();
  check(cands >= 10, `step inspector lists ${cands} candidates`);
  check(cut >= 7, `top-k=3 marks ${cut} of the top-10 as cut`);
  check(await page.locator('#metrics div').count() === 5, 'metrics panel shows 5 statistics');

  // Comparison mode: three cards with stats.
  await page.locator('#maxnew').fill('10');
  await page.click('#cmp');
  await page.waitForFunction(() => window.__lab.compare && window.__lab.compare.length === 3);
  const cmp = await page.evaluate(() => window.__lab.compare);
  check(cmp.length === 3 && cmp.every((r) => r.tokens > 0), 'compare runs greedy, beam and nucleus');
  warn(cmp[0].meanLog2 > cmp[2].meanLog2, `greedy log-prob ${cmp[0].meanLog2.toFixed(2)} > nucleus ${cmp[2].meanLog2.toFixed(2)} (probability–quality paradox)`);
};
