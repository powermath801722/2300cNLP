// Surprisal module: real distilgpt2 surprisal (≈85 MB download on first run).
export default async ({ page, check, warn }) => {
  page.setDefaultTimeout(300000);
  const gp = await page.evaluate(() => window.__lab.analyze('The horse raced past the barn fell.', 'Xenova/distilgpt2'));
  const ctl = await page.evaluate(() => window.__lab.analyze('The horse that was raced past the barn fell.', 'Xenova/distilgpt2'));
  const fell = (r) => r.tokens.find((t) => t.text === ' fell');
  check(gp.tokens.length === 8 && fell(gp), `garden path tokenized into ${gp.tokens.length} tokens`);
  check(gp.tokens.every((t) => t.bits >= 0 && Number.isFinite(t.bits) && t.rank >= 1), 'surprisal values are finite and non-negative');
  const peak = gp.tokens.reduce((a, t) => (t.bits > a.bits ? t : a));
  check(peak.text === ' fell', `garden-path peak is " fell" (got ${JSON.stringify(peak.text)}, ${peak.bits.toFixed(1)} bits)`);
  check(fell(gp).bits > fell(ctl).bits + 2, `" fell" is more surprising in the garden path (${fell(gp).bits.toFixed(1)} vs ${fell(ctl).bits.toFixed(1)} bits)`);
  const ppl = Math.pow(2, gp.tokens.reduce((s, t) => s + t.bits, 0) / gp.tokens.length);
  check(Math.abs(ppl - gp.ppl) < 1e-6, 'perplexity = 2^(mean surprisal)');

  // UI: human vs machine samples, logged in the runs table; GLTR view.
  await page.locator('#samples .chip', { hasText: 'Human' }).click();
  await page.waitForFunction(() => document.querySelectorAll('#runs tbody tr').length >= 1);
  const human = await page.evaluate(() => window.__lab.last);
  await page.locator('#samples .chip', { hasText: 'Machine' }).click();
  await page.waitForFunction(() => document.querySelectorAll('#runs tbody tr').length >= 2);
  const machine = await page.evaluate(() => window.__lab.last);
  const top10 = (r) => r.buckets[0] / r.tokens.length;
  check(top10(machine) > top10(human), `greedy text has more top-10 tokens (${(top10(machine) * 100).toFixed(0)}% vs ${(top10(human) * 100).toFixed(0)}%)`);
  warn(machine.ppl < human.ppl, `greedy text perplexity ${machine.ppl.toFixed(1)} < human ${human.ppl.toFixed(1)}`);
  await page.click('#v-r');
  const n = await page.locator('#tokens button.tk').count();
  check(n === machine.tokens.length, `GLTR view colors all ${n} tokens`);
  await page.locator('#tokens button.tk').nth(3).click();
  check((await page.locator('#detail ol li').count()) === 5, 'token detail shows top-5 predictions');
  check(await page.locator('#line .main-svg').count() > 0 && await page.locator('#hist .main-svg').count() > 0, 'line chart and rank histogram render');
};
