// Watermark: watermarked text is detected (z > 4), plain text is not, human text scores near 0, edits lower z.
export default async ({ page, check, warn }) => {
  await page.evaluate(() => { window.__lab.last = null; });
  await page.click('#gen');
  await page.waitForFunction(() => window.__lab.last || window.__lab.runError, null, { timeout: 300000 });
  const r = await page.evaluate(() => window.__lab.last || { error: window.__lab.runError });
  check(!r.error, 'generation completes' + (r.error ? ': ' + r.error : ''));
  check(r.zWM > 4, `watermarked text detected, z = ${r.zWM && r.zWM.toFixed(2)} > 4`);
  check(r.zPlain < 2, `unwatermarked text not detected, z = ${r.zPlain && r.zPlain.toFixed(2)}`);
  check(r.greenWM > r.greenPlain, `green fraction higher with the watermark (${(r.greenWM * 100).toFixed(0)}% vs ${(r.greenPlain * 100).toFixed(0)}%)`);
  warn(r.pplWM >= r.pplPlain * 0.8, `perplexity cost visible (PPL ${r.pplWM.toFixed(1)} vs ${r.pplPlain.toFixed(1)})`);
  const curve = await page.evaluate(() => window.__lab.curve);
  check(curve && curve.sub[0] > curve.sub.at(-1) + 2, `edit attack lowers z (${curve.sub[0].toFixed(1)} → ${curve.sub.at(-1).toFixed(1)} at ε = 0.6)`);
  await page.evaluate(() => { window.__lab.detect = null; });
  await page.click('#det');
  await page.waitForFunction(() => window.__lab.detect, null, { timeout: 60000 });
  const d = await page.evaluate(() => window.__lab.detect);
  check(Math.abs(d.z) < 2.5, `human text (Austen) scores near 0, z = ${d.z.toFixed(2)}`);
  check(await page.locator('#out-wm .t.g').count() > 10, 'green tokens highlighted in the watermarked output');
};
