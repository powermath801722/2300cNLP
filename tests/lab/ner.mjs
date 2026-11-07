export default async ({ page, check }) => {
  await page.fill('#text', 'Ada Lovelace worked in London for the Royal Society.');
  await page.click('#run');
  await page.waitForFunction(() => window.__lab.lastEntities || window.__lab.runError, null, { timeout: 300000 });
  const ents = await page.evaluate(() => window.__lab.lastEntities || []);
  const err = await page.evaluate(() => window.__lab.runError);
  check(!err, 'model runs' + (err ? ': ' + err : ''));
  check(ents.some((e) => e.type === 'PER' && /Ada Lovelace/.test(e.text)), `finds PER "Ada Lovelace" (${JSON.stringify(ents)})`);
  check(ents.some((e) => e.type === 'LOC' && e.text === 'London'), 'finds LOC "London"');
  check(ents.some((e) => e.type === 'ORG' && /Royal Society/.test(e.text)), 'finds ORG "Royal Society"');
  check(await page.locator('#inline mark.ent').count() >= 3, 'highlights entities inline');
  check(await page.locator('#tokens .tok').count() >= 10, 'shows per-WordPiece BIO tags');
  // Sample chip runs end-to-end and shows its note
  await page.evaluate(() => { window.__lab.lastEntities = null; });
  await page.getByRole('button', { name: 'Which Washington?' }).click();
  await page.waitForFunction(() => window.__lab.lastEntities, null, { timeout: 120000 });
  check((await page.textContent('#sample-note')).includes('Ambiguity'), 'sample chip shows its explanation');
};
