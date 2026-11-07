export default async ({ page, check, warn }) => {
  // Edit distance: J&M example with substitution cost 2 → 8; Levenshtein (sub 1) → 5
  check((await page.textContent('#dist')).trim() === '8', 'intention → execution = 8 with substitution cost 2');
  await page.selectOption('#cSub', '1');
  await page.waitForFunction(() => document.querySelector('#dist').textContent.trim() === '5');
  check(true, 'Levenshtein (sub cost 1) intention → execution = 5');
  const cells = await page.locator('table.dp td').count();
  check(cells === 10 * 10, `DP table has (n+1)(m+1) = 100 cells (got ${cells})`);
  check((await page.locator('table.dp td.path').count()) >= 10, 'backtrace path highlighted');
  // Damerau transposition
  await page.fill('#src', 'teh'); await page.fill('#tgt', 'the');
  await page.check('#cTrans'); await page.click('#edform button[type=submit]');
  await page.waitForFunction(() => document.querySelector('#dist').textContent.trim() === '1');
  check(true, 'teh → the = 1 with transposition');
  check((await page.locator('#align .op.T').count()) === 1, 'alignment shows one transposition');
  // Stepping
  await page.click('#bReset'); await page.click('#bStep'); await page.click('#bStep');
  check(/D\[0, 1\]/.test(await page.textContent('#explain')), 'stepping explains the current cell');
  // Corrector
  const c = await page.evaluate(() => ['acress', 'teh', 'lazzy', 'goverment', 'speling'].map((w) => window.__lab.spell.rank(w)[0].w));
  check(c[1] === 'the' && c[2] === 'lazy' && c[3] === 'government' && c[4] === 'spelling', `top corrections: ${c.join(', ')}`);
  warn(['actress', 'across', 'acres', 'access'].includes(c[0]), `acress → ${c[0]}`);
  await page.fill('#sent', 'the qiuck brown fox jumpd ovr teh lazzy dog');
  await page.click('#bCorrect');
  const out = await page.textContent('#corrected');
  check(/quick/.test(out) && /lazy/.test(out), 'sentence correction fixes qiuck and lazzy');
  check(+(await page.getAttribute('#corrected', 'data-changes')) >= 4, 'at least 4 words corrected');
  await page.locator('#corrected .fix').first().click();
  check((await page.locator('#cands tbody tr').count()) > 0, 'clicking a correction lists candidates');
  await page.selectOption('#model', 'norvig');
  check((await page.locator('#cands thead th').count()) === 5, 'Norvig model table');
};
