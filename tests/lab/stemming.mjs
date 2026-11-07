export default async ({ page, check }) => {
  const v = await page.evaluate(() => window.__lab.validation);
  check(v && v.mismatches === 0 && v.total > 2000, `Porter matches NLTK on all test words (${v && v.total - v.mismatches}/${v && v.total})`);
  check((await page.locator('#stem').innerText()) === 'gener', 'default word "generalizations" stems to "gener"');
  check((await page.locator('#lemma').innerText()) === 'generalization', 'and lemmatizes to "generalization"');
  check(await page.locator('#trace td.fired').count() >= 3, 'trace shows fired rules');
  const cases = [['hopping', 'hop', 'hop'], ['hoping', 'hope', 'hope'], ['ponies', 'poni', 'pony'], ['went', 'went', 'go'], ['relational', 'relat', 'relational']];
  for (const [w, s, l] of cases) {
    await page.fill('#word', w); await page.click('#wform button[type=submit]');
    const got = await page.evaluate(() => window.__lab.last);
    check(got.stem === s && got.lemma === l, `${w}: stem ${got.stem} (want ${s}), lemma ${got.lemma} (want ${l})`);
  }
  await page.fill('#word', 'meeting'); await page.selectOption('#pos', 'noun');
  check((await page.locator('#lemma').innerText()) === 'meeting', 'POS=noun keeps "meeting"');
  await page.selectOption('#pos', 'verb');
  check((await page.locator('#lemma').innerText()) === 'meet', 'POS=verb gives "meet"');
  const b = await page.evaluate(() => window.__lab.batch);
  check(b.stems < b.lemmas && b.lemmas < b.types, `stems < lemmas < types in batch (${b.stems} < ${b.lemmas} < ${b.types})`);
  check(await page.locator('#over tbody tr').count() >= 5 && await page.locator('#under tbody tr').count() >= 5, 'failure-mode tables filled');
};
