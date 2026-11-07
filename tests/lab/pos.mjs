export default async ({ page, check }) => {
  const tag = (s) => page.evaluate((x) => window.__lab.tag(x), s);
  check(await page.locator('#tagged .tw').count() > 5, 'tags the default sentence');
  check(await page.locator('#trellis td.best').count() === await page.locator('#tagged .tw').count(), 'trellis highlights exactly one best cell per word');
  check(await page.locator('#stats div').count() === 5 && await page.locator('#conf tbody tr').count() > 5, 'evaluation stats and confusions shown');

  const a = await tag('The company said its profits rose sharply in the third quarter .');
  check(a.join(' ') === 'DT NN VBD PRP$ NNS VBD RB IN DT JJ NN .', `tags a WSJ-style sentence correctly (${a.join(' ')})`);
  // Same output as the Python Viterbi in training/pos_train.py
  check((await tag('Time flies like an arrow .')).join(' ') === 'NNP NNS IN DT NN .', 'matches the Python decoder on "Time flies like an arrow"');
  const u = await tag('The zorbling quickly blicketed the grommets .');
  check(u[2] === 'RB' && u[3] === 'VBD' && u[5] === 'NNS', `unknown-word classes work (${u.join(' ')})`);
  check((await tag('Supercalifragilisticexpialidocious !')).length === 2, 'decodes a sentence of unseen words without crashing');

  await page.locator('#chips .chip').first().click();
  check((await page.locator('#tagged .tw').count()) === 6, 'clicking an example re-tags');
  await page.fill('#sent', 'Researchers are studying how children learn grammar .');
  await page.press('#sent', 'Enter');
  check((await page.locator('#tagged .tw').first().getAttribute('data-tag')) === 'NNS', 'Enter submits and tags the new sentence');
};
