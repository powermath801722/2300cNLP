// Reference parses from training/parsing_train.py's Python decoder with the same shipped weights (model.json).
const REF = [{"words": ["The", "students", "who", "read", "the", "paper", "asked", "a", "question", "about", "its", "results", "."], "tags": ["DET", "NOUN", "PRON", "VERB", "DET", "NOUN", "VERB", "DET", "NOUN", "ADP", "PRON", "NOUN", "PUNCT"], "heads": [2, 7, 4, 2, 6, 4, 0, 9, 7, 12, 12, 9, 7], "rels": ["det", "nsubj", "nsubj", "acl", "det", "obj", "root", "det", "obj", "case", "nmod", "nmod", "punct"]}, {"words": ["She", "did", "n't", "think", "the", "plan", "would", "work", ",", "but", "it", "did", "."], "tags": ["PRON", "AUX", "PART", "VERB", "DET", "NOUN", "AUX", "VERB", "PUNCT", "CCONJ", "PRON", "AUX", "PUNCT"], "heads": [4, 4, 4, 0, 6, 8, 8, 4, 12, 12, 12, 4, 4], "rels": ["nsubj", "aux", "advmod", "root", "det", "nsubj", "aux", "ccomp", "punct", "cc", "nsubj", "advcl", "punct"]}, {"words": ["Researchers", "in", "Boston", "have", "developed", "a", "cheaper", "test", "for", "the", "virus", "."], "tags": ["NOUN", "ADP", "PROPN", "AUX", "VERB", "DET", "NOUN", "NOUN", "ADP", "DET", "NOUN", "PUNCT"], "heads": [5, 3, 1, 5, 0, 8, 8, 5, 11, 11, 5, 5], "rels": ["nsubj", "case", "nmod", "aux", "root", "det", "compound", "obj", "case", "det", "obl", "punct"]}, {"words": ["I", "'m", "3.5", "miles", "from", "O'Brien's", "caf\u00e9", "\u2014", "wow", "!!"], "tags": ["PRON", "AUX", "NUM", "NOUN", "ADP", "PROPN", "NOUN", "PUNCT", "INTJ", "PUNCT"], "heads": [4, 4, 4, 0, 7, 7, 4, 9, 4, 4], "rels": ["nsubj", "cop", "nummod", "root", "case", "compound", "nmod", "punct", "discourse", "punct"]}];

export default async ({ page, check }) => {
  for (const [i, r] of REF.entries()) {
    const out = await page.evaluate((w) => window.__lab.parse(w), r.words);
    check(JSON.stringify([out.tags, out.heads, out.rels]) === JSON.stringify([r.tags, r.heads, r.rels]), `browser parse matches the Python decoder on sentence ${i + 1}`);
    check(out.steps === 2 * r.words.length, `arc-standard takes exactly 2n transitions (${out.steps} for n=${r.words.length})`);
  }
  check(JSON.stringify(await page.evaluate(() => window.__lab.tokenize("She didn't say it's done; can't won't."))) === JSON.stringify(['She', 'did', "n't", 'say', 'it', "'s", 'done', ';', 'ca', "n't", 'wo', "n't", '.']), 'EWT-style tokenization splits clitics');
  const nWords = await page.locator('#deptable tbody tr').count();
  check(nWords > 5 && (await page.locator('#arcs path.arc').count()) === nWords, `default sentence drawn with one arc per word (${nWords})`);
  check((await page.locator('#evalgrid div').count()) === 5, 'evaluation scores shown');
  const las = await page.evaluate(() => document.querySelector('#evalgrid div:nth-child(2) b').textContent);
  check(parseFloat(las) > 70, `test LAS reported (${las})`);
  // step through: first step = only ROOT on the stack, all words in the buffer
  await page.click('#first');
  check((await page.locator('#stack .tok').count()) === 1 && (await page.locator('#buffer .tok').count()) === nWords && (await page.locator('#arcs path.arc').count()) === 0, 'step 0: ROOT on stack, every word in buffer, no arcs');
  await page.click('#next');
  check((await page.textContent('#action')).length > 0 && (await page.locator('#stack .tok').count()) === 2, 'one SHIFT later the stack holds two items');
  await page.locator('#chips .chip').nth(1).click();
  check((await page.locator('#deptable tbody tr').count()) === 8, 'example chip parses "I saw the man with the telescope."');

  // CKY
  const cky = await page.evaluate(() => window.__lab.cky);
  check(cky.parses.length === 2, `telescope sentence has 2 parses (${cky.parses.length})`);
  check(/\(VP \(VP/.test(cky.parses[0].tree) && Math.abs(cky.parses[0].p / cky.parses[1].p - 2) < 1e-9, 'default grammar prefers VP attachment 2:1');
  check(Math.abs(cky.total - (cky.parses[0].p + cky.parses[1].p)) < 1e-15, 'inside probability equals the sum over trees');
  await page.locator('#cchips .chip').nth(2).click();
  const c3 = await page.evaluate(() => window.__lab.cky.parses.length);
  check(c3 === 5, `two stacked PPs after an object give 5 parses (${c3})`);
  await page.fill('#csent', 'I saw the man with the telescope');
  const g = await page.inputValue('#grammar');
  await page.fill('#grammar', g.replace('VP -> V NP 0.6', 'VP -> V NP 0.9').replace('VP -> VP PP 0.4', 'VP -> VP PP 0.1'));
  await page.waitForTimeout(600);
  const flipped = await page.evaluate(() => window.__lab.cky.parses[0].tree);
  check(/\(NP \(NP/.test(flipped), 'lowering P(VP -> VP PP) flips the best parse to NP attachment');
  await page.fill('#csent', 'I saw the zebra'); await page.click('#ckyform button');
  check(await page.isVisible('#cerr') && /zebra/.test(await page.textContent('#cerr')), 'unknown word explained');
};
